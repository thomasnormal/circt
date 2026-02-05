//===- LowerLLHDRefPorts.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_LOWERLLHDREFPORTS
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {
struct OutputRefConversion {
  unsigned oldOutputIndex = 0;
  unsigned oldPortIndex = 0;
  Value refValue;
  bool dropInternalDrives = false;
};

struct InputRefConversion {
  unsigned oldInputIndex = 0;
  unsigned oldPortIndex = 0;
  Value internalSignal;
};

struct ModuleConversionInfo {
  hw::HWModuleOp module;
  hw::ModuleType oldType;
  SmallVector<bool> inputDrivePorts;
  SmallVector<bool> externallyDrivenOutputs;
  SmallVector<unsigned> keptInputs;
  SmallVector<InputRefConversion> inputsToOutputs;
  SmallVector<OutputRefConversion> outputsToInputs;
  SmallVector<unsigned> keptOutputs;
  bool hasRefPorts = false;
};

struct LowerLLHDRefPortsPass
    : public circt::impl::LowerLLHDRefPortsBase<LowerLLHDRefPortsPass> {
  using LowerLLHDRefPortsBase::LowerLLHDRefPortsBase;
  void runOnOperation() override;
};
} // namespace

static bool hasInternalDrive(Value refValue) {
  for (Operation *user : refValue.getUsers()) {
    auto drive = dyn_cast<llhd::DriveOp>(user);
    if (!drive)
      continue;
    if (drive.getOperand(0) == refValue)
      return true;
  }
  return false;
}

static bool isRefType(Type type) { return isa<llhd::RefType>(type); }

static Type getRefNestedType(Type type) {
  return cast<llhd::RefType>(type).getNestedType();
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  int64_t width = hw::getBitWidth(type);
  if (width <= 0)
    return Value();
  auto intType = IntegerType::get(builder.getContext(), width);
  Value zero = hw::ConstantOp::create(builder, loc, intType, 0);
  if (intType == type)
    return zero;
  return hw::BitcastOp::create(builder, loc, type, zero);
}

static void replaceInvalidProbes(hw::HWModuleOp module) {
  SmallVector<llhd::ProbeOp> probes;
  module.walk([&](llhd::ProbeOp probe) { probes.push_back(probe); });
  for (auto probe : probes) {
    Value signal = probe.getOperand();
    if (isRefType(signal.getType()))
      continue;
    probe.getResult().replaceAllUsesWith(signal);
    probe.erase();
  }
}

static LogicalResult replaceRefUsesWithInput(Value refValue, Value newInput,
                                             bool dropInternalDrives) {
  SmallVector<Operation *> users;
  users.reserve(4);
  for (Operation *user : refValue.getUsers())
    users.push_back(user);

  SmallVector<Operation *> eraseOps;
  for (Operation *user : users) {
    if (auto probe = dyn_cast<llhd::ProbeOp>(user)) {
      probe.getResult().replaceAllUsesWith(newInput);
      eraseOps.push_back(probe);
      continue;
    }
    if (auto drive = dyn_cast<llhd::DriveOp>(user)) {
      if (drive.getSignal() != refValue)
        return drive.emitError("unexpected drive target during ref lowering");
      if (!dropInternalDrives)
        return drive.emitError("unexpected internal drive on externally driven "
                               "ref port");
      eraseOps.push_back(drive);
      continue;
    }
  }

  for (auto *op : eraseOps)
    op->erase();
  refValue.replaceAllUsesWith(newInput);
  return success();
}

void LowerLLHDRefPortsPass::runOnOperation() {
  auto top = getOperation();
  auto *context = top.getContext();
  SymbolTable symTable(top);

  DenseMap<StringAttr, ModuleConversionInfo> moduleInfo;
  SmallVector<hw::HWModuleOp> modules;
  top.walk([&](hw::HWModuleOp module) { modules.push_back(module); });

  for (auto module : modules) {
    ModuleConversionInfo info;
    info.module = module;
    info.oldType = module.getModuleType();
    info.inputDrivePorts.assign(info.oldType.getNumInputs(), false);
    info.externallyDrivenOutputs.assign(info.oldType.getNumOutputs(), false);
    Block *body = module.getBodyBlock();
    auto ports = info.oldType.getPorts();
    for (unsigned i = 0, e = info.oldType.getNumInputs(); i < e; ++i) {
      if (!isRefType(ports[i].type))
        continue;
      info.hasRefPorts = true;
      if (hasInternalDrive(body->getArgument(i)))
        info.inputDrivePorts[i] = true;
    }
    moduleInfo[module.getNameAttr()] = std::move(info);
  }

  SmallVector<hw::InstanceOp> instances;
  top.walk([&](hw::InstanceOp inst) { instances.push_back(inst); });

  // Propagate input drive ports across instance connections.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto inst : instances) {
      auto moduleIt = moduleInfo.find(inst.getReferencedModuleNameAttr());
      if (moduleIt == moduleInfo.end())
        continue;
      auto &calleeInfo = moduleIt->second;
      if (calleeInfo.inputDrivePorts.empty())
        continue;

      auto parentModule = inst->getParentOfType<hw::HWModuleOp>();
      auto parentIt = moduleInfo.find(parentModule.getNameAttr());
      if (parentIt == moduleInfo.end())
        continue;
      auto &parentInfo = parentIt->second;

      for (unsigned i = 0, e = calleeInfo.inputDrivePorts.size(); i < e; ++i) {
        if (!calleeInfo.inputDrivePorts[i])
          continue;
        Value operand = inst.getOperand(i);
        auto arg = dyn_cast<BlockArgument>(operand);
        if (!arg || arg.getOwner() != parentModule.getBodyBlock())
          continue;
        unsigned argIndex = arg.getArgNumber();
        if (argIndex >= parentInfo.inputDrivePorts.size())
          continue;
        if (!isRefType(parentInfo.oldType.getInputType(argIndex)))
          continue;
        if (!parentInfo.inputDrivePorts[argIndex]) {
          parentInfo.inputDrivePorts[argIndex] = true;
          changed = true;
        }
      }
    }
  }

  // Mark externally-driven instance outputs.
  for (auto inst : instances) {
    auto moduleIt = moduleInfo.find(inst.getReferencedModuleNameAttr());
    if (moduleIt == moduleInfo.end())
      continue;
    auto &info = moduleIt->second;
    for (auto result : inst.getResults()) {
      if (!isRefType(result.getType()))
        continue;
      for (Operation *user : result.getUsers()) {
        auto drive = dyn_cast<llhd::DriveOp>(user);
        if (!drive || drive.getOperand(0) != result)
          continue;
        unsigned outIdx = cast<OpResult>(result).getResultNumber();
        if (outIdx < info.externallyDrivenOutputs.size())
          info.externallyDrivenOutputs[outIdx] = true;
        break;
      }
    }

    for (unsigned i = 0, e = info.inputDrivePorts.size(); i < e; ++i) {
      if (!info.inputDrivePorts[i])
        continue;
      Value operand = inst.getOperand(i);
      auto result = dyn_cast<OpResult>(operand);
      if (!result)
        continue;
      auto producerInst = dyn_cast<hw::InstanceOp>(result.getOwner());
      if (!producerInst)
        continue;
      auto producerIt =
          moduleInfo.find(producerInst.getReferencedModuleNameAttr());
      if (producerIt == moduleInfo.end())
        continue;
      auto &producerInfo = producerIt->second;
      unsigned outIdx = result.getResultNumber();
      if (outIdx < producerInfo.externallyDrivenOutputs.size())
        producerInfo.externallyDrivenOutputs[outIdx] = true;
    }
  }

  // Analyze module ports now that external drive information is available.
  for (auto &entry : moduleInfo) {
    auto &info = entry.second;
    auto module = info.module;
    auto outputOp =
        dyn_cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    if (!outputOp)
      continue;

    auto ports = info.oldType.getPorts();
    unsigned numInputs = info.oldType.getNumInputs();
    unsigned numOutputs = info.oldType.getNumOutputs();

    info.keptInputs.reserve(numInputs);
    info.inputsToOutputs.reserve(numInputs);
    for (unsigned inIdx = 0; inIdx < numInputs; ++inIdx) {
      if (isRefType(ports[inIdx].type))
        info.hasRefPorts = true;
      if (inIdx < info.inputDrivePorts.size() && info.inputDrivePorts[inIdx]) {
        InputRefConversion conv;
        conv.oldInputIndex = inIdx;
        conv.oldPortIndex = inIdx;
        info.inputsToOutputs.push_back(conv);
      } else {
        info.keptInputs.push_back(inIdx);
      }
    }

    info.keptOutputs.reserve(numOutputs);
    for (unsigned outIdx = 0; outIdx < numOutputs; ++outIdx) {
      unsigned portIndex = numInputs + outIdx;
      auto portType = ports[portIndex].type;
      if (!isRefType(portType)) {
        info.keptOutputs.push_back(outIdx);
        continue;
      }
      info.hasRefPorts = true;
      Value outVal = outputOp.getOperand(outIdx);
      bool internalDrive = hasInternalDrive(outVal);
      bool externalDrive = info.externallyDrivenOutputs[outIdx];
      if (internalDrive && externalDrive) {
        if (this->strict) {
          module.emitError(
              "externally driven llhd.ref output also driven internally; "
              "rerun without --strict-llhd to allow abstraction");
          signalPassFailure();
          return;
        }
        OutputRefConversion conv;
        conv.oldOutputIndex = outIdx;
        conv.oldPortIndex = portIndex;
        conv.refValue = outVal;
        conv.dropInternalDrives = true;
        info.outputsToInputs.push_back(conv);
        continue;
      }
      if (internalDrive) {
        info.keptOutputs.push_back(outIdx);
        continue;
      }
      if (!externalDrive) {
        info.keptOutputs.push_back(outIdx);
        continue;
      }
      OutputRefConversion conv;
      conv.oldOutputIndex = outIdx;
      conv.oldPortIndex = portIndex;
      conv.refValue = outVal;
      info.outputsToInputs.push_back(conv);
    }
  }

  // Update module port types and convert ref arguments into local signals.
  for (auto &entry : moduleInfo) {
    auto &info = entry.second;
    if (!info.hasRefPorts)
      continue;
    auto module = info.module;
    auto oldType = info.oldType;
    auto ports = oldType.getPorts();
    unsigned oldNumInputs = oldType.getNumInputs();

    SmallVector<hw::ModulePort> newInputs;
    SmallVector<hw::ModulePort> newOutputs;
    newInputs.reserve(info.keptInputs.size() + info.outputsToInputs.size());
    newOutputs.reserve(info.keptOutputs.size() + info.inputsToOutputs.size());

    for (unsigned inputIdx : info.keptInputs) {
      auto port = ports[inputIdx];
      if (isRefType(port.type))
        port.type = getRefNestedType(port.type);
      port.dir = hw::ModulePort::Direction::Input;
      newInputs.push_back(port);
    }

    for (auto &conv : info.outputsToInputs) {
      auto port = ports[conv.oldPortIndex];
      port.type = getRefNestedType(port.type);
      port.dir = hw::ModulePort::Direction::Input;
      newInputs.push_back(port);
    }

    for (unsigned outIdx : info.keptOutputs) {
      auto port = ports[oldNumInputs + outIdx];
      if (isRefType(port.type))
        port.type = getRefNestedType(port.type);
      port.dir = hw::ModulePort::Direction::Output;
      newOutputs.push_back(port);
    }

    for (auto &conv : info.inputsToOutputs) {
      auto port = ports[conv.oldPortIndex];
      port.type = getRefNestedType(port.type);
      port.dir = hw::ModulePort::Direction::Output;
      newOutputs.push_back(port);
    }

    SmallVector<hw::ModulePort> newPorts;
    newPorts.reserve(newInputs.size() + newOutputs.size());
    newPorts.append(newInputs.begin(), newInputs.end());
    newPorts.append(newOutputs.begin(), newOutputs.end());
    module.setModuleType(hw::ModuleType::get(context, newPorts));

    Block *body = module.getBodyBlock();
    for (unsigned i = 0, e = info.keptInputs.size(); i < e; ++i) {
      unsigned oldIndex = info.keptInputs[i];
      Type newType = newInputs[i].type;
      if (body->getArgument(oldIndex).getType() != newType)
        body->getArgument(oldIndex).setType(newType);
    }

    OpBuilder signalBuilder(body, body->begin());
    for (auto &conv : info.inputsToOutputs) {
      Type nestedType = getRefNestedType(ports[conv.oldPortIndex].type);
      Value init = createZeroValue(signalBuilder, module.getLoc(), nestedType);
      if (!init) {
        module.emitError("unsupported ref type for driven input port");
        signalPassFailure();
        return;
      }
      auto sigRefTy = llhd::RefType::get(nestedType);
      auto sigOp = llhd::SignalOp::create(signalBuilder, module.getLoc(),
                                          sigRefTy, StringAttr{}, init);
      conv.internalSignal = sigOp.getResult();
      body->getArgument(conv.oldInputIndex)
          .replaceAllUsesWith(conv.internalSignal);
    }

    SmallVector<unsigned> removeIndices;
    removeIndices.reserve(info.inputsToOutputs.size());
    for (auto &conv : info.inputsToOutputs)
      removeIndices.push_back(conv.oldInputIndex);
    llvm::sort(removeIndices,
               [](unsigned a, unsigned b) { return a > b; });
    for (unsigned idx : removeIndices)
      body->eraseArgument(idx);

    unsigned baseIndex = info.keptInputs.size();
    for (unsigned i = 0, e = info.outputsToInputs.size(); i < e; ++i) {
      auto &port = newInputs[baseIndex + i];
      body->addArgument(port.type, module.getLoc());
      Value newArg = body->getArgument(baseIndex + i);
      if (failed(replaceRefUsesWithInput(info.outputsToInputs[i].refValue,
                                         newArg,
                                         info.outputsToInputs[i]
                                             .dropInternalDrives))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Rewrite instances to match converted module ports.
  for (auto module : modules) {
    DenseMap<Value, Value> refReplacements;
    SmallVector<hw::InstanceOp> moduleInstances;
    SmallVector<hw::InstanceOp> instancesToErase;
    module.walk([&](hw::InstanceOp inst) { moduleInstances.push_back(inst); });

    Block *body = module.getBodyBlock();
    OpBuilder signalBuilder(body, body->begin());
    Value zeroTime;
    auto getZeroTime = [&]() -> Value {
      if (!zeroTime)
        zeroTime = llhd::ConstantTimeOp::create(signalBuilder, module.getLoc(),
                                                0, "ns", 0, 1);
      return zeroTime;
    };

    auto getRefReplacement = [&](Value ref) -> Value {
      auto it = refReplacements.find(ref);
      return it == refReplacements.end() ? ref : it->second;
    };

    auto ensureRefSignal = [&](Value ref) -> Value {
      auto it = refReplacements.find(ref);
      if (it != refReplacements.end())
        return it->second;
      if (!isRefType(ref.getType()))
        return Value();
      Type nestedType = getRefNestedType(ref.getType());
      Value init = createZeroValue(signalBuilder, module.getLoc(), nestedType);
      if (!init)
        return Value();
      auto sigRefTy = llhd::RefType::get(nestedType);
      auto sigOp = llhd::SignalOp::create(signalBuilder, module.getLoc(),
                                          sigRefTy, StringAttr{}, init);
      refReplacements[ref] = sigOp.getResult();
      return sigOp.getResult();
    };

    for (auto inst : moduleInstances) {
      auto moduleIt = moduleInfo.find(inst.getReferencedModuleNameAttr());
      if (moduleIt == moduleInfo.end())
        continue;
      auto &info = moduleIt->second;
      if (!info.hasRefPorts)
        continue;

      auto callee =
          dyn_cast_or_null<hw::HWModuleOp>(symTable.lookup(
              inst.getReferencedModuleName()));
      if (!callee) {
        inst.emitError("missing module for llhd ref port lowering");
        signalPassFailure();
        return;
      }

      SmallVector<Value> newInputs;
      newInputs.reserve(info.keptInputs.size() + info.outputsToInputs.size());
      auto ports = info.oldType.getPorts();

      OpBuilder builder(inst);
      for (unsigned idx : info.keptInputs) {
        Value operand = inst.getOperand(idx);
        auto portType = ports[idx].type;
        if (!isRefType(portType)) {
          newInputs.push_back(operand);
          continue;
        }
        Value refOperand = getRefReplacement(operand);
        if (isRefType(refOperand.getType())) {
          auto probe =
              llhd::ProbeOp::create(builder, inst.getLoc(), refOperand);
          newInputs.push_back(probe.getResult());
        } else {
          newInputs.push_back(refOperand);
        }
      }

      for (auto &conv : info.outputsToInputs) {
        Value oldResult = inst.getResult(conv.oldOutputIndex);
        Value signal = ensureRefSignal(oldResult);
        if (!signal) {
          inst.emitError("unsupported ref type for output conversion");
          signalPassFailure();
          return;
        }
        auto probe = llhd::ProbeOp::create(builder, inst.getLoc(), signal);
        newInputs.push_back(probe.getResult());
      }

      auto newInst = hw::InstanceOp::create(
          builder, inst.getLoc(), callee, inst.getInstanceNameAttr(), newInputs,
          inst.getParametersAttr(), inst.getInnerSymAttr());
      if (auto doNotPrint = inst.getDoNotPrintAttr())
        newInst.setDoNotPrintAttr(doNotPrint);

      for (unsigned newIdx = 0, e = info.keptOutputs.size(); newIdx < e;
           ++newIdx) {
        unsigned oldIdx = info.keptOutputs[newIdx];
        inst.getResult(oldIdx).replaceAllUsesWith(newInst.getResult(newIdx));
      }

      builder.setInsertionPointAfter(newInst);
      unsigned outputBase = info.keptOutputs.size();
      for (unsigned i = 0, e = info.inputsToOutputs.size(); i < e; ++i) {
        auto &conv = info.inputsToOutputs[i];
        Value oldOperand = inst.getOperand(conv.oldInputIndex);
        Value refTarget = getRefReplacement(oldOperand);
        if (!isRefType(refTarget.getType())) {
          inst.emitError("expected ref operand for driven input port");
          signalPassFailure();
          return;
        }
        Value driveValue = newInst.getResult(outputBase + i);
        llhd::DriveOp::create(builder, inst.getLoc(), refTarget, driveValue,
                              getZeroTime(), Value{});
      }

      instancesToErase.push_back(inst);
    }

    for (auto &entry : refReplacements)
      entry.first.replaceAllUsesWith(entry.second);

    for (auto inst : instancesToErase) {
      bool hasUses = llvm::any_of(inst.getResults(),
                                  [](Value res) { return !res.use_empty(); });
      if (hasUses) {
        inst.emitError("unhandled uses of converted instance outputs");
        signalPassFailure();
        return;
      }
      inst.erase();
    }
  }

  // Rebuild outputs and clean up probes.
  for (auto &entry : moduleInfo) {
    auto &info = entry.second;
    auto module = info.module;
    auto outputOp =
        dyn_cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
    if (!outputOp)
      continue;
    OpBuilder builder(outputOp);
    SmallVector<Value> newOutputs;
    newOutputs.reserve(info.keptOutputs.size() + info.inputsToOutputs.size());
    for (unsigned outIdx : info.keptOutputs) {
      Value operand = outputOp.getOperand(outIdx);
      if (isRefType(operand.getType()))
        operand = llhd::ProbeOp::create(builder, operand.getLoc(), operand)
                      .getResult();
      newOutputs.push_back(operand);
    }
    for (auto &conv : info.inputsToOutputs) {
      Value operand = conv.internalSignal;
      if (isRefType(operand.getType()))
        operand = llhd::ProbeOp::create(builder, module.getLoc(), operand)
                      .getResult();
      newOutputs.push_back(operand);
    }
    hw::OutputOp::create(builder, outputOp.getLoc(), newOutputs);
    outputOp.erase();

    replaceInvalidProbes(module);
  }
}
