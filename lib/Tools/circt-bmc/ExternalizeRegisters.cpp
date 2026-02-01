//===- ExternalizeRegisters.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/I1ValueSimplifier.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Twine.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace igraph;

namespace circt {
#define GEN_PASS_DEF_EXTERNALIZEREGISTERS
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Externalize Registers Pass
//===----------------------------------------------------------------------===//

namespace {
static bool isConstantInt(Value value, bool wantAllOnes) {
  if (auto cst = value.getDefiningOp<hw::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValueAttr()))
      return wantAllOnes ? intAttr.getValue().isAllOnes()
                         : intAttr.getValue().isZero();
    if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValueAttr()))
      return wantAllOnes ? boolAttr.getValue() : !boolAttr.getValue();
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return wantAllOnes ? intAttr.getValue().isAllOnes()
                         : intAttr.getValue().isZero();
    if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValue()))
      return wantAllOnes ? boolAttr.getValue() : !boolAttr.getValue();
  }
  return false;
}

static bool isZeroTime(llhd::TimeAttr time) {
  return time.getTime() == 0 && time.getDelta() == 0 &&
         time.getEpsilon() == 0;
}

static bool traceClockRoot(Value value, Value &root);
static StringAttr getClockPortName(HWModuleOp module, Value clock) {
  auto arg = dyn_cast<BlockArgument>(clock);
  if (!arg || arg.getOwner() != module.getBodyBlock())
    return StringAttr::get(module.getContext(), "");
  if (!isa<seq::ClockType>(arg.getType()))
    return StringAttr::get(module.getContext(), "");
  auto inputNames = module.getInputNames();
  if (arg.getArgNumber() >= inputNames.size())
    return StringAttr::get(module.getContext(), "");
  if (auto name = dyn_cast<StringAttr>(inputNames[arg.getArgNumber()]))
    return name;
  return StringAttr::get(module.getContext(), "");
}

static bool traceClockRootFromMemory(Value addr, Value &root) {
  SmallVector<Value, 8> worklist;
  DenseSet<Value> visited;
  worklist.push_back(addr);

  bool foundStore = false;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (auto cast = current.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
        worklist.push_back(cast->getOperand(0));
    }
    if (auto bitcast = current.getDefiningOp<LLVM::BitcastOp>())
      worklist.push_back(bitcast.getArg());
    if (auto addrSpaceCast = current.getDefiningOp<LLVM::AddrSpaceCastOp>())
      worklist.push_back(addrSpaceCast.getArg());

    for (auto *user : current.getUsers()) {
      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        foundStore = true;
        if (!traceClockRoot(store.getValue(), root))
          return false;
        continue;
      }
      if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user))
        worklist.push_back(bitcast.getResult());
      if (auto addrSpaceCast = dyn_cast<LLVM::AddrSpaceCastOp>(user))
        worklist.push_back(addrSpaceCast.getResult());
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
        if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
          worklist.push_back(cast->getResult(0));
      }
    }
  }

  return foundStore;
}

static bool traceClockRoot(Value value, Value &root) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
      return traceClockRoot(cast->getOperand(0), root);
  }
  if (auto bitcast = value.getDefiningOp<LLVM::BitcastOp>())
    return traceClockRoot(bitcast.getArg(), root);
  if (auto toClock = value.getDefiningOp<seq::ToClockOp>())
    return traceClockRoot(toClock.getInput(), root);
  if (auto delay = value.getDefiningOp<llhd::DelayOp>()) {
    if (isZeroTime(delay.getDelayAttr()))
      return traceClockRoot(delay.getInput(), root);
    return false;
  }
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (!root)
      root = arg;
    return arg == root;
  }
  if (auto proc = value.getDefiningOp<llhd::ProcessOp>()) {
    if (!root)
      root = value;
    return value == root;
  }
  if (isConstantInt(value, true) || isConstantInt(value, false))
    return true;
  if (auto extract = value.getDefiningOp<hw::StructExtractOp>())
    return traceClockRoot(extract.getInput(), root);
  if (auto bitcast = value.getDefiningOp<hw::BitcastOp>())
    return traceClockRoot(bitcast.getInput(), root);
  if (auto probe = value.getDefiningOp<llhd::ProbeOp>()) {
    Value signal = probe.getSignal();
    bool sawDrive = false;
    for (auto *user : signal.getUsers()) {
      if (auto drive = dyn_cast<llhd::DriveOp>(user)) {
        sawDrive = true;
        if (traceClockRoot(drive.getValue(), root))
          return true;
      }
    }
    if (sawDrive)
      return false;
    return traceClockRootFromMemory(signal, root);
  }
  if (value.getType().isInteger(1)) {
    BlockArgument i1Root;
    if (traceI1ValueRoot(value, i1Root)) {
      if (!i1Root)
        return true;
      if (!root)
        root = i1Root;
      return root == i1Root;
    }
  }
  if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
    for (auto operand : andOp.getOperands())
      if (!traceClockRoot(operand, root))
        return false;
    return true;
  }
  if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
    for (auto operand : orOp.getOperands())
      if (!traceClockRoot(operand, root))
        return false;
    return true;
  }
  if (auto muxOp = value.getDefiningOp<comb::MuxOp>()) {
    return traceClockRoot(muxOp.getCond(), root) &&
           traceClockRoot(muxOp.getTrueValue(), root) &&
           traceClockRoot(muxOp.getFalseValue(), root);
  }
  if (auto extractOp = value.getDefiningOp<comb::ExtractOp>())
    return traceClockRoot(extractOp.getInput(), root);
  if (auto concatOp = value.getDefiningOp<comb::ConcatOp>()) {
    for (auto operand : concatOp.getOperands())
      if (!traceClockRoot(operand, root))
        return false;
    return true;
  }
  if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
    for (auto operand : xorOp.getOperands())
      if (!traceClockRoot(operand, root))
        return false;
    return true;
  }
  if (auto icmpOp = value.getDefiningOp<comb::ICmpOp>()) {
    return traceClockRoot(icmpOp.getLhs(), root) &&
           traceClockRoot(icmpOp.getRhs(), root);
  }
  return false;
}

struct ExternalizeRegistersPass
    : public circt::impl::ExternalizeRegistersBase<ExternalizeRegistersPass> {
  using ExternalizeRegistersBase::ExternalizeRegistersBase;
  void runOnOperation() override;

private:
  DenseMap<StringAttr, SmallVector<Type>> addedInputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedInputNames;
  DenseMap<StringAttr, SmallVector<Type>> addedOutputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedOutputNames;
  DenseMap<StringAttr, SmallVector<Attribute>> initialValues;
  DenseMap<StringAttr, SmallVector<StringAttr>> regClockNames;

  LogicalResult externalizeReg(HWModuleOp module, Operation *op, Twine regName,
                               Value clock, Attribute initState, Value reset,
                               bool isAsync, Value resetValue, Value next);
};
} // namespace

void ExternalizeRegistersPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  DenseSet<Operation *> handled;

  // Iterate over all instances in the instance graph. This ensures we visit
  // every module, even private top modules (private and never instantiated).
  for (auto *startNode : instanceGraph) {
    if (handled.count(startNode->getModule().getOperation()))
      continue;

    // Visit the instance subhierarchy starting at the current module, in a
    // depth-first manner. This allows us to inline child modules into parents
    // before we attempt to inline parents into their parents.
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!handled.insert(node->getModule().getOperation()).second)
        continue;

      auto module =
          dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
      if (!module)
        continue;

      unsigned numRegs = 0;
      bool foundClk = false;
      for (auto ty : module.getInputTypes()) {
        if (isa<seq::ClockType>(ty)) {
          if (foundClk && !allowMultiClock) {
            module.emitError("modules with multiple clocks not yet supported");
            return signalPassFailure();
          }
          foundClk = true;
        }
      }
      module->walk([&](Operation *op) {
        if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
          mlir::Attribute initState = {};
          if (auto initVal = regOp.getInitialValue()) {
            // Find the constant op that defines the reset value in an initial
            // block (if it exists)
            if (!initVal.getDefiningOp<seq::InitialOp>()) {
              regOp.emitError("registers with initial values not directly "
                              "defined by a seq.initial op not yet supported");
              return signalPassFailure();
            }
            if (auto constantOp = circt::seq::unwrapImmutableValue(initVal)
                                      .getDefiningOp<hw::ConstantOp>()) {
              // Fetch value from constant op - leave removing the dead op to
              // DCE
              initState = constantOp.getValueAttr();
            } else {
              regOp.emitError("registers with initial values not directly "
                              "defined by a hw.constant op in a seq.initial op "
                              "not yet supported");
              return signalPassFailure();
            }
          }
          Twine regName = regOp.getName() && !(regOp.getName().value().empty())
                              ? regOp.getName().value()
                              : "reg_" + Twine(numRegs);

          if (failed(externalizeReg(module, op, regName, regOp.getClk(),
                                    initState, regOp.getReset(), false,
                                    regOp.getResetValue(), regOp.getInput()))) {
            return signalPassFailure();
          }
          regOp->erase();
          ++numRegs;
          return;
        }
        if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
          mlir::Attribute initState = {};
          if (auto preset = regOp.getPreset()) {
            // Get preset value as initState
            initState = mlir::IntegerAttr::get(
                mlir::IntegerType::get(&getContext(), preset->getBitWidth()),
                *preset);
          }
          Twine regName = regOp.getName().empty() ? "reg_" + Twine(numRegs)
                                                  : regOp.getName();

          if (failed(externalizeReg(module, op, regName, regOp.getClk(),
                                    initState, regOp.getReset(),
                                    regOp.getIsAsync(), regOp.getResetValue(),
                                    regOp.getNext()))) {
            return signalPassFailure();
          }
          regOp->erase();
          ++numRegs;
          return;
        }
        if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
          OpBuilder builder(instanceOp);
          auto newInputs =
              addedInputs[instanceOp.getModuleNameAttr().getAttr()];
          auto newInputNames =
              addedInputNames[instanceOp.getModuleNameAttr().getAttr()];
          auto newOutputs =
              addedOutputs[instanceOp.getModuleNameAttr().getAttr()];
          auto newOutputNames =
              addedOutputNames[instanceOp.getModuleNameAttr().getAttr()];
          auto childClockNames =
              regClockNames[instanceOp.getModuleNameAttr().getAttr()];
          addedInputs[module.getSymNameAttr()].append(newInputs);
          addedInputNames[module.getSymNameAttr()].append(newInputNames);
          addedOutputs[module.getSymNameAttr()].append(newOutputs);
          addedOutputNames[module.getSymNameAttr()].append(newOutputNames);
          initialValues[module.getSymNameAttr()].append(
              initialValues[instanceOp.getModuleNameAttr().getAttr()]);
          SmallVector<Attribute> argNames(
              instanceOp.getInputNames().getValue());
          SmallVector<Attribute> resultNames(
              instanceOp.getOutputNames().getValue());

          SmallVector<StringAttr> mappedClockNames;
          mappedClockNames.reserve(childClockNames.size());
          auto inputNamesAttr = instanceOp.getInputNames().getValue();
          auto lookupInputIndex = [&](StringAttr name)
              -> std::optional<size_t> {
            for (auto [idx, attr] : llvm::enumerate(inputNamesAttr)) {
              if (attr == name)
                return idx;
            }
            return std::nullopt;
          };
          for (auto clockName : childClockNames) {
            if (!clockName || clockName.getValue().empty()) {
              mappedClockNames.push_back(
                  StringAttr::get(module.getContext(), ""));
              continue;
            }
            auto inputIdx = lookupInputIndex(clockName);
            if (!inputIdx) {
              mappedClockNames.push_back(
                  StringAttr::get(module.getContext(), ""));
              continue;
            }
            Value operand = instanceOp.getOperand(*inputIdx);
            mappedClockNames.push_back(getClockPortName(module, operand));
          }
          regClockNames[module.getSymNameAttr()].append(mappedClockNames);

          for (auto [input, name] : zip_equal(newInputs, newInputNames)) {
            instanceOp.getInputsMutable().append(
                module.appendInput(name, input).second);
            argNames.push_back(name);
          }
          for (auto outputName : newOutputNames) {
            resultNames.push_back(outputName);
          }
          SmallVector<Type> resTypes(instanceOp->getResultTypes());
          resTypes.append(newOutputs);
          auto newInst = InstanceOp::create(
              builder, instanceOp.getLoc(), resTypes,
              instanceOp.getInstanceNameAttr(), instanceOp.getModuleNameAttr(),
              instanceOp.getInputs(), builder.getArrayAttr(argNames),
              builder.getArrayAttr(resultNames), instanceOp.getParametersAttr(),
              instanceOp.getInnerSymAttr(), instanceOp.getDoNotPrintAttr());
          for (auto [output, name] :
               zip(newInst->getResults().take_back(newOutputs.size()),
                   newOutputNames))
            module.appendOutput(name, output);
          numRegs += newInputs.size();
          instanceOp.replaceAllUsesWith(
              newInst.getResults().take_front(instanceOp->getNumResults()));
          instanceGraph.replaceInstance(instanceOp, newInst);
          instanceOp->erase();
          return;
        }
      });

      module->setAttr(
          "num_regs",
          IntegerAttr::get(IntegerType::get(&getContext(), 32), numRegs));

      module->setAttr("initial_values",
                      ArrayAttr::get(&getContext(),
                                     initialValues[module.getSymNameAttr()]));
      if (!regClockNames[module.getSymNameAttr()].empty()) {
        SmallVector<Attribute> clockAttrs;
        clockAttrs.reserve(regClockNames[module.getSymNameAttr()].size());
        for (auto clockName : regClockNames[module.getSymNameAttr()])
          clockAttrs.push_back(clockName);
        module->setAttr(
            "bmc_reg_clocks",
            ArrayAttr::get(&getContext(), clockAttrs));
      }
    }
  }
}

LogicalResult ExternalizeRegistersPass::externalizeReg(
    HWModuleOp module, Operation *op, Twine regName, Value clock,
    Attribute initState, Value reset, bool isAsync, Value resetValue,
    Value next) {
  // Look through ToClockOp and simple combinational logic to find the clock
  // root. If the clock cannot be traced to a block argument or constant, we
  // currently bail out.
  Value root;
  if (!traceClockRoot(clock, root)) {
    op->emitError("only clocks derived from block arguments, constants, or "
                  "process results are supported");
    return failure();
  }

  OpBuilder builder(op);
  auto result = op->getResult(0);
  auto regType = result.getType();

  // If there's no initial value just add a unit attribute to maintain
  // one-to-one correspondence with module ports
  initialValues[module.getSymNameAttr()].push_back(
      initState ? initState : mlir::UnitAttr::get(&getContext()));

  StringAttr newInputName(builder.getStringAttr(regName + "_state")),
      newOutputName(builder.getStringAttr(regName + "_next"));
  addedInputs[module.getSymNameAttr()].push_back(regType);
  addedInputNames[module.getSymNameAttr()].push_back(newInputName);
  // Use regType for output as well to ensure type consistency between register
  // state input and next-state output. This is critical for BMC's IteOp which
  // requires matching types for then-value (regInput) and else-value (regState).
  addedOutputs[module.getSymNameAttr()].push_back(regType);
  addedOutputNames[module.getSymNameAttr()].push_back(newOutputName);

  // Replace the register with newInput and newOutput
  auto newInput = module.appendInput(newInputName, regType).second;
  result.replaceAllUsesWith(newInput);
  regClockNames[module.getSymNameAttr()].push_back(
      getClockPortName(module, clock));
  if (reset) {
    if (isAsync) {
      // Async reset
      op->emitError("registers with an async reset are not yet supported");
      return failure();
    }
    // Sync reset
    auto mux = comb::MuxOp::create(builder, op->getLoc(), regType, reset,
                                   resetValue, next);
    module.appendOutput(newOutputName, mux);
  } else {
    // No reset
    module.appendOutput(newOutputName, next);
  }

  return success();
}
