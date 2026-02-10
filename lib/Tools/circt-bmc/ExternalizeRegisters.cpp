//===- ExternalizeRegisters.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
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
#include "mlir/IR/Matchers.h"
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

static FailureOr<Attribute>
foldValueToConstantAttr(Value value, DenseMap<Value, Attribute> &cache,
                        DenseSet<Value> &active) {
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  if (!active.insert(value).second)
    return failure();

  auto clearActive = [&]() { active.erase(value); };

  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    clearActive();
    cache[value] = attr;
    return attr;
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    clearActive();
    return failure();
  }

  Operation *def = result.getDefiningOp();
  if (!def) {
    clearActive();
    return failure();
  }

  SmallVector<Attribute> constantOperands;
  constantOperands.reserve(def->getNumOperands());
  for (Value operand : def->getOperands()) {
    auto folded = foldValueToConstantAttr(operand, cache, active);
    if (failed(folded)) {
      clearActive();
      return failure();
    }
    constantOperands.push_back(*folded);
  }

  SmallVector<OpFoldResult> foldResults;
  if (failed(def->fold(constantOperands, foldResults)) ||
      result.getResultNumber() >= foldResults.size()) {
    clearActive();
    return failure();
  }

  auto foldedAttr = dyn_cast<Attribute>(foldResults[result.getResultNumber()]);
  if (!foldedAttr) {
    clearActive();
    return failure();
  }

  clearActive();
  cache[value] = foldedAttr;
  return foldedAttr;
}

static bool traceClockRoot(Value value, Value &root);
static std::optional<std::pair<BlockArgument, bool>>
getClockSourceInfo(Value clock, HWModuleOp module) {
  if (!clock)
    return std::nullopt;
  bool invert = false;
  Value current = clock;
  while (current) {
    if (auto arg = dyn_cast<BlockArgument>(current)) {
      if (arg.getOwner() == module.getBodyBlock())
        return std::make_pair(arg, invert);
      return std::nullopt;
    }
    if (auto inv = current.getDefiningOp<seq::ClockInverterOp>()) {
      invert = !invert;
      current = inv.getInput();
      continue;
    }
    if (auto gate = current.getDefiningOp<seq::ClockGateOp>()) {
      bool anyTrue = false;
      bool anyNonConst = false;
      auto consider = [&](Value operand) {
        if (isConstantInt(operand, true)) {
          anyTrue = true;
          return;
        }
        if (!isConstantInt(operand, false))
          anyNonConst = true;
      };
      consider(gate.getEnable());
      if (auto testEnable = gate.getTestEnable())
        consider(testEnable);
      if (anyTrue) {
        current = gate.getInput();
        continue;
      }
      if (!anyNonConst)
        return std::nullopt;
      return std::nullopt;
    }
    if (auto mux = current.getDefiningOp<seq::ClockMuxOp>()) {
      if (isConstantInt(mux.getCond(), true)) {
        current = mux.getTrueClock();
        continue;
      }
      if (isConstantInt(mux.getCond(), false)) {
        current = mux.getFalseClock();
        continue;
      }
      if (mux.getTrueClock() == mux.getFalseClock()) {
        current = mux.getTrueClock();
        continue;
      }
      return std::nullopt;
    }
    if (auto div = current.getDefiningOp<seq::ClockDividerOp>()) {
      if (div.getPow2() == 0) {
        current = div.getInput();
        continue;
      }
      return std::nullopt;
    }
    if (auto toClock = current.getDefiningOp<seq::ToClockOp>()) {
      auto simplified = simplifyI1Value(toClock.getInput());
      Value base = simplified.value ? simplified.value : toClock.getInput();
      BlockArgument root;
      if (traceI1ValueRoot(base, root) && root &&
          root.getOwner() == module.getBodyBlock())
        return std::make_pair(root, invert ^ simplified.invert);
      return std::nullopt;
    }
    break;
  }
  return std::nullopt;
}

static Attribute makeClockSourceAttr(OpBuilder &builder, BlockArgument root,
                                      bool invert) {
  if (!root)
    return UnitAttr::get(builder.getContext());
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "arg_index", builder.getI32IntegerAttr(root.getArgNumber())),
       builder.getNamedAttr("invert", builder.getBoolAttr(invert))});
}

static Attribute makeClockKeySourceAttr(OpBuilder &builder, StringRef key,
                                        bool invert) {
  if (key.empty())
    return UnitAttr::get(builder.getContext());
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("clock_key", builder.getStringAttr(key)),
       builder.getNamedAttr("invert", builder.getBoolAttr(invert))});
}

static Value getI1ClockValueForKey(Value clock, bool &invert) {
  invert = false;
  auto canonicalizeLLHDAlias = [](Value value) -> Value {
    if (!value)
      return value;
    Value base = getStructFieldBase(value, "value");
    if (!base || !isFourStateStructType(base.getType()))
      return value;

    auto findValueField = [](Value structVal) -> Value {
      for (Operation *user : structVal.getUsers()) {
        if (auto extract = dyn_cast<hw::StructExtractOp>(user)) {
          if (extract.getFieldName() == "value")
            return extract.getResult();
          continue;
        }
        if (auto explode = dyn_cast<hw::StructExplodeOp>(user)) {
          auto structTy =
              dyn_cast<hw::StructType>(explode.getInput().getType());
          if (!structTy)
            continue;
          auto elements = structTy.getElements();
          for (auto [idx, element] : llvm::enumerate(elements)) {
            if (element.name.getValue() == "value") {
              if (idx < explode->getNumResults())
                return explode->getResult(idx);
              break;
            }
          }
        }
      }
      return Value();
    };

    DenseSet<Value> visited;
    Value currentBase = base;
    Value currentValue = value;
    if (Value directValue = findValueField(currentBase))
      currentValue = directValue;
    auto isTrivialDelay = [](Value time) -> bool {
      auto constTime = time.getDefiningOp<llhd::ConstantTimeOp>();
      if (!constTime)
        return false;
      auto attr = constTime.getValueAttr();
      if (attr.getTime() != 0)
        return false;
      if (attr.getDelta() != 0)
        return false;
      return attr.getEpsilon() <= 1;
    };
    while (currentBase) {
      if (!visited.insert(currentBase).second)
        break;
      auto probe = currentBase.getDefiningOp<llhd::ProbeOp>();
      if (!probe)
        break;
      Value signal = probe.getSignal();
      auto sigOp = signal.getDefiningOp<llhd::SignalOp>();
      if (!sigOp)
        break;

      Value driven;
      bool hasDrive = false;
      bool conflicting = false;
      for (Operation *user : signal.getUsers()) {
        auto drive = dyn_cast<llhd::DriveOp>(user);
        if (!drive || drive.getSignal() != signal)
          continue;
        if (drive.getEnable()) {
          if (auto literal = getConstI1Value(drive.getEnable());
              !literal || !*literal) {
            conflicting = true;
            break;
          }
        }
        if (!isTrivialDelay(drive.getTime())) {
          conflicting = true;
          break;
        }
        Value driveValue = drive.getValue();
        if (!hasDrive) {
          driven = driveValue;
          hasDrive = true;
        } else if (driven != driveValue) {
          conflicting = true;
          break;
        }
      }
      if (!hasDrive || conflicting)
        break;
      if (sigOp.getInit() != driven)
        break;
      if (driven.getType() != currentBase.getType())
        break;

      Value nextValue = findValueField(driven);
      if (!nextValue)
        break;
      currentBase = driven;
      currentValue = nextValue;
    }
    return currentValue;
  };
  auto canonicalize = [&](Value value) -> Value {
    if (!value)
      return value;
    if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
      if (andOp.getNumOperands() == 2) {
        if (auto gated =
                matchFourStateClockGate(andOp.getOperand(0), andOp.getOperand(1)))
          return gated;
        if (auto gated =
                matchFourStateClockGate(andOp.getOperand(1), andOp.getOperand(0)))
          return gated;
      }
    }
    return canonicalizeLLHDAlias(value);
  };
  Value cur = clock;
  while (cur) {
    if (auto invOp = cur.getDefiningOp<seq::ClockInverterOp>()) {
      invert = !invert;
      cur = invOp.getInput();
      continue;
    }
    if (auto toClock = cur.getDefiningOp<seq::ToClockOp>()) {
      auto simplified = simplifyI1Value(toClock.getInput());
      invert ^= simplified.invert;
      Value base = simplified.value ? simplified.value : toClock.getInput();
      return canonicalize(base);
    }
    if (cur.getType().isInteger(1)) {
      auto simplified = simplifyI1Value(cur);
      invert ^= simplified.invert;
      Value base = simplified.value ? simplified.value : cur;
      return canonicalize(base);
    }
    break;
  }
  return Value();
}
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
  if (auto inv = value.getDefiningOp<seq::ClockInverterOp>())
    return traceClockRoot(inv.getInput(), root);
  if (auto gate = value.getDefiningOp<seq::ClockGateOp>()) {
    bool anyTrue = false;
    bool anyNonConst = false;
    auto consider = [&](Value operand) {
      if (isConstantInt(operand, true)) {
        anyTrue = true;
        return;
      }
      if (!isConstantInt(operand, false))
        anyNonConst = true;
    };
    consider(gate.getEnable());
    if (auto testEnable = gate.getTestEnable())
      consider(testEnable);
    if (anyTrue)
      return traceClockRoot(gate.getInput(), root);
    if (!anyNonConst)
      return true;
    return false;
  }
  if (auto mux = value.getDefiningOp<seq::ClockMuxOp>()) {
    if (isConstantInt(mux.getCond(), true))
      return traceClockRoot(mux.getTrueClock(), root);
    if (isConstantInt(mux.getCond(), false))
      return traceClockRoot(mux.getFalseClock(), root);
    if (mux.getTrueClock() == mux.getFalseClock())
      return traceClockRoot(mux.getTrueClock(), root);
    return false;
  }
  if (auto div = value.getDefiningOp<seq::ClockDividerOp>()) {
    if (div.getPow2() == 0)
      return traceClockRoot(div.getInput(), root);
    return false;
  }
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, hw::HWDialect, seq::SeqDialect,
                    llhd::LLHDDialect>();
  }
  void runOnOperation() override;

private:
  DenseMap<StringAttr, SmallVector<Type>> addedInputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedInputNames;
  DenseMap<StringAttr, SmallVector<Type>> addedOutputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedOutputNames;
  DenseMap<StringAttr, SmallVector<Attribute>> initialValues;
  DenseMap<StringAttr, SmallVector<StringAttr>> regClockNames;
  DenseMap<StringAttr, SmallVector<Attribute>> regClockSources;

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
            DenseMap<Value, Attribute> foldedValues;
            DenseSet<Value> activeValues;
            auto initConstant = foldValueToConstantAttr(
                circt::seq::unwrapImmutableValue(initVal), foldedValues,
                activeValues);
            if (failed(initConstant)) {
              regOp.emitError("registers with initial values in a seq.initial "
                              "op must fold to constants");
              return signalPassFailure();
            }
            initState = *initConstant;
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
          auto childClockSources =
              regClockSources[instanceOp.getModuleNameAttr().getAttr()];
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
          SmallVector<Attribute> mappedClockSources;
          mappedClockSources.reserve(childClockSources.size());
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
          for (auto attr : childClockSources) {
            auto dict = dyn_cast<DictionaryAttr>(attr);
            if (!dict) {
              mappedClockSources.push_back(
                  UnitAttr::get(module.getContext()));
              continue;
            }
            auto argAttr = dyn_cast<IntegerAttr>(dict.get("arg_index"));
            auto invertAttr = dyn_cast<BoolAttr>(dict.get("invert"));
            if (!argAttr || !invertAttr) {
              mappedClockSources.push_back(
                  UnitAttr::get(module.getContext()));
              continue;
            }
            unsigned argIndex = argAttr.getValue().getZExtValue();
            if (argIndex >= instanceOp.getNumOperands()) {
              mappedClockSources.push_back(
                  UnitAttr::get(module.getContext()));
              continue;
            }
            Value operand = instanceOp.getOperand(argIndex);
            auto info = getClockSourceInfo(operand, module);
            if (!info) {
              mappedClockSources.push_back(
                  UnitAttr::get(module.getContext()));
              continue;
            }
            bool invert = invertAttr.getValue() ^ info->second;
            mappedClockSources.push_back(
                makeClockSourceAttr(builder, info->first, invert));
          }
          regClockNames[module.getSymNameAttr()].append(mappedClockNames);
          regClockSources[module.getSymNameAttr()].append(mappedClockSources);

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
      if (!regClockSources[module.getSymNameAttr()].empty()) {
        module->setAttr(
            "bmc_reg_clock_sources",
            ArrayAttr::get(&getContext(),
                           regClockSources[module.getSymNameAttr()]));
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
  // fall back to using a stable expression key.
  Value root;
  bool tracedRoot = traceClockRoot(clock, root);
  bool keyInvert = false;
  Value keyValue = getI1ClockValueForKey(clock, keyInvert);
  std::optional<std::string> clockKey;
  if (keyValue) {
    auto getBlockArgName = [&](BlockArgument arg) -> StringRef {
      if (!arg || arg.getOwner() != module.getBodyBlock())
        return {};
      auto inputNames = module.getInputNames();
      if (arg.getArgNumber() >= inputNames.size())
        return {};
      if (auto nameAttr =
              dyn_cast_or_null<StringAttr>(inputNames[arg.getArgNumber()])) {
        if (!nameAttr.getValue().empty())
          return nameAttr.getValue();
      }
      return {};
    };
    clockKey = getI1ValueKeyWithBlockArgNames(keyValue, getBlockArgName);
  }
  if (!tracedRoot && !clockKey) {
    op->emitError("only clocks derived from block arguments, constants, "
                  "process results, or keyable i1 expressions are supported");
    return failure();
  }

  OpBuilder builder(op);
  auto clockName =
      getClockPortName(module, root ? root : clock);
  Attribute clockSource = UnitAttr::get(builder.getContext());
  if (auto info = getClockSourceInfo(clock, module)) {
    clockSource = makeClockSourceAttr(builder, info->first, info->second);
  } else if (clockKey) {
    // If the clock isn't traceable to a single input root (e.g. it's derived
    // from a more complex i1 expression), fall back to a stable expression key.
    //
    // This is primarily used to support multi-clock BMC when the clock is a
    // derived expression not rooted in a single module input. LowerToBMC will
    // map derived clock keys to inserted BMC clock inputs.
    clockSource = makeClockKeySourceAttr(builder, *clockKey, keyInvert);
  }

  if (!allowMultiClock) {
    auto &existingNames = regClockNames[module.getSymNameAttr()];
    auto &existingSources = regClockSources[module.getSymNameAttr()];
    for (auto [existingName, existingSource] :
         llvm::zip_equal(existingNames, existingSources)) {
      bool existingKnownSource = !isa<UnitAttr>(existingSource);
      bool newKnownSource = !isa<UnitAttr>(clockSource);
      if (existingKnownSource && newKnownSource) {
        if (existingSource != clockSource) {
          module.emitError("modules with multiple clocks not yet supported");
          return failure();
        }
        continue;
      }
      if (!existingName.getValue().empty() && !clockName.getValue().empty() &&
          existingName != clockName) {
        module.emitError("modules with multiple clocks not yet supported");
        return failure();
      }
    }
  }

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
  regClockNames[module.getSymNameAttr()].push_back(clockName);
  regClockSources[module.getSymNameAttr()].push_back(clockSource);
  if (reset) {
    auto mux = comb::MuxOp::create(builder, op->getLoc(), regType, reset,
                                   resetValue, next);
    module.appendOutput(newOutputName, mux);
  } else {
    // No reset
    module.appendOutput(newOutputName, next);
  }

  return success();
}
