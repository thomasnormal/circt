//===- PruneBMCRegisters.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::verif;

namespace circt {
#define GEN_PASS_DEF_PRUNEBMCREGISTERS
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct PruneBMCRegistersPass
    : public circt::impl::PruneBMCRegistersBase<PruneBMCRegistersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTables;
    SymbolUserMap symbolUsers(symbolTables, module);

    for (auto hwModule : module.getOps<HWModuleOp>()) {
      auto numRegsAttr = hwModule->getAttrOfType<IntegerAttr>("num_regs");
      auto initialValuesAttr =
          hwModule->getAttrOfType<ArrayAttr>("initial_values");
      if (!numRegsAttr || !initialValuesAttr)
        continue;

      auto users = symbolUsers.getUsers(hwModule);
      if (llvm::any_of(users, [](Operation *user) {
            return isa<hw::InstanceOp>(user);
          })) {
        continue;
      }

      auto *bodyBlock = hwModule.getBodyBlock();
      if (!bodyBlock)
        continue;

      bool hasProperties = false;
      hwModule.walk([&](Operation *op) {
        if (isa<AssertOp, AssumeOp, CoverOp, ClockedAssertOp, ClockedAssumeOp,
                ClockedCoverOp>(op))
          hasProperties = true;
      });
      if (!hasProperties)
        continue;

      unsigned numRegs = numRegsAttr.getValue().getZExtValue();
      unsigned numInputs = hwModule.getNumInputPorts();
      unsigned numOutputs = hwModule.getNumOutputPorts();
      if (numRegs == 0)
        continue;
      if (numRegs > numInputs || numRegs > numOutputs) {
        hwModule.emitError("num_regs exceeds input/output port count");
        signalPassFailure();
        return;
      }
      if (initialValuesAttr.size() != numRegs) {
        hwModule.emitError("initial_values size does not match num_regs");
        signalPassFailure();
        return;
      }

      auto outputOp = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator());
      if (!outputOp) {
        hwModule.emitError("expected hw.output terminator");
        signalPassFailure();
        return;
      }

      unsigned baseInputIndex = numInputs - numRegs;
      unsigned baseOutputIndex = numOutputs - numRegs;

      SmallVector<BlockArgument> regStateArgs;
      regStateArgs.reserve(numRegs);
      for (unsigned idx = 0; idx < numRegs; ++idx)
        regStateArgs.push_back(bodyBlock->getArgument(baseInputIndex + idx));

      auto outputs = outputOp.getOutputs();
      SmallVector<Value> regNextValues;
      regNextValues.reserve(numRegs);
      for (unsigned idx = 0; idx < numRegs; ++idx)
        regNextValues.push_back(outputs[baseOutputIndex + idx]);

      DenseMap<Value, unsigned> regArgIndex;
      for (unsigned idx = 0; idx < numRegs; ++idx)
        regArgIndex[regStateArgs[idx]] = idx;

      BackwardSliceOptions sliceOptions;
      sliceOptions.inclusive = true;
      sliceOptions.omitBlockArguments = true;
      sliceOptions.omitUsesFromAbove = true;

      auto collectRegDeps = [&](Value root) {
        BitVector deps(numRegs);
        if (!root)
          return deps;
        auto recordArg = [&](Value value) {
          auto arg = dyn_cast<BlockArgument>(value);
          if (!arg || arg.getOwner() != bodyBlock)
            return;
          auto it = regArgIndex.find(arg);
          if (it != regArgIndex.end())
            deps.set(it->second);
        };

        DenseSet<Value> visited;
        std::function<void(Value)> recordDepsFromValue;
        recordDepsFromValue = [&](Value value) {
          if (!value || !visited.insert(value).second)
            return;
          recordArg(value);
          SetVector<Operation *> slice;
          (void)getBackwardSlice(value, &slice, sliceOptions);
          for (Operation *op : slice) {
            for (Value operand : op->getOperands())
              recordArg(operand);
            if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
              Value signal = probe.getSignal();
              for (Operation *user : signal.getUsers()) {
                auto drv = dyn_cast<llhd::DriveOp>(user);
                if (!drv)
                  continue;
                recordDepsFromValue(drv.getValue());
                if (Value enable = drv.getEnable())
                  recordDepsFromValue(enable);
              }
            }
            if (auto comb = dyn_cast<llhd::CombinationalOp>(op)) {
              comb.getBody().walk([&](llhd::ProbeOp probe) {
                recordDepsFromValue(probe.getResult());
              });
            }
          }
        };

        recordDepsFromValue(root);
        return deps;
      };

      auto collectInputDeps = [&](Value root) {
        BitVector deps(baseInputIndex);
        if (!root)
          return deps;
        auto recordArg = [&](Value value) {
          auto arg = dyn_cast<BlockArgument>(value);
          if (!arg || arg.getOwner() != bodyBlock)
            return;
          unsigned idx = arg.getArgNumber();
          if (idx < baseInputIndex)
            deps.set(idx);
        };

        DenseSet<Value> visited;
        std::function<void(Value)> recordDepsFromValue;
        recordDepsFromValue = [&](Value value) {
          if (!value || !visited.insert(value).second)
            return;
          recordArg(value);
          SetVector<Operation *> slice;
          (void)getBackwardSlice(value, &slice, sliceOptions);
          for (Operation *op : slice) {
            for (Value operand : op->getOperands())
              recordArg(operand);
            if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
              Value signal = probe.getSignal();
              for (Operation *user : signal.getUsers()) {
                auto drv = dyn_cast<llhd::DriveOp>(user);
                if (!drv)
                  continue;
                recordDepsFromValue(drv.getValue());
                if (Value enable = drv.getEnable())
                  recordDepsFromValue(enable);
              }
            }
            if (auto comb = dyn_cast<llhd::CombinationalOp>(op)) {
              comb.getBody().walk([&](llhd::ProbeOp probe) {
                recordDepsFromValue(probe.getResult());
              });
            }
          }
        };

        recordDepsFromValue(root);
        return deps;
      };

      SmallVector<BitVector> regDeps;
      regDeps.reserve(numRegs);
      for (Value nextValue : regNextValues)
        regDeps.push_back(collectRegDeps(nextValue));

      BitVector liveRegs(numRegs);
      SmallVector<Value> propertyRoots;
      hwModule.walk([&](Operation *op) {
        auto addRoot = [&](Value value) {
          if (value)
            propertyRoots.push_back(value);
        };
        if (auto assertOp = dyn_cast<AssertOp>(op)) {
          addRoot(assertOp.getProperty());
          addRoot(assertOp.getEnable());
        } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
          addRoot(assumeOp.getProperty());
          addRoot(assumeOp.getEnable());
        } else if (auto coverOp = dyn_cast<CoverOp>(op)) {
          addRoot(coverOp.getProperty());
          addRoot(coverOp.getEnable());
        } else if (auto clockedAssertOp = dyn_cast<ClockedAssertOp>(op)) {
          addRoot(clockedAssertOp.getProperty());
          addRoot(clockedAssertOp.getEnable());
          addRoot(clockedAssertOp.getClock());
        } else if (auto clockedAssumeOp = dyn_cast<ClockedAssumeOp>(op)) {
          addRoot(clockedAssumeOp.getProperty());
          addRoot(clockedAssumeOp.getEnable());
          addRoot(clockedAssumeOp.getClock());
        } else if (auto clockedCoverOp = dyn_cast<ClockedCoverOp>(op)) {
          addRoot(clockedCoverOp.getProperty());
          addRoot(clockedCoverOp.getEnable());
          addRoot(clockedCoverOp.getClock());
        }
      });

      BitVector liveInputs(baseInputIndex);
      if (auto regClocksAttr =
              hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks")) {
        auto inputNames = hwModule.getInputNames();
        DenseMap<StringAttr, unsigned> inputNameIndex;
        inputNameIndex.reserve(inputNames.size());
        for (auto [idx, nameAttr] : llvm::enumerate(inputNames)) {
          inputNameIndex[cast<StringAttr>(nameAttr)] = idx;
        }
        for (auto clockAttr : regClocksAttr) {
          auto nameAttr = dyn_cast<StringAttr>(clockAttr);
          if (!nameAttr)
            continue;
          auto it = inputNameIndex.find(nameAttr);
          if (it != inputNameIndex.end() && it->second < baseInputIndex)
            liveInputs.set(it->second);
        }
      }
      if (auto regClockSourcesAttr =
              hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clock_sources")) {
        for (auto clockSourceAttr : regClockSourcesAttr) {
          auto sourceDict = dyn_cast<DictionaryAttr>(clockSourceAttr);
          if (!sourceDict)
            continue;
          auto argIndexAttr = sourceDict.getAs<IntegerAttr>("arg_index");
          if (!argIndexAttr)
            continue;
          uint64_t argIndex = argIndexAttr.getValue().getZExtValue();
          if (argIndex < baseInputIndex)
            liveInputs.set(static_cast<unsigned>(argIndex));
        }
      }
      for (Value root : propertyRoots)
        liveRegs |= collectRegDeps(root);
      for (Value root : propertyRoots)
        liveInputs |= collectInputDeps(root);

      SmallVector<unsigned> worklist;
      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (liveRegs.test(idx))
          worklist.push_back(idx);
      }
      while (!worklist.empty()) {
        unsigned idx = worklist.pop_back_val();
        for (int dep = regDeps[idx].find_first(); dep >= 0;
             dep = regDeps[idx].find_next(dep)) {
          if (!liveRegs.test(dep)) {
            liveRegs.set(dep);
            worklist.push_back(dep);
          }
        }
      }

      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (liveRegs.test(idx))
          liveInputs |= collectInputDeps(regNextValues[idx]);
      }

      SetVector<Operation *> liveOps;
      auto addLiveSlice = [&](Value root) {
        if (!root)
          return;
        (void)getBackwardSlice(root, &liveOps, sliceOptions);
      };
      for (Value root : propertyRoots)
        addLiveSlice(root);
      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (liveRegs.test(idx))
          addLiveSlice(regNextValues[idx]);
      }
      // Preserve explicit clock-conversion scaffolding. LowerToBMC discovers
      // non-seq.clock clock inputs by scanning seq.to_clock operations.
      // Pruning these ops can leave verif.bmc with clock args but empty
      // init/loop clock yields.
      hwModule.walk([&](seq::ToClockOp toClockOp) {
        liveOps.insert(toClockOp);
        addLiveSlice(toClockOp.getInput());
        liveInputs |= collectInputDeps(toClockOp.getInput());
      });
      hwModule.walk([&](Operation *op) {
        if (isa<AssertOp, AssumeOp, CoverOp, ClockedAssertOp, ClockedAssumeOp,
                ClockedCoverOp>(op))
          liveOps.insert(op);
      });
      SmallVector<Value> liveSignals;
      liveSignals.reserve(liveOps.size());
      SmallVector<llhd::DriveOp> liveDrives;
      for (Operation *op : liveOps) {
        if (auto probe = dyn_cast<llhd::ProbeOp>(op))
          liveSignals.push_back(probe.getSignal());
        if (auto comb = dyn_cast<llhd::CombinationalOp>(op)) {
          comb.getBody().walk([&](llhd::ProbeOp probe) {
            liveSignals.push_back(probe.getSignal());
          });
        }
      }
      for (Value signal : liveSignals) {
        for (Operation *user : signal.getUsers()) {
          if (auto drv = dyn_cast<llhd::DriveOp>(user)) {
            liveOps.insert(drv);
            liveDrives.push_back(drv);
          }
        }
      }
      for (auto drv : liveDrives) {
        liveInputs |= collectInputDeps(drv.getValue());
        if (Value enable = drv.getEnable())
          liveInputs |= collectInputDeps(enable);
      }

      SmallVector<Operation *> toErase;
      for (Operation &op : bodyBlock->getOperations()) {
        if (&op == outputOp)
          continue;
        if (liveOps.contains(&op))
          continue;
        toErase.push_back(&op);
      }
      DenseSet<Operation *> eraseSet;
      eraseSet.reserve(toErase.size());
      for (Operation *op : toErase)
        eraseSet.insert(op);
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<Operation *> keepOps;
        for (Operation *op : eraseSet) {
          bool usedByLive = false;
          for (Value result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
              if (!eraseSet.contains(user)) {
                usedByLive = true;
                break;
              }
            }
            if (usedByLive)
              break;
          }
          if (usedByLive)
            keepOps.push_back(op);
        }
        for (Operation *op : keepOps) {
          eraseSet.erase(op);
          changed = true;
        }
      }
      SmallVector<Operation *> pendingErase;
      pendingErase.reserve(eraseSet.size());
      for (Operation &op : bodyBlock->getOperations()) {
        if (eraseSet.contains(&op))
          pendingErase.push_back(&op);
      }
      bool progress = true;
      while (progress && !pendingErase.empty()) {
        progress = false;
        for (auto it = pendingErase.begin(); it != pendingErase.end();) {
          Operation *op = *it;
          if (!op->use_empty()) {
            ++it;
            continue;
          }
          op->erase();
          it = pendingErase.erase(it);
          progress = true;
        }
      }

      SmallVector<unsigned> eraseInputs;
      SmallVector<unsigned> eraseOutputs;
      eraseInputs.reserve(numRegs);
      eraseOutputs.reserve(numRegs);
      for (unsigned idx = 0; idx < baseInputIndex; ++idx) {
        if (!liveInputs.test(idx))
          eraseInputs.push_back(idx);
      }
      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (!liveRegs.test(idx))
          eraseInputs.push_back(baseInputIndex + idx);
      }
      BitVector liveOutputSet(numOutputs);
      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (liveRegs.test(idx))
          liveOutputSet.set(baseOutputIndex + idx);
      }
      for (unsigned idx = 0; idx < numOutputs; ++idx) {
        if (!liveOutputSet.test(idx))
          eraseOutputs.push_back(idx);
      }
      llvm::sort(eraseInputs);
      llvm::sort(eraseOutputs);

      BitVector eraseInputSet(numInputs);
      for (unsigned idx : eraseInputs)
        eraseInputSet.set(idx);
      BitVector eraseOutputSet(numOutputs);
      for (unsigned idx : eraseOutputs)
        eraseOutputSet.set(idx);
      SmallVector<int64_t> remappedInputIndex(numInputs, -1);
      unsigned nextInputIndex = 0;
      for (unsigned idx = 0; idx < numInputs; ++idx) {
        if (eraseInputSet.test(idx))
          continue;
        remappedInputIndex[idx] = static_cast<int64_t>(nextInputIndex++);
      }

      hwModule.erasePorts(eraseInputs, eraseOutputs);
      bodyBlock->eraseArguments(eraseInputSet);

      SmallVector<Value> newOutputs;
      newOutputs.reserve(numOutputs - eraseOutputs.size());
      for (auto [idx, output] : llvm::enumerate(outputOp.getOutputs())) {
        if (!eraseOutputSet.test(idx))
          newOutputs.push_back(output);
      }
      outputOp.getOutputsMutable().assign(newOutputs);

      unsigned newNumRegs = liveRegs.count();
      auto *ctx = hwModule.getContext();
      hwModule->setAttr(
          "num_regs",
          IntegerAttr::get(IntegerType::get(ctx, 32), newNumRegs));

      SmallVector<Attribute> newInitialValues;
      newInitialValues.reserve(newNumRegs);
      for (unsigned idx = 0; idx < numRegs; ++idx) {
        if (liveRegs.test(idx))
          newInitialValues.push_back(initialValuesAttr[idx]);
      }
      hwModule->setAttr("initial_values",
                        ArrayAttr::get(ctx, newInitialValues));

      if (auto regClocksAttr =
              hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks")) {
        if (regClocksAttr.size() != numRegs) {
          hwModule.emitError("bmc_reg_clocks size does not match num_regs");
          signalPassFailure();
          return;
        }
        SmallVector<Attribute> newRegClocks;
        newRegClocks.reserve(newNumRegs);
        for (unsigned idx = 0; idx < numRegs; ++idx) {
          if (liveRegs.test(idx))
            newRegClocks.push_back(regClocksAttr[idx]);
        }
        hwModule->setAttr("bmc_reg_clocks",
                          ArrayAttr::get(ctx, newRegClocks));
      }
      if (auto regClockSourcesAttr =
              hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clock_sources")) {
        if (regClockSourcesAttr.size() != numRegs) {
          hwModule.emitError(
              "bmc_reg_clock_sources size does not match num_regs");
          signalPassFailure();
          return;
        }
        SmallVector<Attribute> newRegClockSources;
        newRegClockSources.reserve(newNumRegs);
        for (unsigned idx = 0; idx < numRegs; ++idx) {
          if (!liveRegs.test(idx))
            continue;
          auto sourceDict =
              dyn_cast<DictionaryAttr>(regClockSourcesAttr[idx]);
          if (!sourceDict) {
            newRegClockSources.push_back(regClockSourcesAttr[idx]);
            continue;
          }
          NamedAttrList attrs(sourceDict);
          if (auto argIndexAttr = sourceDict.getAs<IntegerAttr>("arg_index")) {
            uint64_t oldArgIndex = argIndexAttr.getValue().getZExtValue();
            if (oldArgIndex >= remappedInputIndex.size() ||
                remappedInputIndex[oldArgIndex] < 0) {
              hwModule.emitError("bmc_reg_clock_sources arg_index references "
                                 "a pruned input port");
              signalPassFailure();
              return;
            }
            auto remapped = static_cast<uint64_t>(remappedInputIndex[oldArgIndex]);
            attrs.set("arg_index",
                      IntegerAttr::get(argIndexAttr.getType(), remapped));
          }
          newRegClockSources.push_back(DictionaryAttr::get(ctx, attrs));
        }
        hwModule->setAttr("bmc_reg_clock_sources",
                          ArrayAttr::get(ctx, newRegClockSources));
      }
    }
  }
};
} // namespace
