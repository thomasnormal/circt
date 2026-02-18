//===- LLHDProcessInterpreterModuleLevelInit.cpp - Module-level init ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "LLHDProcessInterpreterStorePatterns.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

LogicalResult LLHDProcessInterpreter::executeModuleLevelLLVMOps(
    hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: Executing module-level LLVM ops\n");

  // Create a temporary process state for executing module-level ops.
  // Must use a non-zero ID because InvalidProcessId == 0 and
  // findMemoryBlockByAddress's walk loop skips process ID 0.
  ProcessExecutionState tempState;
  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId) || tempProcId == InvalidProcessId)
    tempProcId = nextTempProcId++;
  processStates[tempProcId] = std::move(tempState);

  unsigned opsExecuted = 0;

  // Walk the module body (but not inside processes) and execute LLVM ops.
  for (Operation &op : hwModule.getBody().front()) {
    if (isa<llhd::ProcessOp, seq::InitialOp, llhd::CombinationalOp,
            llhd::SignalOp, hw::InstanceOp, hw::OutputOp>(&op))
      continue;

    if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(&op)) {
      (void)interpretLLVMAlloca(tempProcId, allocaOp);
      ++opsExecuted;
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(&op)) {
      (void)interpretLLVMStore(tempProcId, storeOp);
      ++opsExecuted;

      InterpretedValue destAddr = getValue(tempProcId, storeOp.getAddr());
      if (!destAddr.isX() && destAddr.getUInt64() != 0) {
        uint64_t dest = destAddr.getUInt64();
        auto resolveAddr = [&](Value v) -> uint64_t {
          InterpretedValue val = getValue(tempProcId, v);
          if (!val.isX() && val.getUInt64() != 0)
            return val.getUInt64();
          return 0;
        };
        auto resolveSignal = [&](Value v) -> SignalId { return getSignalId(v); };

        uint64_t srcAddr = 0;
        if (matchFourStateCopyStore(storeOp.getValue(), resolveAddr, srcAddr) &&
            srcAddr != 0) {
          auto pair = std::make_pair(srcAddr, dest);
          if (std::find(childModuleCopyPairs.begin(), childModuleCopyPairs.end(),
                        pair) == childModuleCopyPairs.end())
            childModuleCopyPairs.push_back(pair);
        }

        SignalId srcSignalId = 0;
        if (matchFourStateProbeCopyStore(storeOp.getValue(), resolveSignal,
                                         srcSignalId) &&
            srcSignalId != 0) {
          auto pair = std::make_pair(srcSignalId, dest);
          if (std::find(interfaceSignalCopyPairs.begin(),
                        interfaceSignalCopyPairs.end(),
                        pair) == interfaceSignalCopyPairs.end())
            interfaceSignalCopyPairs.push_back(pair);
        }

        InterfaceTriStateStorePattern triPattern;
        if (matchInterfaceTriStateStore(storeOp.getValue(), resolveAddr,
                                        triPattern) &&
            triPattern.srcAddr != 0 && triPattern.condAddr != 0) {
          InterpretedValue elseVal = getValue(tempProcId, triPattern.elseValue);
          bool duplicate = false;
          for (const auto &cand : interfaceTriStateCandidates) {
            bool elseMatch = false;
            if (cand.elseValue.isX() && elseVal.isX()) {
              elseMatch = true;
            } else if (!cand.elseValue.isX() && !elseVal.isX() &&
                       cand.elseValue.getAPInt() == elseVal.getAPInt()) {
              elseMatch = true;
            }
            if (cand.condAddr == triPattern.condAddr &&
                cand.srcAddr == triPattern.srcAddr && cand.destAddr == dest &&
                cand.condBitIndex == triPattern.condBitIndex && elseMatch) {
              duplicate = true;
              break;
            }
          }
          if (!duplicate) {
            InterfaceTriStateCandidate cand;
            cand.condAddr = triPattern.condAddr;
            cand.srcAddr = triPattern.srcAddr;
            cand.destAddr = dest;
            cand.condBitIndex = triPattern.condBitIndex;
            cand.elseValue = elseVal;
            interfaceTriStateCandidates.push_back(cand);
          }
        }
      }
    } else if (auto callOp = dyn_cast<LLVM::CallOp>(&op)) {
      (void)interpretLLVMCall(tempProcId, callOp);
      ++opsExecuted;
    } else if (auto constOp = dyn_cast<LLVM::ConstantOp>(&op)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        setValue(tempProcId, constOp.getResult(),
                 InterpretedValue(intAttr.getValue()));
        ++opsExecuted;
      }
    } else if (auto hwConstOp = dyn_cast<hw::ConstantOp>(&op)) {
      setValue(tempProcId, hwConstOp.getResult(),
               InterpretedValue(hwConstOp.getValue()));
      ++opsExecuted;
    } else if (auto undefOp = dyn_cast<LLVM::UndefOp>(&op)) {
      unsigned width = getTypeWidth(undefOp.getType());
      setValue(tempProcId, undefOp.getResult(),
               InterpretedValue(APInt::getZero(width)));
      ++opsExecuted;
    } else if (auto zeroOp = dyn_cast<LLVM::ZeroOp>(&op)) {
      setValue(tempProcId, zeroOp.getResult(), InterpretedValue(0, 64));
      ++opsExecuted;
    } else if (auto addrOfOp = dyn_cast<LLVM::AddressOfOp>(&op)) {
      (void)interpretLLVMAddressOf(tempProcId, addrOfOp);
      ++opsExecuted;
    } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
      (void)interpretLLVMLoad(tempProcId, loadOp);
      ++opsExecuted;
    } else if (auto probeOp = dyn_cast<llhd::ProbeOp>(&op)) {
      Value sig = probeOp.getSignal();
      bool handled = false;
      if (auto sigOp = sig.getDefiningOp<llhd::SignalOp>()) {
        auto initIt = processStates[tempProcId].valueMap.find(sigOp.getInit());
        if (initIt != processStates[tempProcId].valueMap.end() &&
            !initIt->second.isX()) {
          setValue(tempProcId, probeOp.getResult(), initIt->second);
          handled = true;
        }
      }
      if (!handled)
        (void)interpretOperation(tempProcId, &op);
      ++opsExecuted;
    } else if (isa<LLVM::InsertValueOp, LLVM::ExtractValueOp>(&op)) {
      (void)interpretOperation(tempProcId, &op);
      ++opsExecuted;
    } else if (isa<LLVM::GEPOp>(&op)) {
      (void)interpretOperation(tempProcId, &op);
      ++opsExecuted;
    } else if (succeeded(interpretOperation(tempProcId, &op))) {
      ++opsExecuted;
    }
  }

  for (Operation &op : hwModule.getBody().front()) {
    auto sigOp = dyn_cast<llhd::SignalOp>(&op);
    if (!sigOp)
      continue;
    auto it = processStates[tempProcId].valueMap.find(sigOp.getInit());
    if (it == processStates[tempProcId].valueMap.end())
      continue;
    InterpretedValue initVal = it->second;
    if (initVal.isX() || initVal.getUInt64() == 0)
      continue;
    SignalId sigId = valueToSignal.lookup(sigOp.getResult());
    if (sigId == 0)
      continue;
    scheduler.updateSignal(sigId, initVal.toSignalValue());
    LLVM_DEBUG(llvm::dbgs()
               << "  Updated signal " << sigId << " initial value from "
               << "module-level LLVM op result\n");
  }

  for (Operation &op : hwModule.getBody().front()) {
    auto sigOp = dyn_cast<llhd::SignalOp>(&op);
    if (!sigOp || !isa<LLVM::LLVMPointerType>(sigOp.getInit().getType()))
      continue;
    SignalId sigId = valueToSignal.lookup(sigOp.getResult());
    if (sigId == 0)
      continue;
    const SignalValue &sigVal = scheduler.getSignalValue(sigId);
    if (sigVal.isUnknown() || sigVal.getWidth() < 64 || sigVal.getValue() == 0)
      continue;

    auto &tmpMap = processStates[tempProcId].valueMap;
    auto it = tmpMap.find(sigOp.getInit());
    if (it == tmpMap.end() || it->second.isX() || it->second.getUInt64() == 0)
      tmpMap[sigOp.getInit()] = InterpretedValue(sigVal.getValue(), 64);
  }

  for (auto &[val, intVal] : processStates[tempProcId].valueMap)
    moduleInitValueMap[val] = intVal;

  for (auto &[value, block] : processStates[tempProcId].memoryBlocks) {
    auto addrIt = processStates[tempProcId].valueMap.find(value);
    if (addrIt != processStates[tempProcId].valueMap.end() &&
        !addrIt->second.isX())
      moduleLevelAllocaBaseAddr[value] = addrIt->second.getUInt64();
    moduleLevelAllocas[value] = std::move(block);
  }

  processStates.erase(tempProcId);

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executed " << opsExecuted
                          << " module-level LLVM ops\n");

  return success();
}
