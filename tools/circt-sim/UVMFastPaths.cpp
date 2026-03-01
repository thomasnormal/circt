//===- UVMFastPaths.cpp - UVM-specific circt-sim fast paths --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains targeted UVM fast paths used by LLHDProcessInterpreter
// to bypass hot report/printer plumbing code during simulation.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "circt/Runtime/MooreRuntime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <cstring>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

namespace {
enum class UvmFastPathCallForm : uint8_t {
  FuncCall,
  CallIndirect,
};

enum class UvmFastPathAction : uint8_t {
  None,
  WaitForSelfAndSiblingsToDrop,
  AdjustNamePassthrough,
  PrinterNoOp,
  ReportInfoSuppress,
  ReportWarningSuppress,
};

static UvmFastPathAction lookupUvmFastPath(UvmFastPathCallForm callForm,
                                           llvm::StringRef calleeName) {
  using llvm::StringSwitch;
  switch (callForm) {
  case UvmFastPathCallForm::CallIndirect:
    return StringSwitch<UvmFastPathAction>(calleeName)
        .Case("uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop",
              UvmFastPathAction::WaitForSelfAndSiblingsToDrop)
        .Case("uvm_pkg::uvm_printer::adjust_name",
              UvmFastPathAction::AdjustNamePassthrough)
        .Case("uvm_pkg::uvm_printer::print_field_int",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_field",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_generic_element",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_generic",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_time",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_string",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_real",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_header",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_footer",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_range",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_object_header",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_object",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_report_object::uvm_report_info",
              UvmFastPathAction::ReportInfoSuppress)
        .Case("uvm_pkg::uvm_report_object::uvm_report_warning",
              UvmFastPathAction::ReportWarningSuppress)
        .Default(UvmFastPathAction::None);
  case UvmFastPathCallForm::FuncCall:
    return StringSwitch<UvmFastPathAction>(calleeName)
        .Case("uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop",
              UvmFastPathAction::WaitForSelfAndSiblingsToDrop)
        .Case("uvm_pkg::uvm_printer::print_field_int",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_field",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_generic_element",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_generic",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_time",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_string",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_real",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_header",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_footer",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_array_range",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_object_header",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_printer::print_object",
              UvmFastPathAction::PrinterNoOp)
        .Case("uvm_pkg::uvm_report_object::uvm_report_info",
              UvmFastPathAction::ReportInfoSuppress)
        .Case("uvm_pkg::uvm_report_object::uvm_report_warning",
              UvmFastPathAction::ReportWarningSuppress)
        .Default(UvmFastPathAction::None);
  }
  return UvmFastPathAction::None;
}

static uint64_t sequencerProcKey(ProcessId procId) {
  return 0xF1F1000000000000ULL | static_cast<uint64_t>(procId);
}

static bool disableAllUvmFastPaths() {
  const char *env = std::getenv("CIRCT_SIM_DISABLE_UVM_FASTPATHS");
  return env && env[0] != '\0' && env[0] != '0';
}

static bool enableUvmComponentChildFastPaths() {
  const char *env =
      std::getenv("CIRCT_SIM_ENABLE_UVM_COMPONENT_CHILD_FASTPATHS");
  return env && env[0] != '\0' && env[0] != '0';
}
} // namespace

bool LLHDProcessInterpreter::handleUvmWaitForSelfAndSiblingsToDrop(
    ProcessId procId, uint64_t phaseAddr, mlir::Operation *callOp) {
  if (disableAllUvmFastPaths())
    return false;
  noteUvmFastPathActionHit("uvm.wait_for_self_and_siblings_to_drop");
  if (phaseAddr == 0 || !callOp)
    return true;

  // Look up or create the objection handle for this phase.
  MooreObjectionHandle handle = MOORE_OBJECTION_INVALID_HANDLE;
  auto it = phaseObjectionHandles.find(phaseAddr);
  if (it != phaseObjectionHandles.end()) {
    handle = it->second;
  } else {
    handle = __moore_objection_create("", 0);
    phaseObjectionHandles[phaseAddr] = handle;
  }

  // Check current objection count.
  int64_t count = 0;
  if (handle != MOORE_OBJECTION_INVALID_HANDLE)
    count = __moore_objection_get_count(handle);

  if (count <= 0) {
    // Yield repeatedly to give forked task processes a chance to raise.
    int &yields = phaseWaitYieldCountByProc[procId];
    if (yields >= 10) {
      yields = 0;
      return true;
    }
    ++yields;
  } else {
    phaseWaitYieldCountByProc[procId] = 0;
  }

  // Suspend and poll later.
  auto &state = processStates[procId];
  state.waiting = true;

  SimTime currentTime = scheduler.getCurrentTime();
  constexpr int64_t kObjectionPollDelayFs = 10000000; // 10 ps

  // Polling every delta while objections are raised creates a large
  // zero-time spin in real UVM workloads. Once objections are non-zero,
  // poll on a small physical-time delay instead of delta-cycling.
  SimTime targetTime =
      count > 0 ? currentTime.advanceTime(kObjectionPollDelayFs)
                : currentTime.nextDelta();

  auto callIt = mlir::Block::iterator(callOp);
  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::Active,
      Event([this, procId, callIt]() {
        auto &st = processStates[procId];
        st.waiting = false;
        st.currentOp = callIt;
        scheduler.scheduleProcess(procId, SchedulingRegion::Active);
      }));
  return true;
}

bool LLHDProcessInterpreter::handleUvmFuncBodyFastPath(
    ProcessId procId, mlir::func::FuncOp funcOp,
    llvm::ArrayRef<InterpretedValue> args,
    llvm::SmallVectorImpl<InterpretedValue> &results, mlir::Operation *callOp) {
  if (disableAllUvmFastPaths())
    return false;
  // Fast exit for functions known to have no fast-path match.
  // Avoids re-running StringSwitch + matchesMethod on every call.
  if (funcBodyNoFastPathSet.contains(funcOp.getOperation())) {
    ++funcBodyFastPathCacheSkips;
    return false;
  }
  llvm::StringRef funcName = funcOp.getSymName();
  auto matchesMethod = [&](llvm::StringRef method) {
    return funcName.ends_with(method) || funcName.contains(method);
  };

  auto appendBoolResult = [&](bool value) {
    if (funcOp.getNumResults() < 1)
      return;
    unsigned width = std::max(1u, getTypeWidth(funcOp.getResultTypes()[0]));
    results.push_back(InterpretedValue(llvm::APInt(width, value ? 1 : 0)));
    for (unsigned i = 1, e = funcOp.getNumResults(); i < e; ++i) {
      unsigned extraWidth =
          std::max(1u, getTypeWidth(funcOp.getResultTypes()[i]));
      results.push_back(
          InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
  };

  auto writePointerToOutAddr = [&](const InterpretedValue &outAddrVal,
                                   uint64_t ptrValue) {
    if (outAddrVal.isX())
      return;
    uint64_t addr = outAddrVal.getUInt64();
    if (!addr)
      return;

    uint64_t offset = 0;
    MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);
    if (!block)
      block = findBlockByAddress(addr, offset);
    if (block && offset + 8 <= block->size) {
      for (unsigned i = 0; i < 8; ++i)
        block->bytes()[offset + i] =
            static_cast<uint8_t>((ptrValue >> (i * 8)) & 0xFF);
      block->initialized = true;
      return;
    }

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
        nativeOffset + 8 <= nativeSize)
      std::memcpy(reinterpret_cast<void *>(addr), &ptrValue, 8);
  };

  auto waitOnHopperData = [&](uint64_t hopperAddr) -> bool {
    if (!callOp)
      return false;
    auto &state = processStates[procId];
    state.waiting = true;
    state.sequencerGetRetryCallOp = callOp;
    auto &waiters = phaseHopperWaiters[hopperAddr];
    if (std::find(waiters.begin(), waiters.end(), procId) == waiters.end())
      waiters.push_back(procId);
    return true;
  };

  auto wakeHopperWaiters = [&](uint64_t hopperAddr) {
    auto waitIt = phaseHopperWaiters.find(hopperAddr);
    if (waitIt == phaseHopperWaiters.end())
      return;
    auto waiters = waitIt->second;
    phaseHopperWaiters.erase(waitIt);
    for (ProcessId waiterProc : waiters) {
      auto stateIt = processStates.find(waiterProc);
      if (stateIt == processStates.end())
        continue;
      if (!stateIt->second.waiting)
        continue;
      stateIt->second.waiting = false;
      scheduler.scheduleProcess(waiterProc, SchedulingRegion::Active);
    }
  };

  auto resolveVtableSlotFunc = [&](uint64_t objAddr, uint64_t slot,
                                   mlir::func::FuncOp &outFunc) -> bool {
    if (objAddr == 0)
      return false;

    uint64_t objOff = 0;
    MemoryBlock *objBlk = findMemoryBlockByAddress(objAddr + 4, procId, &objOff);
    if (!objBlk)
      objBlk = findBlockByAddress(objAddr + 4, objOff);
    if (!objBlk || !objBlk->initialized || objOff + 8 > objBlk->size)
      return false;

    uint64_t vtableAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      vtableAddr |= static_cast<uint64_t>(objBlk->bytes()[objOff + i]) << (i * 8);
    if (vtableAddr == 0)
      return false;

    uint64_t entryOff = 0;
    MemoryBlock *entryBlk =
        findMemoryBlockByAddress(vtableAddr + slot * 8, procId, &entryOff);
    if (!entryBlk)
      entryBlk = findBlockByAddress(vtableAddr + slot * 8, entryOff);
    if (!entryBlk || !entryBlk->initialized ||
        entryOff + 8 > entryBlk->size)
      return false;

    uint64_t funcAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      funcAddr |= static_cast<uint64_t>(entryBlk->bytes()[entryOff + i]) << (i * 8);
    auto addrIt = addressToFunction.find(funcAddr);
    if (addrIt == addressToFunction.end())
      return false;

    outFunc = rootModule.lookupSymbol<mlir::func::FuncOp>(addrIt->second);
    return static_cast<bool>(outFunc);
  };

  auto registerFactoryWrapper = [&](uint64_t wrapperAddr) -> bool {
    mlir::func::FuncOp getTypeNameFunc;
    if (!resolveVtableSlotFunc(wrapperAddr, /*slot=*/2, getTypeNameFunc))
      return false;

    llvm::SmallVector<InterpretedValue, 1> typeNameResults;
    if (failed(interpretFuncBody(
            procId, getTypeNameFunc, {InterpretedValue(wrapperAddr, 64)},
            typeNameResults, nullptr)) ||
        typeNameResults.empty())
      return false;

    uint64_t strPtr = 0;
    uint64_t strLenBits = 0;
    if (!decodePackedPtrLenPayload(typeNameResults.front(), strPtr, strLenBits))
      return false;
    int64_t strLen = static_cast<int64_t>(strLenBits);
    std::string typeName;
    if (strPtr == 0 || strLen <= 0 || strLen > 1024 ||
        !tryReadStringKey(procId, strPtr, strLen, typeName) || typeName.empty())
      return false;

    nativeFactoryTypeNames[typeName] = wrapperAddr;
    return true;
  };

  auto fastCreateComponentByName = [&](const InterpretedValue &typeNameArg,
                                       const InterpretedValue &instNameArg,
                                       const InterpretedValue &parentArg,
                                       InterpretedValue &outValue) -> bool {
    uint64_t strPtr = 0;
    uint64_t strLenBits = 0;
    if (!decodePackedPtrLenPayload(typeNameArg, strPtr, strLenBits))
      return false;
    int64_t strLen = static_cast<int64_t>(strLenBits);
    std::string typeName;
    if (strPtr == 0 || strLen <= 0 || strLen > 1024 ||
        !tryReadStringKey(procId, strPtr, strLen, typeName) || typeName.empty())
      return false;

    auto typeIt = nativeFactoryTypeNames.find(typeName);
    if (typeIt == nativeFactoryTypeNames.end())
      return false;
    uint64_t wrapperAddr = typeIt->second;

    mlir::func::FuncOp createComponentFunc;
    if (!resolveVtableSlotFunc(wrapperAddr, /*slot=*/1, createComponentFunc))
      return false;

    llvm::SmallVector<InterpretedValue, 1> createResults;
    if (failed(interpretFuncBody(
            procId, createComponentFunc,
            {InterpretedValue(wrapperAddr, 64), instNameArg, parentArg},
            createResults, callOp)) ||
        createResults.empty())
      return false;

    outValue = createResults.front();
    return true;
  };

  auto hashPhaseAddArgs = [&](llvm::ArrayRef<InterpretedValue> values) -> uint64_t {
    uint64_t hash = 0x517cc1b727220a95ULL;
    for (const auto &value : values.take_front(7)) {
      uint64_t bits = value.isX() ? 0xDEADBEEFDEADBEEFULL : value.getUInt64();
      hash ^= bits + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }
    return hash;
  };

  auto isSequencerHandshakeFunc = [&](llvm::StringRef suffix) {
    if (!(funcName.contains("uvm_sequencer") ||
          funcName.contains("sqr_if_base")))
      return false;
    return funcName.ends_with(suffix);
  };

  auto invokeVirtualNoArgMethodSlot = [&](uint64_t selfAddr,
                                          uint64_t slot) -> bool {
    mlir::func::FuncOp targetFunc;
    if (!resolveVtableSlotFunc(selfAddr, slot, targetFunc))
      return false;
    llvm::SmallVector<InterpretedValue, 2> ignoredResults;
    return succeeded(interpretFuncBody(
        procId, targetFunc, {InterpretedValue(selfAddr, 64)}, ignoredResults,
        callOp));
  };

  // Fast-path uvm_sequence_base::start for interpreted-mode UVM workloads.
  // The lowered start() body uses deep fork/process-guard plumbing that can
  // leave sub-sequences parked forever in interpreted mode. Execute the
  // virtual sequence hooks directly through runtime vtable slots instead.
  static bool enableSequenceStartFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_ENABLE_UVM_SEQUENCE_START_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (enableSequenceStartFastPath &&
      matchesMethod("uvm_sequence_base::start") && args.size() >= 1 &&
      !args[0].isX()) {
    uint64_t selfAddr = args[0].getUInt64();
    if (selfAddr != 0) {
      // Keep item context (self,parent,sequencer) consistent with normal
      // start() setup before entering virtual hooks.
      if (args.size() >= 3) {
        if (auto setItemContextFunc =
                rootModule.lookupSymbol<mlir::func::FuncOp>(
                    "uvm_pkg::uvm_sequence_item::set_item_context")) {
          llvm::SmallVector<InterpretedValue, 1> ignored;
          (void)interpretFuncBody(procId, setItemContextFunc,
                                  {args[0], args[2], args[1]}, ignored,
                                  callOp);
        }
      }

      // best-effort: clear stale response queue before running body.
      (void)invokeVirtualNoArgMethodSlot(selfAddr, /*clear_response_queue=*/56);

      bool callPrePost = args.size() >= 5 && !args[4].isX() &&
                         args[4].getUInt64() != 0;
      if (callPrePost) {
        (void)invokeVirtualNoArgMethodSlot(selfAddr, /*pre_start=*/39);
        (void)invokeVirtualNoArgMethodSlot(selfAddr, /*pre_body=*/40);
      }

      // Always execute the runtime-overridden body implementation.
      (void)invokeVirtualNoArgMethodSlot(selfAddr, /*body=*/43);

      if (callPrePost) {
        (void)invokeVirtualNoArgMethodSlot(selfAddr, /*post_body=*/45);
        (void)invokeVirtualNoArgMethodSlot(selfAddr, /*post_start=*/46);
      }

      noteUvmFastPathActionHit("func.body.sequence.start");
      return true;
    }
  }

  // Dispatch-path agnostic sequencer handshake intercepts.
  // These mirror call_indirect intercepts but run at function entry so cache
  // fast paths and wrapper hops cannot bypass the rendezvous logic.
  if (isSequencerHandshakeFunc("::wait_for_grant")) {
    if (!args.empty() && !args[0].isX()) {
      uint64_t sqrAddr = normalizeUvmSequencerAddress(procId, args[0].getUInt64());
      if (sqrAddr != 0)
        itemToSequencer[sequencerProcKey(procId)] = sqrAddr;
    }
    noteUvmFastPathActionHit("func.body.sequencer.wait_for_grant");
    return true;
  }

  if (isSequencerHandshakeFunc("::send_request") && args.size() >= 3) {
    uint64_t sqrAddr = args[0].isX() ? 0
                                     : normalizeUvmSequencerAddress(
                                           procId, args[0].getUInt64());
    uint64_t seqAddr = args[1].isX()
                           ? 0
                           : normalizeUvmObjectKey(procId, args[1].getUInt64());
    uint64_t itemAddr = args[2].isX() ? 0 : args[2].getUInt64();
    uint64_t queueAddr = 0;
    if (itemAddr != 0) {
      if (auto ownerIt = itemToSequencer.find(itemAddr);
          ownerIt != itemToSequencer.end())
        queueAddr = ownerIt->second;
    }
    if (queueAddr == 0) {
      if (auto procIt = itemToSequencer.find(sequencerProcKey(procId));
          procIt != itemToSequencer.end())
        queueAddr = procIt->second;
    }
    if (queueAddr == 0)
      queueAddr = sqrAddr;
    queueAddr = normalizeUvmSequencerAddress(procId, queueAddr);
    if (itemAddr != 0 && queueAddr != 0) {
      sequencerItemFifo[queueAddr].push_back(itemAddr);
      recordUvmSequencerItemOwner(itemAddr, queueAddr);
      sequencePendingItemsByProc[procId].push_back(itemAddr);
      if (seqAddr != 0)
        sequencePendingItemsBySeq[seqAddr].push_back(itemAddr);
      wakeUvmSequencerGetWaiterForPush(queueAddr);
      if (traceSeqEnabled) {
        llvm::errs() << "[SEQ-FBODY] send_request item=0x"
                     << llvm::utohexstr(itemAddr) << " sqr=0x"
                     << llvm::utohexstr(queueAddr) << " depth="
                     << sequencerItemFifo[queueAddr].size() << "\n";
      }
    }
    noteUvmFastPathActionHit("func.body.sequencer.send_request");
    return true;
  }

  if (isSequencerHandshakeFunc("::wait_for_item_done")) {
    uint64_t seqAddr =
        args.size() > 1 && !args[1].isX()
            ? normalizeUvmObjectKey(procId, args[1].getUInt64())
            : 0;
    uint64_t itemAddr = 0;
    auto seqIt = sequencePendingItemsBySeq.end();
    if (seqAddr != 0) {
      seqIt = sequencePendingItemsBySeq.find(seqAddr);
      if (seqIt != sequencePendingItemsBySeq.end() && !seqIt->second.empty())
        itemAddr = seqIt->second.front();
    }
    auto procIt = sequencePendingItemsByProc.find(procId);
    if (itemAddr == 0 && procIt != sequencePendingItemsByProc.end() &&
        !procIt->second.empty())
      itemAddr = procIt->second.front();
    if (itemAddr == 0) {
      noteUvmFastPathActionHit("func.body.sequencer.wait_for_item_done_empty");
      return true;
    }

    auto erasePendingItemByProc = [&](uint64_t item) {
      auto it = sequencePendingItemsByProc.find(procId);
      if (it == sequencePendingItemsByProc.end())
        return;
      auto &pending = it->second;
      auto match = std::find(pending.begin(), pending.end(), item);
      if (match != pending.end())
        pending.erase(match);
      if (pending.empty())
        sequencePendingItemsByProc.erase(it);
    };
    auto erasePendingItemBySeq = [&](uint64_t seqKey, uint64_t item) {
      if (seqKey == 0)
        return;
      auto it = sequencePendingItemsBySeq.find(seqKey);
      if (it == sequencePendingItemsBySeq.end())
        return;
      auto &pending = it->second;
      auto match = std::find(pending.begin(), pending.end(), item);
      if (match != pending.end())
        pending.erase(match);
      if (pending.empty())
        sequencePendingItemsBySeq.erase(it);
    };

    if (itemDoneReceived.count(itemAddr)) {
      erasePendingItemByProc(itemAddr);
      erasePendingItemBySeq(seqAddr, itemAddr);
      itemDoneReceived.erase(itemAddr);
      finishItemWaiters.erase(itemAddr);
      (void)takeUvmSequencerItemOwner(itemAddr);
      noteUvmFastPathActionHit("func.body.sequencer.wait_for_item_done_done");
      return true;
    }

    finishItemWaiters[itemAddr] = procId;
    auto &state = processStates[procId];
    state.waiting = true;
    state.sequencerGetRetryCallOp = callOp;
    if (traceSeqEnabled)
      llvm::errs() << "[SEQ-FBODY] wait_for_item_done item=0x"
                   << llvm::utohexstr(itemAddr) << "\n";
    noteUvmFastPathActionHit("func.body.sequencer.wait_for_item_done_wait");
    return true;
  }

  if ((funcName.contains("uvm_pkg::uvm_object_registry_") ||
       funcName.contains("uvm_pkg::uvm_component_registry_") ||
       funcName.contains("uvm_pkg::uvm_abstract_object_registry_")) &&
      funcName.ends_with("::initialize") && !args.empty() && !args[0].isX()) {
    uint64_t wrapperAddr = args[0].getUInt64();
    if (wrapperAddr != 0 &&
        nativeFactoryInitializedWrappers.contains(wrapperAddr)) {
      noteUvmFastPathActionHit("func.body.registry.initialize_cached");
      return true;
    }
    if (wrapperAddr != 0 && registerFactoryWrapper(wrapperAddr)) {
      nativeFactoryInitializedWrappers.insert(wrapperAddr);
      noteUvmFastPathActionHit("func.body.registry.initialize_register");
      return true;
    }
  }

  if ((matchesMethod("uvm_default_factory::create_component_by_name") ||
       matchesMethod("uvm_factory::create_component_by_name")) &&
      args.size() >= 5 && funcOp.getNumResults() >= 1) {
    InterpretedValue createdValue;
    if (fastCreateComponentByName(args[1], args[3], args[4], createdValue)) {
      unsigned resultWidth =
          std::max(1u, getTypeWidth(funcOp.getResultTypes().front()));
      if (createdValue.isX()) {
        results.push_back(InterpretedValue::makeX(resultWidth));
      } else if (createdValue.getWidth() != resultWidth) {
        results.push_back(InterpretedValue(
            createdValue.getAPInt().zextOrTrunc(resultWidth)));
      } else {
        results.push_back(createdValue);
      }
      for (unsigned i = 1, e = funcOp.getNumResults(); i < e; ++i) {
        unsigned extraWidth =
            std::max(1u, getTypeWidth(funcOp.getResultTypes()[i]));
        results.push_back(InterpretedValue(llvm::APInt::getZero(extraWidth)));
      }
      noteUvmFastPathActionHit("func.body.factory.create_component_by_name");
      return true;
    }
  }

  if (funcName.contains("uvm_pkg::uvm_phase::add") && args.size() >= 7) {
    bool allOptionalNull = true;
    for (unsigned i = 2; i < 7; ++i) {
      if (args[i].isX() || args[i].getUInt64() != 0) {
        allOptionalNull = false;
        break;
      }
    }
    if (allOptionalNull && !args[0].isX() && !args[1].isX() &&
        args[0].getUInt64() != 0 && args[1].getUInt64() != 0) {
      uint64_t edgeKey = hashPhaseAddArgs(args);
      if (!nativePhaseAddEdgeKeys.insert(edgeKey).second) {
        noteUvmFastPathActionHit("func.body.phase.add_duplicate");
        return true;
      }
      noteUvmFastPathActionHit("func.body.phase.add_observed");
    }
  }

  // m_uvm_get_root fast-path is opt-in. Returning
  // m_inst too early can race root construction and leave phase/domain state
  // partially initialized for subsequent component::new calls.
  static bool enableUvmGetRootFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_ENABLE_UVM_GET_ROOT_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (enableUvmGetRootFastPath && matchesMethod("m_uvm_get_root") &&
      funcOp.getNumResults() >= 1) {
    // Look up uvm_root::m_inst global for the root singleton pointer.
    uint64_t rootAddr = 0;
    auto *addressofOp = rootModule.lookupSymbol(
        "uvm_pkg::uvm_pkg::uvm_root::m_inst");
    if (addressofOp) {
      auto globalIt = globalMemoryBlocks.find(
          "uvm_pkg::uvm_pkg::uvm_root::m_inst");
      if (globalIt != globalMemoryBlocks.end() &&
          globalIt->second.initialized &&
          globalIt->second.size >= 8) {
        for (unsigned i = 0; i < 8; ++i)
          rootAddr |= static_cast<uint64_t>(globalIt->second[i]) << (i * 8);
      }
    }
    if (rootAddr != 0) {
      unsigned width = std::max(1u, getTypeWidth(funcOp.getResultTypes()[0]));
      results.push_back(InterpretedValue(llvm::APInt(width, rootAddr)));
      noteUvmFastPathActionHit("func.body.m_uvm_get_root");
      return true;
    }
  }

  // m_execute_scheduled_forks: no-op. The interpreter handles fork scheduling
  // directly; this UVM infrastructure method can be safely bypassed.
  if (matchesMethod("m_execute_scheduled_forks")) {
    noteUvmFastPathActionHit("func.body.m_execute_scheduled_forks");
    return true;
  }

  if (matchesMethod("uvm_phase_hopper::try_put") && args.size() >= 2) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    uint64_t phaseAddr = args[1].isX() ? 0 : args[1].getUInt64();
    phaseHopperQueue[hopperAddr].push_back(phaseAddr);
    wakeHopperWaiters(hopperAddr);

    if (hopperAddr != 0) {
      auto it = phaseObjectionHandles.find(hopperAddr);
      int64_t handle = 0;
      if (it != phaseObjectionHandles.end()) {
        handle = it->second;
      } else {
        std::string hopperName = "phase_hopper_" + std::to_string(hopperAddr);
        handle = __moore_objection_create(
            hopperName.c_str(), static_cast<int64_t>(hopperName.size()));
        phaseObjectionHandles[hopperAddr] = handle;
      }
      __moore_objection_raise(handle, "", 0, "", 0, 1);
    }

    appendBoolResult(true);
    noteUvmFastPathActionHit("func.body.phase_hopper.try_put");
    return true;
  }

  if (matchesMethod("uvm_phase_hopper::try_get") && args.size() >= 2) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    uint64_t phaseAddr = 0;
    bool hasPhase = false;
    auto it = phaseHopperQueue.find(hopperAddr);
    if (it != phaseHopperQueue.end() && !it->second.empty()) {
      phaseAddr = it->second.front();
      it->second.pop_front();
      hasPhase = true;
    }
    writePointerToOutAddr(args[1], phaseAddr);
    // Drop the objection that try_put raised when the phase was enqueued.
    if (hasPhase && hopperAddr != 0) {
      auto objIt = phaseObjectionHandles.find(hopperAddr);
      if (objIt != phaseObjectionHandles.end())
        __moore_objection_drop(objIt->second, "", 0, "", 0, 1);
    }
    appendBoolResult(hasPhase);
    noteUvmFastPathActionHit("func.body.phase_hopper.try_get");
    return true;
  }

  if (matchesMethod("uvm_phase_hopper::try_peek") && args.size() >= 2) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    uint64_t phaseAddr = 0;
    bool hasPhase = false;
    auto it = phaseHopperQueue.find(hopperAddr);
    if (it != phaseHopperQueue.end() && !it->second.empty()) {
      phaseAddr = it->second.front();
      hasPhase = true;
    }
    writePointerToOutAddr(args[1], phaseAddr);
    appendBoolResult(hasPhase);
    noteUvmFastPathActionHit("func.body.phase_hopper.try_peek");
    return true;
  }

  if (matchesMethod("uvm_phase_hopper::peek") && args.size() >= 2) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    auto it = phaseHopperQueue.find(hopperAddr);
    if (it != phaseHopperQueue.end() && !it->second.empty()) {
      writePointerToOutAddr(args[1], it->second.front());
      noteUvmFastPathActionHit("func.body.phase_hopper.peek");
      return true;
    }
    if (!waitOnHopperData(hopperAddr))
      return false;
    noteUvmFastPathActionHit("func.body.phase_hopper.peek_retry");
    return true;
  }

  if (matchesMethod("uvm_phase_hopper::get") && args.size() >= 2) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    // Decline if the output pointer is null/unwritable — the caller passed
    // an invalid destination and we must not drain the queue.
    uint64_t outAddr = args[1].isX() ? 0 : args[1].getUInt64();
    if (outAddr == 0)
      return false;
    auto it = phaseHopperQueue.find(hopperAddr);
    if (it != phaseHopperQueue.end() && !it->second.empty()) {
      uint64_t phaseAddr = it->second.front();
      it->second.pop_front();
      writePointerToOutAddr(args[1], phaseAddr);
      // Drop the objection that try_put raised.
      if (hopperAddr != 0) {
        auto objIt = phaseObjectionHandles.find(hopperAddr);
        if (objIt != phaseObjectionHandles.end())
          __moore_objection_drop(objIt->second, "", 0, "", 0, 1);
      }
      noteUvmFastPathActionHit("func.body.phase_hopper.get");
      return true;
    }
    if (!waitOnHopperData(hopperAddr))
      return false;
    noteUvmFastPathActionHit("func.body.phase_hopper.get_retry");
    return true;
  }

  // Return the objection handle associated with a phase hopper.
  if (matchesMethod("uvm_phase_hopper::get_objection") && args.size() >= 1 &&
      funcOp.getNumResults() >= 1) {
    uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
    uint64_t objHandle = 0;
    auto it = phaseObjectionHandles.find(hopperAddr);
    if (it != phaseObjectionHandles.end())
      objHandle = static_cast<uint64_t>(it->second);
    // Keep phase-hopper objection handles ABI-compatible with the normal
    // call-site interception path by returning the synthetic pointer encoding.
    uint64_t syntheticAddr = 0;
    if (objHandle != 0)
      syntheticAddr = 0xE0000000ULL + objHandle;
    unsigned width = std::max(1u, getTypeWidth(funcOp.getResultTypes()[0]));
    results.push_back(InterpretedValue(llvm::APInt(width, syntheticAddr)));
    noteUvmFastPathActionHit("func.body.phase_hopper.get_objection");
    return true;
  }

  // No fast path matched — cache this FuncOp as a negative result.
  funcBodyNoFastPathSet.insert(funcOp.getOperation());
  return false;
}

bool LLHDProcessInterpreter::handleUvmCallIndirectFastPath(
    ProcessId procId, mlir::func::CallIndirectOp callIndirectOp,
    llvm::StringRef calleeName) {
  if (disableAllUvmFastPaths())
    return false;
  auto recordFastPathHit = [&](llvm::StringRef key) {
    noteUvmFastPathActionHit(key);
  };

  auto zeroCallIndirectResults = [&]() {
    for (Value result : callIndirectOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
  };

  auto setCallIndirectResultInt = [&](Value result, uint64_t value) {
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, value)));
  };

  auto readU64AtAddr = [&](uint64_t addr, uint64_t &out) -> bool {
    uint64_t offset = 0;
    MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);
    if (!block)
      block = findBlockByAddress(addr, offset);
    if (block && block->initialized && offset + 8 <= block->size) {
      out = 0;
      for (unsigned i = 0; i < 8; ++i)
        out |= static_cast<uint64_t>(block->bytes()[offset + i]) << (i * 8);
      return true;
    }

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
        nativeOffset + 8 <= nativeSize) {
      std::memcpy(&out, reinterpret_cast<void *>(addr), 8);
      return true;
    }
    return false;
  };

  auto writeStringStructToRef = [&](Value outRef, uint64_t strPtr,
                                    int64_t strLen) -> bool {
    InterpretedValue refAddrVal = getValue(procId, outRef);
    if (refAddrVal.isX())
      return false;
    uint64_t addr = refAddrVal.getUInt64();
    if (!addr)
      return false;

    uint64_t offset = 0;
    MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);
    if (!block)
      block = findBlockByAddress(addr, offset);
    if (block && offset + 16 <= block->size) {
      std::memcpy(block->bytes() + offset, &strPtr, 8);
      std::memcpy(block->bytes() + offset + 8, &strLen, 8);
      block->initialized = true;
      return true;
    }

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
        nativeOffset + 16 <= nativeSize) {
      std::memcpy(reinterpret_cast<void *>(addr), &strPtr, 8);
      std::memcpy(reinterpret_cast<void *>(addr + 8), &strLen, 8);
      return true;
    }
    return false;
  };

  auto readPackedStringParts = [&](const InterpretedValue &strVal,
                                   uint64_t &strPtr,
                                   int64_t &strLen) -> bool {
    strPtr = 0;
    strLen = 0;
    uint64_t strLenBits = 0;
    if (!decodePackedPtrLenPayload(strVal, strPtr, strLenBits))
      return false;
    strLen = static_cast<int64_t>(strLenBits);
    if (strLen < 0 || strLen > 4096)
      return false;
    return true;
  };

  auto readRefStringParts = [&](Value refArg, uint64_t &strPtr,
                                int64_t &strLen) -> bool {
    strPtr = 0;
    strLen = 0;
    InterpretedValue refAddrVal = getValue(procId, refArg);
    if (refAddrVal.isX())
      return false;
    uint64_t refAddr = refAddrVal.getUInt64();
    if (!refAddr)
      return true;
    uint64_t strLenBits = 0;
    if (!readU64AtAddr(refAddr, strPtr) ||
        !readU64AtAddr(refAddr + 8, strLenBits))
      return false;
    strLen = static_cast<int64_t>(strLenBits);
    return strLen >= 0 && strLen <= 4096;
  };

  auto getComponentChildrenAssocAddr = [&](uint64_t selfAddr,
                                           uint64_t &assocAddr) -> bool {
    assocAddr = 0;
    if (selfAddr < 0x1000)
      return false;
    constexpr uint64_t kChildrenOff = 95;
    if (!readU64AtAddr(selfAddr + kChildrenOff, assocAddr))
      return false;
    if (assocAddr == 0)
      return true;
    return validAssocArrayAddresses.contains(assocAddr);
  };

  if (enableUvmComponentChildFastPaths()) {
    if (calleeName.contains("uvm_component::get_num_children") &&
        callIndirectOp.getArgOperands().size() >= 1 &&
        callIndirectOp.getNumResults() >= 1) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        return false;
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
        return false;

      uint64_t count = 0;
      if (assocAddr != 0)
        count = static_cast<uint64_t>(
            std::max<int64_t>(
                0, __moore_assoc_size(reinterpret_cast<void *>(assocAddr))));
      setCallIndirectResultInt(callIndirectOp.getResult(0), count);
      recordFastPathHit("call_indirect.component.get_num_children");
      return true;
    }

    if (calleeName.contains("uvm_component::has_child") &&
        callIndirectOp.getArgOperands().size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        return false;
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
        return false;
      uint64_t keyPtr = 0;
      int64_t keyLen = 0;
      if (!readPackedStringParts(
              getValue(procId, callIndirectOp.getArgOperands()[1]), keyPtr,
              keyLen))
        return false;

      std::string keyStorage;
      if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
        return false;

      bool exists = false;
      if (assocAddr != 0) {
        MooreString key{const_cast<char *>(keyStorage.data()),
                        static_cast<int64_t>(keyStorage.size())};
        exists =
            __moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key) != 0;
      }
      setCallIndirectResultInt(callIndirectOp.getResult(0), exists ? 1 : 0);
      recordFastPathHit("call_indirect.component.has_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_child") &&
        callIndirectOp.getArgOperands().size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        return false;
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
        return false;
      uint64_t keyPtr = 0;
      int64_t keyLen = 0;
      if (!readPackedStringParts(
              getValue(procId, callIndirectOp.getArgOperands()[1]), keyPtr,
              keyLen))
        return false;

      std::string keyStorage;
      if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
        return false;

      uint64_t childAddr = 0;
      if (assocAddr != 0) {
        MooreString key{const_cast<char *>(keyStorage.data()),
                        static_cast<int64_t>(keyStorage.size())};
        if (__moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key)) {
          void *ref = __moore_assoc_get_ref(reinterpret_cast<void *>(assocAddr),
                                            &key,
                                            /*value_size=*/8);
          if (ref)
            std::memcpy(&childAddr, ref, 8);
        }
      }
      setCallIndirectResultInt(callIndirectOp.getResult(0), childAddr);
      recordFastPathHit("call_indirect.component.get_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_first_child") &&
        callIndirectOp.getArgOperands().size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        return false;
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
        return false;
      if (assocAddr == 0) {
        setCallIndirectResultInt(callIndirectOp.getResult(0), 0);
        recordFastPathHit("call_indirect.component.get_first_child_empty");
        return true;
      }

      MooreString keyOut{nullptr, 0};
      bool ok = __moore_assoc_first(reinterpret_cast<void *>(assocAddr), &keyOut);
      if (!ok) {
        setCallIndirectResultInt(callIndirectOp.getResult(0), 0);
        recordFastPathHit("call_indirect.component.get_first_child_empty");
        return true;
      }
      if (keyOut.data && keyOut.len > 0)
        dynamicStrings[reinterpret_cast<uint64_t>(keyOut.data)] = {keyOut.data,
                                                                   keyOut.len};
      if (!writeStringStructToRef(callIndirectOp.getArgOperands()[1],
                                  reinterpret_cast<uint64_t>(keyOut.data),
                                  keyOut.len))
        return false;

      setCallIndirectResultInt(callIndirectOp.getResult(0), 1);
      recordFastPathHit("call_indirect.component.get_first_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_next_child") &&
        callIndirectOp.getArgOperands().size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        return false;
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
        return false;
      if (assocAddr == 0) {
        setCallIndirectResultInt(callIndirectOp.getResult(0), 0);
        recordFastPathHit("call_indirect.component.get_next_child_end");
        return true;
      }

      uint64_t curPtr = 0;
      int64_t curLen = 0;
      if (!readRefStringParts(callIndirectOp.getArgOperands()[1], curPtr,
                              curLen))
        return false;

      MooreString keyRef{reinterpret_cast<char *>(curPtr), curLen};
      bool ok = __moore_assoc_next(reinterpret_cast<void *>(assocAddr), &keyRef);
      if (!ok) {
        setCallIndirectResultInt(callIndirectOp.getResult(0), 0);
        recordFastPathHit("call_indirect.component.get_next_child_end");
        return true;
      }
      if (keyRef.data && keyRef.len > 0)
        dynamicStrings[reinterpret_cast<uint64_t>(keyRef.data)] = {keyRef.data,
                                                                   keyRef.len};
      if (!writeStringStructToRef(callIndirectOp.getArgOperands()[1],
                                  reinterpret_cast<uint64_t>(keyRef.data),
                                  keyRef.len))
        return false;

      setCallIndirectResultInt(callIndirectOp.getResult(0), 1);
      recordFastPathHit("call_indirect.component.get_next_child");
      return true;
    }
  }

  // First-tier registry dispatch keyed by (call form, exact symbol).
  switch (lookupUvmFastPath(UvmFastPathCallForm::CallIndirect, calleeName)) {
  case UvmFastPathAction::WaitForSelfAndSiblingsToDrop: {
    if (callIndirectOp.getArgOperands().empty())
      break;
    InterpretedValue phaseVal =
        getValue(procId, callIndirectOp.getArgOperands()[0]);
    uint64_t phaseAddr = phaseVal.isX() ? 0 : phaseVal.getUInt64();
    return handleUvmWaitForSelfAndSiblingsToDrop(
        procId, phaseAddr, callIndirectOp.getOperation());
  }
  case UvmFastPathAction::AdjustNamePassthrough: {
    if (callIndirectOp.getNumResults() < 1 ||
        callIndirectOp.getArgOperands().size() < 2)
      break;
    Value result0 = callIndirectOp.getResult(0);
    unsigned resultWidth = std::max(1u, getTypeWidth(result0.getType()));
    InterpretedValue nameVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    if (nameVal.isX()) {
      setValue(procId, result0, InterpretedValue::makeX(resultWidth));
    } else if (nameVal.getWidth() != resultWidth) {
      setValue(procId, result0,
               InterpretedValue(nameVal.getAPInt().zextOrTrunc(resultWidth)));
    } else {
      setValue(procId, result0, nameVal);
    }
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value result = callIndirectOp.getResult(i);
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry adjust_name passthrough: "
               << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.adjust_name");
    return true;
  }
  case UvmFastPathAction::PrinterNoOp:
    zeroCallIndirectResults();
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry printer no-op: " << calleeName
               << "\n");
    recordFastPathHit("registry.call_indirect.printer_noop");
    return true;
  case UvmFastPathAction::ReportInfoSuppress:
    if (!fastPathUvmReportInfo)
      break;
    zeroCallIndirectResults();
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_info suppress: "
               << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.report_info_suppress");
    return true;
  case UvmFastPathAction::ReportWarningSuppress:
    if (!fastPathUvmReportWarning)
      break;
    zeroCallIndirectResults();
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_warning suppress: "
               << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.report_warning_suppress");
    return true;
  case UvmFastPathAction::None:
    break;
  }

  if (calleeName.contains("wait_for_self_and_siblings_to_drop") &&
      callIndirectOp.getArgOperands().size() >= 1) {
    InterpretedValue phaseVal =
        getValue(procId, callIndirectOp.getArgOperands()[0]);
    uint64_t phaseAddr = phaseVal.isX() ? 0 : phaseVal.getUInt64();
    return handleUvmWaitForSelfAndSiblingsToDrop(
        procId, phaseAddr, callIndirectOp.getOperation());
  }

  // Intercept printer name adjustment. This is report-string formatting and
  // does not affect simulation behavior; preserve intent by returning the
  // input name unchanged.
  if (calleeName.contains("uvm_printer::adjust_name") &&
      callIndirectOp.getNumResults() >= 1 &&
      callIndirectOp.getArgOperands().size() >= 2) {
    Value result0 = callIndirectOp.getResult(0);
    unsigned resultWidth = std::max(1u, getTypeWidth(result0.getType()));
    InterpretedValue nameVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    if (nameVal.isX()) {
      setValue(procId, result0, InterpretedValue::makeX(resultWidth));
    } else if (nameVal.getWidth() != resultWidth) {
      setValue(procId, result0,
               InterpretedValue(nameVal.getAPInt().zextOrTrunc(resultWidth)));
    } else {
      setValue(procId, result0, nameVal);
    }
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value result = callIndirectOp.getResult(i);
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: adjust_name fast-path (passthrough): "
               << calleeName << "\n");
    return true;
  }

  // Intercept expensive UVM printer formatting calls. These build report
  // strings only and are safe to no-op for simulation behavior.
  if (calleeName.contains("uvm_printer::print_field_int") ||
      calleeName.contains("uvm_printer::print_field") ||
      calleeName.contains("uvm_printer::print_generic_element") ||
      calleeName.contains("uvm_printer::print_generic") ||
      calleeName.contains("uvm_printer::print_time") ||
      calleeName.contains("uvm_printer::print_string") ||
      calleeName.contains("uvm_printer::print_real") ||
      calleeName.contains("uvm_printer::print_array_header") ||
      calleeName.contains("uvm_printer::print_array_footer") ||
      calleeName.contains("uvm_printer::print_array_range") ||
      calleeName.contains("uvm_printer::print_object_header") ||
      calleeName.contains("uvm_printer::print_object")) {
    zeroCallIndirectResults();
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: printer format fast-path (no-op): "
               << calleeName << "\n");
    return true;
  }

  // Optional fast-path for low-severity report traffic. This bypasses
  // high-volume info/warning report formatting and dispatch logic while
  // preserving default behavior unless explicitly enabled.
  if ((fastPathUvmReportInfo &&
       calleeName.contains("uvm_report_object::uvm_report_info")) ||
      (fastPathUvmReportWarning &&
       calleeName.contains("uvm_report_object::uvm_report_warning"))) {
    zeroCallIndirectResults();
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: report traffic fast-path (suppressed): "
               << calleeName << "\n");
    return true;
  }

  return false;
}

bool LLHDProcessInterpreter::handleUvmFuncCallFastPath(
    ProcessId procId, mlir::func::CallOp callOp, llvm::StringRef calleeName) {
  if (disableAllUvmFastPaths())
    return false;
  auto recordFastPathHit = [&](llvm::StringRef key) {
    noteUvmFastPathActionHit(key);
  };

  auto setCallResultInt = [&](Value result, uint64_t value) {
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, value)));
  };

  auto readU64AtAddr = [&](uint64_t addr, uint64_t &out) -> bool {
    uint64_t offset = 0;
    MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);
    if (!block)
      block = findBlockByAddress(addr, offset);
    if (block && block->initialized && offset + 8 <= block->size) {
      out = 0;
      for (unsigned i = 0; i < 8; ++i)
        out |= static_cast<uint64_t>(block->bytes()[offset + i]) << (i * 8);
      return true;
    }

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
        nativeOffset + 8 <= nativeSize) {
      std::memcpy(&out, reinterpret_cast<void *>(addr), 8);
      return true;
    }
    return false;
  };

  auto writeStringStructToRef = [&](Value outRef, uint64_t strPtr,
                                    int64_t strLen) -> bool {
    InterpretedValue refAddrVal = getValue(procId, outRef);
    if (refAddrVal.isX())
      return false;
    uint64_t addr = refAddrVal.getUInt64();
    if (!addr)
      return false;

    uint64_t offset = 0;
    MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);
    if (!block)
      block = findBlockByAddress(addr, offset);
    if (block && offset + 16 <= block->size) {
      std::memcpy(block->bytes() + offset, &strPtr, 8);
      std::memcpy(block->bytes() + offset + 8, &strLen, 8);
      block->initialized = true;
      return true;
    }

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
        nativeOffset + 16 <= nativeSize) {
      std::memcpy(reinterpret_cast<void *>(addr), &strPtr, 8);
      std::memcpy(reinterpret_cast<void *>(addr + 8), &strLen, 8);
      return true;
    }
    return false;
  };

  auto readPackedStringParts = [&](const InterpretedValue &strVal,
                                   uint64_t &strPtr,
                                   int64_t &strLen) -> bool {
    strPtr = 0;
    strLen = 0;
    uint64_t strLenBits = 0;
    if (!decodePackedPtrLenPayload(strVal, strPtr, strLenBits))
      return false;
    strLen = static_cast<int64_t>(strLenBits);
    if (strLen < 0 || strLen > 4096)
      return false;
    return true;
  };

  auto readRefStringParts = [&](Value refArg, uint64_t &strPtr,
                                int64_t &strLen) -> bool {
    strPtr = 0;
    strLen = 0;
    InterpretedValue refAddrVal = getValue(procId, refArg);
    if (refAddrVal.isX())
      return false;
    uint64_t refAddr = refAddrVal.getUInt64();
    if (!refAddr)
      return true;
    uint64_t strLenBits = 0;
    if (!readU64AtAddr(refAddr, strPtr) ||
        !readU64AtAddr(refAddr + 8, strLenBits))
      return false;
    strLen = static_cast<int64_t>(strLenBits);
    return strLen >= 0 && strLen <= 4096;
  };

  auto getComponentChildrenAssocAddr = [&](uint64_t selfAddr,
                                           uint64_t &assocAddr) -> bool {
    assocAddr = 0;
    if (selfAddr < 0x1000)
      return false;
    // uvm_component::m_children (field[10]) in packed layout.
    constexpr uint64_t kChildrenOff = 95;
    if (!readU64AtAddr(selfAddr + kChildrenOff, assocAddr))
      return false;
    if (assocAddr == 0)
      return true;
    return validAssocArrayAddresses.contains(assocAddr);
  };

  if (enableUvmComponentChildFastPaths()) {
    if (calleeName.contains("uvm_component::get_num_children") &&
        callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
      InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
      if (selfVal.isX())
        return false;
      uint64_t selfAddr = selfVal.getUInt64();
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfAddr, assocAddr))
        return false;

      uint64_t count = 0;
      if (assocAddr != 0)
        count = static_cast<uint64_t>(
            std::max<int64_t>(
                0, __moore_assoc_size(reinterpret_cast<void *>(assocAddr))));
      setCallResultInt(callOp.getResult(0), count);
      recordFastPathHit("func.call.component.get_num_children");
      return true;
    }

    if (calleeName.contains("uvm_component::has_child") &&
        callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
      InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
      if (selfVal.isX())
        return false;
      uint64_t selfAddr = selfVal.getUInt64();
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfAddr, assocAddr))
        return false;
      uint64_t keyPtr = 0;
      int64_t keyLen = 0;
      if (!readPackedStringParts(getValue(procId, callOp.getOperand(1)), keyPtr,
                                 keyLen))
        return false;

      std::string keyStorage;
      if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
        return false;

      bool exists = false;
      if (assocAddr != 0) {
        MooreString key{const_cast<char *>(keyStorage.data()),
                        static_cast<int64_t>(keyStorage.size())};
        exists =
            __moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key) != 0;
      }
      setCallResultInt(callOp.getResult(0), exists ? 1 : 0);
      recordFastPathHit("func.call.component.has_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_child") &&
        callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
      InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
      if (selfVal.isX())
        return false;
      uint64_t selfAddr = selfVal.getUInt64();
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfAddr, assocAddr))
        return false;
      uint64_t keyPtr = 0;
      int64_t keyLen = 0;
      if (!readPackedStringParts(getValue(procId, callOp.getOperand(1)), keyPtr,
                                 keyLen))
        return false;

      std::string keyStorage;
      if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
        return false;

      uint64_t childAddr = 0;
      if (assocAddr != 0) {
        MooreString key{const_cast<char *>(keyStorage.data()),
                        static_cast<int64_t>(keyStorage.size())};
        if (__moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key)) {
          void *ref = __moore_assoc_get_ref(reinterpret_cast<void *>(assocAddr),
                                            &key,
                                            /*value_size=*/8);
          if (ref)
            std::memcpy(&childAddr, ref, 8);
        }
      }

      setCallResultInt(callOp.getResult(0), childAddr);
      recordFastPathHit("func.call.component.get_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_first_child") &&
        callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
      InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
      if (selfVal.isX())
        return false;
      uint64_t selfAddr = selfVal.getUInt64();
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfAddr, assocAddr))
        return false;

      if (assocAddr == 0) {
        setCallResultInt(callOp.getResult(0), 0);
        recordFastPathHit("func.call.component.get_first_child_empty");
        return true;
      }

      MooreString keyOut{nullptr, 0};
      bool ok = __moore_assoc_first(reinterpret_cast<void *>(assocAddr), &keyOut);
      if (!ok) {
        setCallResultInt(callOp.getResult(0), 0);
        recordFastPathHit("func.call.component.get_first_child_empty");
        return true;
      }
      if (keyOut.data && keyOut.len > 0)
        dynamicStrings[reinterpret_cast<uint64_t>(keyOut.data)] = {keyOut.data,
                                                                   keyOut.len};
      if (!writeStringStructToRef(callOp.getOperand(1),
                                  reinterpret_cast<uint64_t>(keyOut.data),
                                  keyOut.len))
        return false;

      setCallResultInt(callOp.getResult(0), ok ? 1 : 0);
      recordFastPathHit("func.call.component.get_first_child");
      return true;
    }

    if (calleeName.contains("uvm_component::get_next_child") &&
        callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
      InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
      if (selfVal.isX())
        return false;
      uint64_t selfAddr = selfVal.getUInt64();
      uint64_t assocAddr = 0;
      if (!getComponentChildrenAssocAddr(selfAddr, assocAddr))
        return false;
      if (assocAddr == 0) {
        setCallResultInt(callOp.getResult(0), 0);
        recordFastPathHit("func.call.component.get_next_child_end");
        return true;
      }

      uint64_t curPtr = 0;
      int64_t curLen = 0;
      if (!readRefStringParts(callOp.getOperand(1), curPtr, curLen))
        return false;

      MooreString keyRef{reinterpret_cast<char *>(curPtr), curLen};
      bool ok = __moore_assoc_next(reinterpret_cast<void *>(assocAddr), &keyRef);
      if (!ok) {
        setCallResultInt(callOp.getResult(0), 0);
        recordFastPathHit("func.call.component.get_next_child_end");
        return true;
      }
      if (keyRef.data && keyRef.len > 0)
        dynamicStrings[reinterpret_cast<uint64_t>(keyRef.data)] = {keyRef.data,
                                                                   keyRef.len};
      if (!writeStringStructToRef(callOp.getOperand(1),
                                  reinterpret_cast<uint64_t>(keyRef.data),
                                  keyRef.len))
        return false;

      setCallResultInt(callOp.getResult(0), ok ? 1 : 0);
      recordFastPathHit("func.call.component.get_next_child");
      return true;
    }
  }

  // First-tier registry dispatch keyed by (call form, exact symbol).
  switch (lookupUvmFastPath(UvmFastPathCallForm::FuncCall, calleeName)) {
  case UvmFastPathAction::WaitForSelfAndSiblingsToDrop: {
    if (callOp.getNumOperands() < 1)
      break;
    InterpretedValue phaseVal = getValue(procId, callOp.getOperand(0));
    uint64_t phaseAddr = phaseVal.isX() ? 0 : phaseVal.getUInt64();
    return handleUvmWaitForSelfAndSiblingsToDrop(procId, phaseAddr,
                                                 callOp.getOperation());
  }
  case UvmFastPathAction::ReportInfoSuppress:
    if (!fastPathUvmReportInfo)
      break;
    for (Value result : callOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry report_info suppress: " << calleeName
               << "\n");
    recordFastPathHit("registry.func.call.report_info_suppress");
    return true;
  case UvmFastPathAction::ReportWarningSuppress:
    if (!fastPathUvmReportWarning)
      break;
    for (Value result : callOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry report_warning suppress: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.report_warning_suppress");
    return true;
  case UvmFastPathAction::PrinterNoOp:
    // Suppress printer body and zero all results.
    for (Value result : callOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: printer no-op: " << calleeName << "\n");
    recordFastPathHit("registry.func.call.printer_noop");
    return true;
  case UvmFastPathAction::AdjustNamePassthrough:
  case UvmFastPathAction::None:
    break;
  }

  if (calleeName.contains("wait_for_self_and_siblings_to_drop") &&
      callOp.getNumOperands() >= 1) {
    InterpretedValue phaseVal = getValue(procId, callOp.getOperand(0));
    uint64_t phaseAddr = phaseVal.isX() ? 0 : phaseVal.getUInt64();
    return handleUvmWaitForSelfAndSiblingsToDrop(procId, phaseAddr,
                                                 callOp.getOperation());
  }

  auto readRootSingletonForFuncCall = [&](uint64_t &rootAddr) -> bool {
    rootAddr = 0;
    if (!rootModule)
      return false;
    auto *addressofOp =
        rootModule.lookupSymbol("uvm_pkg::uvm_pkg::uvm_root::m_inst");
    if (!addressofOp)
      return false;
    auto globalIt =
        globalMemoryBlocks.find("uvm_pkg::uvm_pkg::uvm_root::m_inst");
    if (globalIt == globalMemoryBlocks.end() || !globalIt->second.initialized ||
        globalIt->second.size < 8)
      return false;
    for (unsigned i = 0; i < 8; ++i)
      rootAddr |= static_cast<uint64_t>(globalIt->second[i]) << (i * 8);
    return rootAddr != 0;
  };

  bool isRootWrapperGetter = (calleeName == "m_uvm_get_root" ||
                              calleeName.ends_with("::m_uvm_get_root"));
  if (isRootWrapperGetter && callOp.getNumResults() >= 1 &&
      callOp.getNumOperands() == 0) {
    uint64_t rootAddr = 0;
    if (readRootSingletonForFuncCall(rootAddr)) {
      Value result = callOp.getResult(0);
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt(width, rootAddr)));
      for (unsigned i = 1, e = callOp.getNumResults(); i < e; ++i) {
        Value extra = callOp.getResult(i);
        unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
        setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
      }
      recordFastPathHit("func.call.root_wrapper_getter");
      return true;
    }
  }

  // Pattern-based printer interception: catches subclass-qualified names
  // like uvm_pkg::custom_printer::uvm_printer::print_field_int_foo that
  // the exact StringSwitch above misses.
  if (calleeName.contains("uvm_printer::print_field_int") ||
      calleeName.contains("uvm_printer::print_field") ||
      calleeName.contains("uvm_printer::print_generic_element") ||
      calleeName.contains("uvm_printer::print_generic") ||
      calleeName.contains("uvm_printer::print_time") ||
      calleeName.contains("uvm_printer::print_string") ||
      calleeName.contains("uvm_printer::print_real") ||
      calleeName.contains("uvm_printer::print_array_header") ||
      calleeName.contains("uvm_printer::print_array_footer") ||
      calleeName.contains("uvm_printer::print_array_range") ||
      calleeName.contains("uvm_printer::print_object_header") ||
      calleeName.contains("uvm_printer::print_object")) {
    for (Value result : callOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: printer no-op (pattern): " << calleeName
               << "\n");
    recordFastPathHit("func.call.printer_noop_pattern");
    return true;
  }

  // Optional fast-path for low-severity report traffic. This bypasses
  // high-volume info/warning report formatting and dispatch logic while
  // preserving default behavior unless explicitly enabled.
  if ((fastPathUvmReportInfo &&
       calleeName.contains("uvm_report_object::uvm_report_info")) ||
      (fastPathUvmReportWarning &&
       calleeName.contains("uvm_report_object::uvm_report_warning"))) {
    for (Value result : callOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: report traffic fast-path (suppressed): "
               << calleeName << "\n");
    return true;
  }

  return false;
}
