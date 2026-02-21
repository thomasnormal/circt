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
  GetReportObject,
  GetReportVerbosityLevel,
  GetReportAction,
  ReportHandlerGetVerbosityLevel,
  ReportHandlerGetAction,
  ReportHandlerSetSeverityActionNoOp,
  ReportHandlerSetSeverityFileNoOp,
};

static int32_t defaultUvmActionForSeverity(uint64_t sev) {
  switch (sev) {
  case 0:
    return 1; // UVM_INFO -> UVM_DISPLAY
  case 1:
    return 1; // UVM_WARNING -> UVM_DISPLAY
  case 2:
    return 9; // UVM_ERROR -> UVM_DISPLAY | UVM_COUNT
  case 3:
    return 33; // UVM_FATAL -> UVM_DISPLAY | UVM_EXIT
  default:
    return 1;
  }
}

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
        .Case("uvm_pkg::uvm_report_object::uvm_get_report_object",
              UvmFastPathAction::GetReportObject)
        .Case("uvm_pkg::uvm_report_object::get_report_verbosity_level",
              UvmFastPathAction::GetReportVerbosityLevel)
        .Case("uvm_pkg::uvm_report_object::get_report_action",
              UvmFastPathAction::GetReportAction)
        .Case("uvm_pkg::uvm_report_handler::get_verbosity_level",
              UvmFastPathAction::ReportHandlerGetVerbosityLevel)
        .Case("uvm_pkg::uvm_report_handler::get_action",
              UvmFastPathAction::ReportHandlerGetAction)
        .Case("uvm_pkg::uvm_report_handler::set_severity_action",
              UvmFastPathAction::ReportHandlerSetSeverityActionNoOp)
        .Case("uvm_pkg::uvm_report_handler::set_severity_file",
              UvmFastPathAction::ReportHandlerSetSeverityFileNoOp)
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
        .Case("uvm_pkg::uvm_report_object::uvm_get_report_object",
              UvmFastPathAction::GetReportObject)
        .Case("uvm_pkg::uvm_report_object::get_report_verbosity_level",
              UvmFastPathAction::GetReportVerbosityLevel)
        .Case("uvm_pkg::uvm_report_object::get_report_action",
              UvmFastPathAction::GetReportAction)
        .Case("uvm_pkg::uvm_report_handler::get_verbosity_level",
              UvmFastPathAction::ReportHandlerGetVerbosityLevel)
        .Case("uvm_pkg::uvm_report_handler::get_action",
              UvmFastPathAction::ReportHandlerGetAction)
        .Case("uvm_pkg::uvm_report_handler::set_severity_action",
              UvmFastPathAction::ReportHandlerSetSeverityActionNoOp)
        .Case("uvm_pkg::uvm_report_handler::set_severity_file",
              UvmFastPathAction::ReportHandlerSetSeverityFileNoOp)
        .Default(UvmFastPathAction::None);
  }
  return UvmFastPathAction::None;
}
} // namespace

bool LLHDProcessInterpreter::handleUvmWaitForSelfAndSiblingsToDrop(
    ProcessId procId, uint64_t phaseAddr, mlir::Operation *callOp) {
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

  static std::map<ProcessId, int> yieldCountByProc;
  if (count <= 0) {
    // Yield repeatedly to give forked task processes a chance to raise.
    int &yields = yieldCountByProc[procId];
    if (yields >= 10) {
      yields = 0;
      return true;
    }
    ++yields;
  } else {
    yieldCountByProc[procId] = 0;
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
    if (block && offset + 8 <= block->data.size()) {
      for (unsigned i = 0; i < 8; ++i)
        block->data[offset + i] =
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
    if (!objBlk || !objBlk->initialized || objOff + 8 > objBlk->data.size())
      return false;

    uint64_t vtableAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      vtableAddr |= static_cast<uint64_t>(objBlk->data[objOff + i]) << (i * 8);
    if (vtableAddr == 0)
      return false;

    uint64_t entryOff = 0;
    MemoryBlock *entryBlk =
        findMemoryBlockByAddress(vtableAddr + slot * 8, procId, &entryOff);
    if (!entryBlk)
      entryBlk = findBlockByAddress(vtableAddr + slot * 8, entryOff);
    if (!entryBlk || !entryBlk->initialized ||
        entryOff + 8 > entryBlk->data.size())
      return false;

    uint64_t funcAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      funcAddr |= static_cast<uint64_t>(entryBlk->data[entryOff + i]) << (i * 8);
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
        typeNameResults.empty() || typeNameResults.front().isX() ||
        typeNameResults.front().getWidth() < 128)
      return false;

    llvm::APInt packed = typeNameResults.front().getAPInt();
    uint64_t strPtr = packed.extractBits(64, 0).getZExtValue();
    int64_t strLen = packed.extractBits(64, 64).getSExtValue();
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
    if (typeNameArg.isX() || typeNameArg.getWidth() < 128)
      return false;

    llvm::APInt packed = typeNameArg.getAPInt();
    uint64_t strPtr = packed.extractBits(64, 0).getZExtValue();
    int64_t strLen = packed.extractBits(64, 64).getSExtValue();
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

  // m_uvm_get_root / uvm_get_report_object: return cached uvm_root::m_inst.
  if ((matchesMethod("m_uvm_get_root") ||
       matchesMethod("uvm_get_report_object")) &&
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
          globalIt->second.data.size() >= 8) {
        for (unsigned i = 0; i < 8; ++i)
          rootAddr |= static_cast<uint64_t>(globalIt->second.data[i]) << (i * 8);
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
    unsigned width = std::max(1u, getTypeWidth(funcOp.getResultTypes()[0]));
    results.push_back(InterpretedValue(llvm::APInt(width, objHandle)));
    noteUvmFastPathActionHit("func.body.phase_hopper.get_objection");
    return true;
  }

  // Return the total objection count for a handle.
  if (matchesMethod("get_objection_total") && args.size() >= 1 &&
      funcOp.getNumResults() >= 1) {
    uint64_t objHandle = args[0].isX() ? 0 : args[0].getUInt64();
    int64_t count = 0;
    if (objHandle != 0)
      count = __moore_objection_get_count(
          static_cast<MooreObjectionHandle>(objHandle));
    unsigned width = std::max(1u, getTypeWidth(funcOp.getResultTypes()[0]));
    results.push_back(InterpretedValue(
        llvm::APInt(width, static_cast<uint64_t>(std::max<int64_t>(0, count)))));
    noteUvmFastPathActionHit("func.body.objection.get_objection_total");
    return true;
  }

  return false;
}

bool LLHDProcessInterpreter::handleUvmCallIndirectFastPath(
    ProcessId procId, mlir::func::CallIndirectOp callIndirectOp,
    llvm::StringRef calleeName) {
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
    if (block && block->initialized && offset + 8 <= block->data.size()) {
      out = 0;
      for (unsigned i = 0; i < 8; ++i)
        out |= static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
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
    if (block && offset + 16 <= block->data.size()) {
      std::memcpy(block->data.data() + offset, &strPtr, 8);
      std::memcpy(block->data.data() + offset + 8, &strLen, 8);
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
    if (strVal.isX() || strVal.getWidth() < 128)
      return false;
    llvm::APInt bits = strVal.getAPInt();
    strPtr = bits.extractBits(64, 0).getZExtValue();
    strLen = bits.extractBits(64, 64).getSExtValue();
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

  if (calleeName.contains("uvm_component::get_num_children") &&
      callIndirectOp.getArgOperands().size() >= 1 &&
      callIndirectOp.getNumResults() >= 1) {
    InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
    if (selfVal.isX())
      return false;
    uint64_t assocAddr = 0;
    if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
      return false;

    uint64_t count = 0;
    if (assocAddr != 0)
      count = static_cast<uint64_t>(
          std::max<int64_t>(0, __moore_assoc_size(reinterpret_cast<void *>(assocAddr))));
    setCallIndirectResultInt(callIndirectOp.getResult(0), count);
    recordFastPathHit("call_indirect.component.get_num_children");
    return true;
  }

  if (calleeName.contains("uvm_component::has_child") &&
      callIndirectOp.getArgOperands().size() >= 2 &&
      callIndirectOp.getNumResults() >= 1) {
    InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
    if (selfVal.isX())
      return false;
    uint64_t assocAddr = 0;
    if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
      return false;
    uint64_t keyPtr = 0;
    int64_t keyLen = 0;
    if (!readPackedStringParts(getValue(procId, callIndirectOp.getArgOperands()[1]),
                               keyPtr, keyLen))
      return false;

    std::string keyStorage;
    if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
      return false;

    bool exists = false;
    if (assocAddr != 0) {
      MooreString key{const_cast<char *>(keyStorage.data()),
                      static_cast<int64_t>(keyStorage.size())};
      exists = __moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key) != 0;
    }
    setCallIndirectResultInt(callIndirectOp.getResult(0), exists ? 1 : 0);
    recordFastPathHit("call_indirect.component.has_child");
    return true;
  }

  if (calleeName.contains("uvm_component::get_child") &&
      callIndirectOp.getArgOperands().size() >= 2 &&
      callIndirectOp.getNumResults() >= 1) {
    InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
    if (selfVal.isX())
      return false;
    uint64_t assocAddr = 0;
    if (!getComponentChildrenAssocAddr(selfVal.getUInt64(), assocAddr))
      return false;
    uint64_t keyPtr = 0;
    int64_t keyLen = 0;
    if (!readPackedStringParts(getValue(procId, callIndirectOp.getArgOperands()[1]),
                               keyPtr, keyLen))
      return false;

    std::string keyStorage;
    if (!tryReadStringKey(procId, keyPtr, keyLen, keyStorage))
      return false;

    uint64_t childAddr = 0;
    if (assocAddr != 0) {
      MooreString key{const_cast<char *>(keyStorage.data()),
                      static_cast<int64_t>(keyStorage.size())};
      if (__moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key)) {
        void *ref = __moore_assoc_get_ref(reinterpret_cast<void *>(assocAddr), &key,
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
    InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
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
      dynamicStrings[reinterpret_cast<uint64_t>(keyOut.data)] = {keyOut.data, keyOut.len};
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
    InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
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
    if (!readRefStringParts(callIndirectOp.getArgOperands()[1], curPtr, curLen))
      return false;

    MooreString keyRef{reinterpret_cast<char *>(curPtr), curLen};
    bool ok = __moore_assoc_next(reinterpret_cast<void *>(assocAddr), &keyRef);
    if (!ok) {
      setCallIndirectResultInt(callIndirectOp.getResult(0), 0);
      recordFastPathHit("call_indirect.component.get_next_child_end");
      return true;
    }
    if (keyRef.data && keyRef.len > 0)
      dynamicStrings[reinterpret_cast<uint64_t>(keyRef.data)] = {keyRef.data, keyRef.len};
    if (!writeStringStructToRef(callIndirectOp.getArgOperands()[1],
                                reinterpret_cast<uint64_t>(keyRef.data),
                                keyRef.len))
      return false;

    setCallIndirectResultInt(callIndirectOp.getResult(0), 1);
    recordFastPathHit("call_indirect.component.get_next_child");
    return true;
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
  case UvmFastPathAction::GetReportObject: {
    if (!fastPathUvmGetReportObject || callIndirectOp.getNumResults() < 1 ||
        callIndirectOp.getArgOperands().empty())
      break;
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    InterpretedValue selfVal =
        getValue(procId, callIndirectOp.getArgOperands()[0]);
    if (selfVal.isX()) {
      setValue(procId, result, InterpretedValue::makeX(width));
    } else if (selfVal.getWidth() != width) {
      setValue(procId, result,
               InterpretedValue(selfVal.getAPInt().zextOrTrunc(width)));
    } else {
      setValue(procId, result, selfVal);
    }
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra,
               InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry get_report_object return self: "
               << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.get_report_object");
    return true;
  }
  case UvmFastPathAction::GetReportVerbosityLevel: {
    if (callIndirectOp.getNumResults() < 1)
      break;
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, 200)));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra,
               InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry get_report_verbosity_level -> "
               << "200: " << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.get_report_verbosity");
    return true;
  }
  case UvmFastPathAction::GetReportAction: {
    if (callIndirectOp.getNumResults() < 1 ||
        callIndirectOp.getArgOperands().size() < 2)
      break;
    InterpretedValue sevVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(
                                 width, defaultUvmActionForSeverity(sev))));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra,
               InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry get_report_action (sev=" << sev
               << "): " << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.get_report_action");
    return true;
  }
  case UvmFastPathAction::ReportHandlerGetVerbosityLevel: {
    if (callIndirectOp.getNumResults() < 1)
      break;
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, 200)));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra,
               InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_handler::get_verbosity "
               << "-> 200: " << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.report_handler_get_verbosity");
    return true;
  }
  case UvmFastPathAction::ReportHandlerGetAction: {
    if (callIndirectOp.getNumResults() < 1 ||
        callIndirectOp.getArgOperands().size() < 2)
      break;
    InterpretedValue sevVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(
                                 width, defaultUvmActionForSeverity(sev))));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra,
               InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_handler::get_action "
               << "(sev=" << sev << "): " << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.report_handler_get_action");
    return true;
  }
  case UvmFastPathAction::ReportHandlerSetSeverityActionNoOp:
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_handler::set_severity_action "
                  "no-op: "
               << calleeName << "\n");
    recordFastPathHit(
        "registry.call_indirect.report_handler_set_severity_action");
    return true;
  case UvmFastPathAction::ReportHandlerSetSeverityFileNoOp:
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: registry report_handler::set_severity_file "
                  "no-op: "
               << calleeName << "\n");
    recordFastPathHit("registry.call_indirect.report_handler_set_severity_file");
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

  // Fast-path report-object wrappers that otherwise call m_rh_init() and
  // dispatch into report-handler getters on every report.
  if (fastPathUvmGetReportObject &&
      calleeName.contains("uvm_report_object::uvm_get_report_object") &&
      callIndirectOp.getNumResults() >= 1 &&
      callIndirectOp.getArgOperands().size() >= 1) {
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    InterpretedValue selfVal =
        getValue(procId, callIndirectOp.getArgOperands()[0]);
    if (selfVal.isX()) {
      setValue(procId, result, InterpretedValue::makeX(width));
    } else if (selfVal.getWidth() != width) {
      setValue(procId, result,
               InterpretedValue(selfVal.getAPInt().zextOrTrunc(width)));
    } else {
      setValue(procId, result, selfVal);
    }
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: uvm_get_report_object fast-path (return "
                  "self): "
               << calleeName << "\n");
    return true;
  }

  if (calleeName.contains("uvm_report_object::get_report_verbosity_level") &&
      callIndirectOp.getNumResults() >= 1) {
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, 200)));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: get_report_verbosity_level fast-path -> "
               << "200: " << calleeName << "\n");
    return true;
  }

  if (calleeName.contains("uvm_report_object::get_report_action") &&
      callIndirectOp.getNumResults() >= 1 &&
      callIndirectOp.getArgOperands().size() >= 2) {
    InterpretedValue sevVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result,
             InterpretedValue(llvm::APInt(width, defaultUvmActionForSeverity(sev))));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: get_report_action fast-path (sev=" << sev
               << "): " << calleeName << "\n");
    return true;
  }

  // Mirror direct-call report-handler getter fast-paths in vtable dispatch.
  if (calleeName.contains("uvm_report_handler") &&
      calleeName.contains("::get_verbosity_level") &&
      callIndirectOp.getNumResults() >= 1) {
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result, InterpretedValue(llvm::APInt(width, 200)));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: report_handler::get_verbosity_level "
               << "fast-path -> 200: " << calleeName << "\n");
    return true;
  }

  if (calleeName.contains("uvm_report_handler") &&
      calleeName.contains("::get_action") &&
      callIndirectOp.getNumResults() >= 1 &&
      callIndirectOp.getArgOperands().size() >= 2) {
    InterpretedValue sevVal =
        getValue(procId, callIndirectOp.getArgOperands()[1]);
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    Value result = callIndirectOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    setValue(procId, result,
             InterpretedValue(llvm::APInt(width, defaultUvmActionForSeverity(sev))));
    for (unsigned i = 1, e = callIndirectOp.getNumResults(); i < e; ++i) {
      Value extra = callIndirectOp.getResult(i);
      unsigned extraWidth = std::max(1u, getTypeWidth(extra.getType()));
      setValue(procId, extra, InterpretedValue(llvm::APInt::getZero(extraWidth)));
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  call_indirect: report_handler::get_action fast-path "
               << "(sev=" << sev << "): " << calleeName << "\n");
    return true;
  }

  return false;
}

bool LLHDProcessInterpreter::handleUvmFuncCallFastPath(
    ProcessId procId, mlir::func::CallOp callOp, llvm::StringRef calleeName) {
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
    if (block && block->initialized && offset + 8 <= block->data.size()) {
      out = 0;
      for (unsigned i = 0; i < 8; ++i)
        out |= static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
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
    if (block && offset + 16 <= block->data.size()) {
      std::memcpy(block->data.data() + offset, &strPtr, 8);
      std::memcpy(block->data.data() + offset + 8, &strLen, 8);
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
    if (strVal.isX() || strVal.getWidth() < 128)
      return false;
    llvm::APInt bits = strVal.getAPInt();
    strPtr = bits.extractBits(64, 0).getZExtValue();
    strLen = bits.extractBits(64, 64).getSExtValue();
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
          std::max<int64_t>(0, __moore_assoc_size(reinterpret_cast<void *>(assocAddr))));
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
      exists = __moore_assoc_exists(reinterpret_cast<void *>(assocAddr), &key) != 0;
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
        void *ref = __moore_assoc_get_ref(reinterpret_cast<void *>(assocAddr), &key,
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
      dynamicStrings[reinterpret_cast<uint64_t>(keyOut.data)] = {keyOut.data, keyOut.len};
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
      dynamicStrings[reinterpret_cast<uint64_t>(keyRef.data)] = {keyRef.data, keyRef.len};
    if (!writeStringStructToRef(callOp.getOperand(1),
                                reinterpret_cast<uint64_t>(keyRef.data),
                                keyRef.len))
      return false;

    setCallResultInt(callOp.getResult(0), ok ? 1 : 0);
    recordFastPathHit("func.call.component.get_next_child");
    return true;
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
  case UvmFastPathAction::GetReportObject: {
    if (!fastPathUvmGetReportObject || callOp.getNumResults() != 1 ||
        callOp.getNumOperands() < 1)
      break;
    Value result = callOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
    if (selfVal.isX()) {
      setValue(procId, result, InterpretedValue::makeX(width));
    } else if (selfVal.getWidth() != width) {
      setValue(procId, result,
               InterpretedValue(selfVal.getAPInt().zextOrTrunc(width)));
    } else {
      setValue(procId, result, selfVal);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry get_report_object return self: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.get_report_object");
    return true;
  }
  case UvmFastPathAction::GetReportVerbosityLevel:
    if (callOp.getNumResults() != 1)
      break;
    setValue(procId, callOp.getResult(0), InterpretedValue(llvm::APInt(32, 200)));
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry get_report_verbosity_level -> 200: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.get_report_verbosity");
    return true;
  case UvmFastPathAction::GetReportAction: {
    if (callOp.getNumResults() != 1 || callOp.getNumOperands() < 2)
      break;
    InterpretedValue sevVal = getValue(procId, callOp.getOperand(1));
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    int32_t action = defaultUvmActionForSeverity(sev);
    setValue(procId, callOp.getResult(0),
             InterpretedValue(llvm::APInt(32, static_cast<uint64_t>(action))));
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry get_report_action (sev=" << sev
               << ") -> " << action << ": " << calleeName << "\n");
    recordFastPathHit("registry.func.call.get_report_action");
    return true;
  }
  case UvmFastPathAction::ReportHandlerGetVerbosityLevel:
  case UvmFastPathAction::ReportHandlerGetAction:
    // Registry handles common direct-call report-handler symbols only.
    // Fall through to existing generic report-handler logic below, which
    // preserves all current memory-backed behavior.
    break;
  case UvmFastPathAction::ReportHandlerSetSeverityActionNoOp:
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry report_handler::set_severity_action "
                  "no-op: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.report_handler_set_severity_action");
    return true;
  case UvmFastPathAction::ReportHandlerSetSeverityFileNoOp:
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry report_handler::set_severity_file "
                  "no-op: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.report_handler_set_severity_file");
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

  // Intercept report-object wrappers that repeatedly call m_rh_init() and
  // then delegate to report-handler getters.
  if (fastPathUvmGetReportObject &&
      calleeName.contains("uvm_report_object::uvm_get_report_object") &&
      callOp.getNumResults() == 1 && callOp.getNumOperands() >= 1) {
    Value result = callOp.getResult(0);
    unsigned width = std::max(1u, getTypeWidth(result.getType()));
    InterpretedValue selfVal = getValue(procId, callOp.getOperand(0));
    if (selfVal.isX()) {
      setValue(procId, result, InterpretedValue::makeX(width));
    } else if (selfVal.getWidth() != width) {
      setValue(procId, result,
               InterpretedValue(selfVal.getAPInt().zextOrTrunc(width)));
    } else {
      setValue(procId, result, selfVal);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: " << calleeName
               << " intercepted (return self)\n");
    return true;
  }

  if (calleeName.contains("uvm_report_object::get_report_verbosity_level") &&
      callOp.getNumResults() == 1) {
    setValue(procId, callOp.getResult(0), InterpretedValue(llvm::APInt(32, 200)));
    LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                            << " intercepted -> 200\n");
    return true;
  }
  if (calleeName.contains("uvm_report_object::get_report_action") &&
      callOp.getNumResults() == 1 && callOp.getNumOperands() >= 2) {
    InterpretedValue sevVal = getValue(procId, callOp.getOperand(1));
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    int32_t action = defaultUvmActionForSeverity(sev);
    setValue(procId, callOp.getResult(0),
             InterpretedValue(llvm::APInt(32, static_cast<uint64_t>(action))));
    LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                            << " intercepted (sev=" << sev << ") -> " << action
                            << "\n");
    return true;
  }

  // Intercept report_handler severity-map mutators. These methods only affect
  // report formatting/routing policy and are safely bypassed by our default
  // report getter fast-path behavior.
  if (calleeName.contains("uvm_report_handler") &&
      (calleeName.contains("::set_severity_file") ||
       calleeName.contains("::set_severity_action"))) {
    LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                            << " intercepted (no-op)\n");
    return true;
  }

  // Intercept uvm_report_handler::get_verbosity_level to avoid failures when
  // associative array fields are not fully initialized. In the default case
  // (no per-id/per-severity overrides), this returns UVM_MEDIUM (200).
  if (calleeName.contains("uvm_report_handler") &&
      calleeName.contains("::get_verbosity_level") &&
      callOp.getNumResults() == 1) {
    int32_t verbosity = 200; // UVM_MEDIUM default
    InterpretedValue handlerVal = getValue(procId, callOp.getOperand(0));
    if (!handlerVal.isX()) {
      uint64_t handlerAddr = handlerVal.getUInt64();
      uint64_t off = 0;
      MemoryBlock *blk = findMemoryBlockByAddress(handlerAddr, procId, &off);
      if (blk && blk->initialized) {
        // uvm_object base size: uvm_void(i32=4, ptr=8) +
        // string(ptr=8, i64=8) + i32=4 = 32.
        // field 1 (m_max_verbosity_level) is at offset 32 from handler base.
        size_t fieldOff = off + 32;
        if (fieldOff + 4 <= blk->data.size()) {
          verbosity = 0;
          for (int i = 0; i < 4; ++i)
            verbosity |= static_cast<int32_t>(blk->data[fieldOff + i])
                         << (i * 8);
        }
      }
    }
    setValue(procId, callOp.getResult(0),
             InterpretedValue(llvm::APInt(32, static_cast<uint64_t>(verbosity))));
    LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                            << " intercepted -> " << verbosity << "\n");
    return true;
  }

  // Intercept uvm_report_handler::get_action to return default severity actions.
  if (calleeName.contains("uvm_report_handler") &&
      calleeName.contains("::get_action") &&
      callOp.getNumResults() == 1 && callOp.getNumOperands() >= 2) {
    InterpretedValue sevVal = getValue(procId, callOp.getOperand(1));
    uint64_t sev = sevVal.isX() ? 0 : sevVal.getUInt64();
    int32_t action = defaultUvmActionForSeverity(sev);
    setValue(procId, callOp.getResult(0),
             InterpretedValue(llvm::APInt(32, static_cast<uint64_t>(action))));
    LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                            << " intercepted (sev=" << sev << ") -> " << action
                            << "\n");
    return true;
  }

  return false;
}
