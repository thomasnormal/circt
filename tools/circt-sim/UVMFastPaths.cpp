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
        .Case("uvm_pkg::uvm_report_handler::set_severity_file",
              UvmFastPathAction::ReportHandlerSetSeverityFileNoOp)
        .Default(UvmFastPathAction::None);
  case UvmFastPathCallForm::FuncCall:
    return StringSwitch<UvmFastPathAction>(calleeName)
        .Case("uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop",
              UvmFastPathAction::WaitForSelfAndSiblingsToDrop)
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
  constexpr uint32_t kMaxDeltaPolls = 1000;
  constexpr int64_t kFallbackPollDelayFs = 10000000; // 10 ps
  SimTime targetTime = currentTime.deltaStep < kMaxDeltaPolls
                           ? currentTime.nextDelta()
                           : currentTime.advanceTime(kFallbackPollDelayFs);

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
  case UvmFastPathAction::ReportHandlerSetSeverityFileNoOp:
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: registry report_handler::set_severity_file "
                  "no-op: "
               << calleeName << "\n");
    recordFastPathHit("registry.func.call.report_handler_set_severity_file");
    return true;
  case UvmFastPathAction::AdjustNamePassthrough:
  case UvmFastPathAction::PrinterNoOp:
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

  // Intercept uvm_report_handler::set_severity_file which fails due to
  // uninitialized associative array field[11] in the interpreter's memory
  // model. This function maps severity levels to file handles only.
  if (calleeName.contains("uvm_report_handler") &&
      calleeName.contains("::set_severity_file")) {
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
