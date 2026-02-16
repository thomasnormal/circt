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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

namespace {
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
} // namespace

bool LLHDProcessInterpreter::handleUvmCallIndirectFastPath(
    ProcessId procId, mlir::func::CallIndirectOp callIndirectOp,
    llvm::StringRef calleeName) {
  auto zeroCallIndirectResults = [&]() {
    for (Value result : callIndirectOp.getResults()) {
      unsigned width = std::max(1u, getTypeWidth(result.getType()));
      setValue(procId, result, InterpretedValue(llvm::APInt::getZero(width)));
    }
  };

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

  // Fast-path report-object wrappers that otherwise call m_rh_init() and
  // dispatch into report-handler getters on every report.
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
  // Intercept report-object wrappers that repeatedly call m_rh_init() and
  // then delegate to report-handler getters.
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
