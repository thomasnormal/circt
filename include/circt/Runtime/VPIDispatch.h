//===- VPIDispatch.h - VPI Function Dispatch Table ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin dispatch table for VPI functions. When VPIRuntime is active (cocotb
// mode), it registers function pointers here. MooreRuntime's VPI stubs check
// this table and delegate when set.
//
// This header has NO dependencies on MLIR/LLVM, so it can be included by
// the standalone MooreRuntime library.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_RUNTIME_VPIDISPATCH_H
#define CIRCT_RUNTIME_VPIDISPATCH_H

#include <cstdint>

/// VPI function dispatch table.
/// When VPIRuntime is active, it fills in these function pointers.
/// MooreRuntime's VPI stubs check `isActive` and delegate through these.
struct VPIDispatchTable {
  bool isActive = false;

  // Handle management
  void *(*handleByName)(const char *name, void *scope) = nullptr;
  int32_t (*getProperty)(int32_t property, void *obj) = nullptr;
  const char *(*getStrProperty)(int32_t property, void *obj) = nullptr;
  void (*getValue)(void *obj, void *value_p) = nullptr;
  int32_t (*putValue)(void *obj, void *value_p, void *time_p,
                      int32_t flags) = nullptr;
  int32_t (*freeObject)(void *obj) = nullptr;
  void (*getTime)(void *obj, void *time_p) = nullptr;
  int32_t (*getVlogInfo)(void *vlog_info_p) = nullptr;
  void *(*handle)(int32_t type, void *refHandle) = nullptr;
  void *(*handleByIndex)(void *obj, int32_t indx) = nullptr;
  void *(*iterate)(int32_t type, void *refHandle) = nullptr;
  void *(*scan)(void *iterator) = nullptr;
  void *(*registerCb)(void *cb_data_p) = nullptr;
  int32_t (*removeCb)(void *cb_obj) = nullptr;
  int32_t (*chkError)(void *error_info_p) = nullptr;
  int32_t (*control)(int32_t operation) = nullptr;
  void (*releaseHandle)(void *obj) = nullptr;
};

/// Global VPI dispatch table. Defined in MooreRuntime.cpp so that linking
/// MooreRuntime alone doesn't pull in VPIRuntime dependencies.
extern VPIDispatchTable gVPIDispatch;

#endif // CIRCT_RUNTIME_VPIDISPATCH_H
