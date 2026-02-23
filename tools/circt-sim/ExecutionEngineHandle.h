//===- ExecutionEngineHandle.h - Optional ExecutionEngine ownership -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared ownership handle for JIT execution engines used by circt-sim.
//
// In non-JIT builds we intentionally type-erase the handle to avoid requiring
// full mlir::ExecutionEngine type visibility in headers.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_EXECUTIONENGINEHANDLE_H
#define CIRCT_TOOLS_CIRCT_SIM_EXECUTIONENGINEHANDLE_H

#include <memory>

namespace mlir {
#ifdef CIRCT_SIM_JIT_ENABLED
class ExecutionEngine;
#endif
} // namespace mlir

namespace circt {
namespace sim {

#ifndef CIRCT_SIM_JIT_ENABLED
struct ExecutionEngineNoopDelete {
  void operator()(void *) const noexcept {}
};
using ExecutionEngineHandle = std::unique_ptr<void, ExecutionEngineNoopDelete>;
#else
using ExecutionEngineHandle = std::unique_ptr<mlir::ExecutionEngine>;
#endif

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_EXECUTIONENGINEHANDLE_H
