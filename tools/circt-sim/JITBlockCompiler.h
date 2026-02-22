//===- JITBlockCompiler.h - Block-level JIT for hot process blocks -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Identifies "hot" basic blocks in LLHD process bodies, extracts them into
// standalone MLIR functions, lowers them to LLVM IR, and JIT-compiles them
// via LLVM ORC. The resulting native function pointers can be called by the
// thunk dispatch system instead of interpreting the block op-by-op.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_JITBLOCKCOMPILER_H
#define CIRCT_TOOLS_CIRCT_SIM_JITBLOCKCOMPILER_H

#include "JITSchedulerRuntime.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <memory>
#include <vector>

namespace mlir {
class ExecutionEngine;
class MLIRContext;
} // namespace mlir

namespace circt {
namespace sim {

/// Describes a hot block extracted from an LLHD process.
struct JITBlockSpec {
  /// The MLIR block to JIT-compile (the "hot" resume block).
  mlir::Block *hotBlock = nullptr;

  /// The wait op that resumes into this block.
  llhd::WaitOp waitOp = nullptr;

  /// Signal IDs probed in this block, in order of appearance.
  llvm::SmallVector<SignalId, 4> signalReads;

  /// Signal IDs driven in this block, in order of appearance.
  llvm::SmallVector<SignalId, 4> signalDrives;

  /// Widths corresponding to each signalReads entry.
  llvm::SmallVector<uint32_t, 4> readWidths;

  /// Widths corresponding to each signalDrives entry.
  llvm::SmallVector<uint32_t, 4> driveWidths;

  /// Pre-resolved delay for drives (packed via encodeJITDelay).
  int64_t driveDelayEncoded = 0;

  /// JIT-compiled native function pointer.
  /// Signature: void (*)(JITRuntimeContext *ctx)
  using NativeFuncTy = void (*)(void *);
  NativeFuncTy nativeFunc = nullptr;

  /// Name of the generated function (for debugging).
  std::string funcName;
};

/// The block-level JIT compiler. Owns the LLVM ORC execution engine and
/// manages compiled block functions.
class JITBlockCompiler {
public:
  explicit JITBlockCompiler(mlir::MLIRContext &ctx);
  ~JITBlockCompiler();

  /// Analyze an LLHD process and identify hot blocks suitable for JIT.
  /// Returns true if a JIT-eligible hot block was found.
  ///
  /// @param processOp      The LLHD process operation.
  /// @param signalIdMap     Maps MLIR signal Value → SignalId.
  /// @param scheduler       The process scheduler (for signal widths).
  /// @param[out] spec       Populated with the hot block specification.
  bool identifyHotBlock(
      llhd::ProcessOp processOp,
      const llvm::DenseMap<mlir::Value, SignalId> &signalIdMap,
      ProcessScheduler &scheduler, JITBlockSpec &spec);

  /// Extract the hot block into a standalone MLIR function, lower to LLVM IR,
  /// and JIT-compile it. Populates spec.nativeFunc on success.
  ///
  /// @param spec            The block spec from identifyHotBlock.
  /// @param sourceModule    The parent module (for type info and context).
  /// @return true on success.
  bool compileBlock(JITBlockSpec &spec, mlir::ModuleOp sourceModule);

  /// Get compilation statistics.
  struct Stats {
    unsigned blocksAnalyzed = 0;
    unsigned blocksEligible = 0;
    unsigned blocksCompiled = 0;
    unsigned blocksFailed = 0;
    double totalCompileTimeMs = 0.0;
  };
  const Stats &getStats() const { return stats; }

private:
  /// Check if all ops in a block are JIT-compatible.
  bool isBlockJITCompatible(mlir::Block *block);

  /// Create a standalone MLIR function from the hot block.
  /// Returns true on success.
  bool extractBlockFunction(JITBlockSpec &spec, mlir::ModuleOp microModule);

  /// Lower the micro-module to LLVM IR and JIT-compile.
  bool lowerAndCompile(JITBlockSpec &spec, mlir::ModuleOp microModule);

  mlir::MLIRContext &mlirContext;
  Stats stats;

  /// Engines that own compiled code memory.
  ///
  /// Keep every engine alive for the lifetime of the compiler instance so
  /// previously returned native function pointers remain valid.
  std::vector<std::unique_ptr<mlir::ExecutionEngine>> engines;

  /// Counter for generating unique function names.
  unsigned funcCounter = 0;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_JITBLOCKCOMPILER_H
