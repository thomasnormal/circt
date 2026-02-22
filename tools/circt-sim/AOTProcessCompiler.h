//===- AOTProcessCompiler.h - AOT batch process compilation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ahead-of-time batch compilation of multiple LLHD processes into a single
// combined LLVM module with one ExecutionEngine. Amortizes lowering pass
// overhead across all processes.
//
// Unlike FullProcessJITCompiler (one process per engine), AOTProcessCompiler:
// - Extracts all eligible processes into ONE micro-module
// - Runs lowering pipeline ONCE on the combined module
// - Creates ONE ExecutionEngine for all processes
// - Returns compiled function pointers for each process
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_AOTPROCESSCOMPILER_H
#define CIRCT_TOOLS_CIRCT_SIM_AOTPROCESSCOMPILER_H

#include "UcontextProcess.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
class ModuleOp;
class ExecutionEngine;
} // namespace mlir

namespace circt {
namespace sim {

/// Execution model for a compiled process. Determines how the process is
/// dispatched at runtime.
///
/// Two fundamental execution models (callback vs coroutine), with five
/// variants matching the classification from classifyProcess():
enum class ExecModel : uint8_t {
  /// Static signal/edge sensitivity list, registered once at elaboration.
  /// The common `always @(posedge clk)` / `always_comb` pattern (~80-95% of
  /// RTL). No resuspend step after callback returns — permanent sensitivity.
  CallbackStaticObserved,

  /// Single wait but observed signals determined at runtime. Still a callback
  /// (no stack), but needs re-registration after each activation.
  CallbackDynamicWait,

  /// Waits only on time (constant delay). Natural candidate for lightweight
  /// timer events. Clock generators: `always #5 clk = ~clk;`
  CallbackTimeOnly,

  /// Runs once, returns, never re-armed. `seq.initial`, `llhd.combinational`
  /// (no explicit wait).
  OneShotCallback,

  /// Dynamic waits, multiple waits, fork, etc. Needs a private stack
  /// (setjmp/longjmp). UVM, fork/join, complex timed control.
  Coroutine,
};

/// Output of classifyProcess(). Describes how to compile and dispatch a
/// process.
struct CallbackPlan {
  /// The classification.
  ExecModel model = ExecModel::Coroutine;

  /// The single wait op (null for OneShotCallback).
  llhd::WaitOp wait = nullptr;

  /// The block entered on activation (successor of the wait).
  mlir::Block *resumeBlock = nullptr;

  /// The block containing the wait op.
  mlir::Block *waitBlock = nullptr;

  /// True if the process entry block differs from resumeBlock, meaning the
  /// process has preamble code before the first wait. The init run executes
  /// entry→wait, populates the CallbackFrame, then returns.
  bool needsInitRun = false;

  /// For CallbackStaticObserved: the static sensitivity list extracted from
  /// the wait's observed signals.
  llvm::SmallVector<std::pair<SignalId, EdgeType>> staticSignals;

  /// For CallbackTimeOnly: the constant delay in simulation time units.
  uint64_t delayValue = 0;

  /// Types of the loop-carried values (wait.destOperands). These define the
  /// CallbackFrame layout — each value is stored/loaded across activations.
  /// Empty if the wait has no destOperands (no loop-carried state).
  llvm::SmallVector<mlir::Type> frameSlotTypes;

  /// True if the process has loop-carried state (non-empty frameSlotTypes).
  bool hasFrame() const { return !frameSlotTypes.empty(); }

  /// True if the process is any callback variant (not a coroutine).
  bool isCallback() const { return model != ExecModel::Coroutine; }
};

/// Result of compiling a single process in an AOT batch.
struct AOTCompiledProcess {
  /// The process ID (from scheduler).
  ProcessId procId = 0;

  /// The compiled entry function (no-arg, signal IDs baked as constants).
  UcontextProcessState::EntryFuncTy entryFunc = nullptr;

  /// Human-readable name for debugging.
  std::string funcName;

  /// The execution model determined by classifyProcess().
  ExecModel execModel = ExecModel::Coroutine;

  /// True if this is a run-to-completion callback (stackless).
  /// False if it requires a coroutine (ucontext) for mid-body yields.
  bool isCallback = false;

  /// For callback processes: the sensitivity list extracted from the
  /// single llhd.wait op at compile time. The scheduler uses this to
  /// set up the process's wait list without executing the init path.
  llvm::SmallVector<std::pair<SignalId, EdgeType>> waitSignals;

  /// Whether the process needs an init run before steady-state activation.
  bool needsInitRun = false;

  /// Size of the CallbackFrame in bytes (0 if no frame needed).
  size_t frameSize = 0;
};

/// AOT batch compiler. Extracts multiple LLHD processes into a single
/// combined LLVM module, runs lowering once, and compiles via one engine.
class AOTProcessCompiler {
public:
  explicit AOTProcessCompiler(::mlir::MLIRContext &context);
  ~AOTProcessCompiler();

  /// Check if a process can be compiled. Returns false if the process
  /// contains ops that require the interpreter (moore.wait_event, sim.fork).
  static bool isProcessCompilable(llhd::ProcessOp processOp);

  /// Check if a compilable process is run-to-completion (stackless callback).
  /// Returns true if the process has exactly one llhd.wait at the loop tail
  /// with no time delay — the common `always @(posedge clk)` pattern.
  /// These processes can be executed as direct function calls without a
  /// coroutine stack.
  static bool isRunToCompletion(llhd::ProcessOp processOp);

  /// Classify a process into one of five ExecModel variants using the
  /// 6-step algorithm:
  ///   A. Fast coroutine filters (sim.fork, moore.wait_event, suspending calls)
  ///   B. Collect suspension points (count llhd.wait ops)
  ///   C. SCC sink verification (every path from resumeBlock reaches waitBlock)
  ///   D. Init run detection (entry block ≠ resumeBlock)
  ///   E. Wait kind classification (delay-only, static signals, dynamic)
  ///   F. Static sensitivity extraction (for CallbackStaticObserved)
  ///
  /// @param processOp     The process to classify.
  /// @param valueToSignal Mapping of signal Values → SignalIds (for step F).
  /// @return CallbackPlan describing how to compile and dispatch the process.
  static CallbackPlan classifyProcess(
      llhd::ProcessOp processOp,
      const llvm::DenseMap<::mlir::Value, SignalId> &valueToSignal);

  /// Compile all eligible processes in a single batch to native code.
  ///
  /// @param processes     List of (ProcessId, ProcessOp) pairs to compile.
  /// @param valueToSignal Mapping of MLIR signal Values → SignalIds.
  /// @param parentModule  The root module (for type info and context).
  /// @param[out] results  Populated with compiled process info.
  /// @return true on success.
  bool compileAllProcesses(
      const llvm::SmallVector<std::pair<ProcessId, llhd::ProcessOp>>
          &processes,
      const llvm::DenseMap<::mlir::Value, SignalId> &valueToSignal,
      ::mlir::ModuleOp parentModule,
      llvm::SmallVector<AOTCompiledProcess> &results);

private:
  ::mlir::MLIRContext &mlirContext;

  /// Keep execution engines alive (they own the compiled code).
  std::vector<std::unique_ptr<::mlir::ExecutionEngine>> engines;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_AOTPROCESSCOMPILER_H
