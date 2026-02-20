//===- circt-sim.cpp - CIRCT Event-Driven Simulation Tool -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-sim' tool, which provides event-driven
// simulation of hardware designs using the CIRCT simulation infrastructure.
// It supports:
// - Event-driven simulation with IEEE 1800 scheduling semantics
// - VCD/FST waveform output
// - Multi-core parallel simulation
// - DPI-C foreign function interface
// - UVM-compatible simulation control
// - Performance profiling
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "JITCompileManager.h"
#include "circt/Runtime/MooreRuntime.h"
#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Sim/EventQueue.h"
#include "circt/Dialect/Sim/ParallelScheduler.h"
#include "circt/Dialect/Sim/PerformanceProfiler.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/Sim/SimulationControl.h"
#include "circt/Dialect/Sim/VPIRuntime.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/Version.h"
#include "circt/Support/WallClockTimeout.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <sys/resource.h>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace mlir;
using namespace circt;
using namespace circt::sim;

#define DEBUG_TYPE "circt-sim"

// Force the linker to keep VPI C API symbols from VPIRuntime.cpp.
// Without this, the static linker drops unreferenced extern "C" functions
// from the CIRCTSim static library, making them invisible to dlopen'd
// VPI libraries (e.g., cocotb).
extern "C" {
extern vpiHandle vpi_register_cb(p_cb_data);
extern PLI_INT32 vpi_remove_cb(vpiHandle);
extern vpiHandle vpi_handle(PLI_INT32, vpiHandle);
extern vpiHandle vpi_handle_by_index(vpiHandle, PLI_INT32);
extern vpiHandle vpi_iterate(PLI_INT32, vpiHandle);
extern vpiHandle vpi_scan(vpiHandle);
extern PLI_INT32 vpi_free_object(vpiHandle);
extern void vpi_get_time(vpiHandle, p_vpi_time);
extern PLI_INT32 vpi_chk_error(p_vpi_error_info);
extern PLI_INT32 vpi_get_vlog_info(p_vpi_vlog_info);
extern PLI_INT32 vpi_control(PLI_INT32, ...);
}
// NOLINTBEGIN(cert-dcl50-cpp)
[[maybe_unused]] static volatile void *vpiSymbolAnchors[] = {
    reinterpret_cast<void *>(&vpi_register_cb),
    reinterpret_cast<void *>(&vpi_remove_cb),
    reinterpret_cast<void *>(&vpi_handle),
    reinterpret_cast<void *>(&vpi_handle_by_index),
    reinterpret_cast<void *>(&vpi_iterate),
    reinterpret_cast<void *>(&vpi_scan),
    reinterpret_cast<void *>(&vpi_free_object),
    reinterpret_cast<void *>(&vpi_get_time),
    reinterpret_cast<void *>(&vpi_chk_error),
    reinterpret_cast<void *>(&vpi_get_vlog_info),
    reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(&vpi_control)),
};
// NOLINTEND(cert-dcl50-cpp)

//===----------------------------------------------------------------------===//
// Loop Partial-Width NBA Accumulation Fix
//===----------------------------------------------------------------------===//
//
// When a partial-width non-blocking assignment (NBA) occurs inside a for-loop:
//   for (i = 0; i < N; i++)
//     reg[i*W +: W] <= data[i];
// the MooreToCore conversion emits a read-modify-write pattern:
//   base = llhd.prb %reg        // read current signal value
//   new = insert(base, data, shift)  // modify
//   llhd.drv %reg, new              // write
// After llhd-mem2reg, the drive value is threaded through block arguments,
// but the probe (which gives the stale pre-edge value) is still used as the
// base for each iteration's insert. This means only the LAST iteration's
// slice survives (last-write-wins).
//
// The fix: in loop bodies, replace uses of the probe with the loop header's
// block argument that carries the accumulated value from previous iterations.

/// Get the operand that a branch terminator passes to a specific block argument
/// of a given successor block.
static Value getBranchArgForSuccessor(Block *predBlock, Block *succBlock,
                                      unsigned argIdx) {
  auto *terminator = predBlock->getTerminator();
  if (auto brOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
    if (brOp.getDest() == succBlock)
      return brOp.getDestOperands()[argIdx];
  } else if (auto condBr = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (condBr.getTrueDest() == succBlock)
      return condBr.getTrueDestOperands()[argIdx];
    if (condBr.getFalseDest() == succBlock)
      return condBr.getFalseDestOperands()[argIdx];
  }
  return {};
}

/// Fix loop partial-width NBA accumulation in all processes in the module.
/// Returns the number of fixes applied.
static int fixLoopPartialDriveAccumulation(Operation *moduleOp) {
  int fixCount = 0;
  moduleOp->walk([&](llhd::ProcessOp processOp) {
    Region &body = processOp.getBody();
    if (body.empty())
      return;

    // Build dominance info for this process.
    DominanceInfo domInfo(processOp);

    // Find loop headers: blocks where at least one predecessor is dominated
    // by the block itself (back-edge).
    for (Block &block : body) {
      if (block.getNumArguments() == 0)
        continue;

      // Collect back-edge and entry predecessors.
      SmallVector<Block *> backEdgePreds, entryPreds;
      for (Block *pred : block.getPredecessors()) {
        if (domInfo.dominates(&block, pred))
          backEdgePreds.push_back(pred);
        else
          entryPreds.push_back(pred);
      }
      if (backEdgePreds.empty() || entryPreds.empty())
        continue;

      // For each block argument, check if initial value is a probe.
      for (unsigned argIdx = 0; argIdx < block.getNumArguments(); argIdx++) {
        Value blockArg = block.getArgument(argIdx);

        // Check: do all entry predecessors provide the same probe value?
        llhd::ProbeOp commonProbe;
        bool allSameProbe = true;
        for (Block *entry : entryPreds) {
          Value initVal = getBranchArgForSuccessor(entry, &block, argIdx);
          if (!initVal) {
            allSameProbe = false;
            break;
          }
          auto probeOp = initVal.getDefiningOp<llhd::ProbeOp>();
          if (!probeOp) {
            allSameProbe = false;
            break;
          }
          if (!commonProbe) {
            commonProbe = probeOp;
          } else if (commonProbe != probeOp) {
            allSameProbe = false;
            break;
          }
        }
        if (!allSameProbe || !commonProbe)
          continue;

        // Check: do back-edge predecessors provide a computed value (not the
        // probe) for this argument?  This confirms it's an accumulator pattern.
        bool isAccumulator = false;
        for (Block *latch : backEdgePreds) {
          Value backVal = getBranchArgForSuccessor(latch, &block, argIdx);
          if (backVal && backVal != commonProbe.getResult())
            isAccumulator = true;
        }
        if (!isAccumulator)
          continue;

        // Find uses of the probe result in blocks dominated by this loop
        // header (i.e., inside the loop body).
        SmallVector<OpOperand *> usesToReplace;
        for (OpOperand &use : commonProbe.getResult().getUses()) {
          Block *useBlock = use.getOwner()->getBlock();
          // Only replace uses strictly inside the loop (dominated by header,
          // but not in the header itself — the header's condition check may
          // legitimately use the probe).
          if (useBlock != &block &&
              domInfo.properlyDominates(&block, useBlock)) {
            usesToReplace.push_back(&use);
          }
        }
        if (usesToReplace.empty())
          continue;

        // Replace the probe uses with the loop-carried block argument.
        LLVM_DEBUG(llvm::dbgs()
                   << "[fix-loop-nba] Replacing " << usesToReplace.size()
                   << " uses of probe " << commonProbe
                   << " with block arg #" << argIdx << " in loop at "
                   << block.front().getLoc() << "\n");
        for (OpOperand *use : usesToReplace)
          use->set(blockArg);
        fixCount += usesToReplace.size();
      }
    }
  });
  return fixCount;
}

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Signal Handler for Clean Shutdown
//===----------------------------------------------------------------------===//

static std::atomic<bool> interruptRequested(false);
static std::atomic<bool> simulationStarted(false);
static void signalHandler(int) { interruptRequested.store(true); }

/// Verilog plusargs (+key, +key=value) extracted from the command line.
/// These are passed through to vpi_get_vlog_info() for cocotb/VPI use.
std::vector<std::string> vlogPlusargs;

static llvm::cl::OptionCategory mainCategory("circt-sim Options");
static llvm::cl::OptionCategory simCategory("Simulation Options");
static llvm::cl::OptionCategory waveCategory("Waveform Options");
static llvm::cl::OptionCategory parallelCategory("Parallel Simulation Options");
static llvm::cl::OptionCategory debugCategory("Debug Options");

// Input/Output options
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"),
                                                llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(mainCategory));

// Top module selection - supports multiple top modules for UVM testbenches
// (e.g., --top hdl_top --top hvl_top)
static llvm::cl::list<std::string>
    topModules("top", llvm::cl::desc("Name of the top module (can be repeated)"),
               llvm::cl::value_desc("name"),
               llvm::cl::cat(mainCategory));

// Simulation control options
static llvm::cl::opt<uint64_t>
    maxTime("max-time",
            llvm::cl::desc("Maximum simulation time in femtoseconds"),
            llvm::cl::init(0), llvm::cl::cat(simCategory));

static llvm::cl::opt<uint64_t>
    maxCycles("max-cycles",
              llvm::cl::desc("Maximum number of clock cycles to simulate"),
              llvm::cl::init(0), llvm::cl::cat(simCategory));

static llvm::cl::opt<uint64_t>
    maxDeltas("max-deltas",
              llvm::cl::desc("Maximum delta cycles per time step (detect loops)"),
              llvm::cl::init(10000), llvm::cl::cat(simCategory));

static llvm::cl::opt<uint64_t>
    maxProcessSteps("max-process-steps",
                    llvm::cl::desc("Maximum total operations per process (0 = no limit)"),
                    llvm::cl::init(0), llvm::cl::cat(simCategory));

static llvm::cl::opt<uint64_t>
    timeout("timeout",
            llvm::cl::desc("Wall-clock timeout in seconds (0 = no timeout)"),
            llvm::cl::init(0), llvm::cl::cat(simCategory));

// Waveform options
static llvm::cl::opt<std::string>
    vcdFile("vcd", llvm::cl::desc("VCD waveform output file"),
            llvm::cl::value_desc("filename"), llvm::cl::init(""),
            llvm::cl::cat(waveCategory));

static llvm::cl::opt<std::string>
    fstFile("fst", llvm::cl::desc("FST compressed waveform output file"),
            llvm::cl::value_desc("filename"), llvm::cl::init(""),
            llvm::cl::cat(waveCategory));

static llvm::cl::opt<bool>
    traceAll("trace-all",
             llvm::cl::desc("Trace all signals (default: only ports)"),
             llvm::cl::init(false), llvm::cl::cat(waveCategory));

static llvm::cl::list<std::string>
    traceSignals("trace",
                 llvm::cl::desc("Trace specific signals (can be repeated)"),
                 llvm::cl::value_desc("signal"), llvm::cl::cat(waveCategory));

// VPI options
static llvm::cl::opt<std::string>
    vpiLibrary("vpi",
               llvm::cl::desc("Load VPI shared library (e.g., cocotb)"),
               llvm::cl::value_desc("library.so"), llvm::cl::init(""),
               llvm::cl::cat(simCategory));

// Parallel simulation options
static llvm::cl::opt<unsigned>
    numThreads("parallel",
               llvm::cl::desc("Number of threads for parallel simulation (0 = auto)"),
               llvm::cl::init(1), llvm::cl::cat(parallelCategory));

static llvm::cl::opt<bool>
    enableWorkStealing("work-stealing",
                       llvm::cl::desc("Enable work stealing for load balancing"),
                       llvm::cl::init(true), llvm::cl::cat(parallelCategory));

static llvm::cl::opt<bool>
    autoPartition("auto-partition",
                  llvm::cl::desc("Automatically partition design for parallel simulation"),
                  llvm::cl::init(true), llvm::cl::cat(parallelCategory));

// Profiling and debug options
static llvm::cl::opt<bool>
    profile("profile", llvm::cl::desc("Enable performance profiling"),
            llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<std::string>
    profileOutput("profile-output",
                  llvm::cl::desc("Profile output file (default: stdout)"),
                  llvm::cl::value_desc("filename"), llvm::cl::init(""),
                  llvm::cl::cat(debugCategory));

static llvm::cl::opt<int>
    verbosity("v", llvm::cl::desc("Verbosity level (0-4)"),
              llvm::cl::init(1), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printStats("sim-stats", llvm::cl::desc("Print simulation statistics"),
               llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printOpStats("op-stats",
                 llvm::cl::desc("Print operation execution statistics"),
                 llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<unsigned>
    opStatsTop("op-stats-top",
               llvm::cl::desc("Number of top operations to print"),
               llvm::cl::init(10), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printProcessStats("process-stats",
                      llvm::cl::desc("Print per-process execution statistics"),
                      llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<unsigned>
    processStatsTop("process-stats-top",
                    llvm::cl::desc("Number of top processes to print"),
                    llvm::cl::init(10), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printProcessOpCounts("process-op-counts",
                         llvm::cl::desc("Print per-process operation counts"),
                         llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<unsigned>
    processOpCountsTop("process-op-counts-top",
                       llvm::cl::desc("Number of top processes to print"),
                       llvm::cl::init(10), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printProcessOpDumps("process-op-counts-dump",
                        llvm::cl::desc("Dump process bodies in analyze mode"),
                        llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printProcessOpBreakdown(
        "process-op-counts-breakdown",
        llvm::cl::desc("Print per-process op breakdown in analyze mode"),
        llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<unsigned>
    processOpBreakdownTop("process-op-counts-breakdown-top",
                          llvm::cl::desc("Number of ops to show per process"),
                          llvm::cl::init(5), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    printProcessOpExtractBreakdown(
        "process-op-counts-breakdown-extracts",
        llvm::cl::desc("Print comb.extract width/offset breakdowns"),
        llvm::cl::init(false), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    verifyPasses("verify-each",
                 llvm::cl::desc("Run verifier after each pass"),
                 llvm::cl::init(true), llvm::cl::cat(debugCategory));

static llvm::cl::opt<bool>
    skipPasses("skip-passes",
               llvm::cl::desc("Skip preprocessing passes (input already lowered)"),
               llvm::cl::init(false), llvm::cl::cat(mainCategory));

// DPI options
static llvm::cl::list<std::string>
    sharedLibs("shared-libs",
               llvm::cl::desc("Shared libraries for DPI-C functions"),
               llvm::cl::value_desc("lib"), llvm::cl::cat(mainCategory));

// Run modes
enum class RunMode { Interpret, Compile, Analyze };
static llvm::cl::opt<RunMode> runMode(
    "mode", llvm::cl::desc("Execution mode"),
    llvm::cl::values(
        clEnumValN(RunMode::Interpret, "interpret",
                   "Interpret the design directly (default)"),
        clEnumValN(RunMode::Compile, "compile",
                   "Compile to native code then simulate"),
        clEnumValN(RunMode::Analyze, "analyze",
                   "Analyze the design without simulating")),
    llvm::cl::init(RunMode::Interpret), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> jitReportPath(
    "jit-report",
    llvm::cl::desc("Write machine-readable JIT telemetry JSON report"),
    llvm::cl::value_desc("path"), llvm::cl::init(""),
    llvm::cl::cat(mainCategory));

static llvm::cl::opt<uint64_t> jitHotThreshold(
    "jit-hot-threshold",
    llvm::cl::desc("Hotness threshold used by compile-mode JIT governor"),
    llvm::cl::init(1), llvm::cl::cat(mainCategory));

static llvm::cl::opt<int64_t> jitCompileBudget(
    "jit-compile-budget",
    llvm::cl::desc("Maximum compile promotions allowed (0 = disabled)"),
    llvm::cl::init(100000), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> jitCachePolicy(
    "jit-cache-policy",
    llvm::cl::desc("JIT cache policy label (for telemetry/governance)"),
    llvm::cl::value_desc("policy"), llvm::cl::init("memory"),
    llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> jitFailOnDeopt(
    "jit-fail-on-deopt",
    llvm::cl::desc("Fail compile-mode run when any JIT deopt occurs"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::StringRef getRunModeName(RunMode mode) {
  switch (mode) {
  case RunMode::Interpret:
    return "interpret";
  case RunMode::Compile:
    return "compile";
  case RunMode::Analyze:
    return "analyze";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// VCD Waveform Writer
//===----------------------------------------------------------------------===//

/// VCD file format writer for waveform output.
class VCDWriter {
public:
  VCDWriter(const std::string &filename) : filename(filename) {}

  bool open() {
    file.open(filename);
    if (!file.is_open()) {
      llvm::errs() << "Error: Could not open VCD file: " << filename << "\n";
      return false;
    }
    return true;
  }

  void close() {
    if (file.is_open())
      file.close();
  }

  void writeHeader(const std::string &moduleName) {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);

    file << "$date\n  " << std::ctime(&timeT) << "$end\n";
    file << "$version\n  CIRCT circt-sim\n$end\n";
    file << "$timescale\n  1fs\n$end\n";
    file << "$scope module " << moduleName << " $end\n";
  }

  void declareSignal(const std::string &name, uint32_t width, char identifier) {
    file << "$var wire " << width << " " << identifier << " " << name
         << " $end\n";
    signalIds[name] = identifier;
  }

  void endHeader() {
    file << "$upscope $end\n";
    file << "$enddefinitions $end\n";
    file << "$dumpvars\n";
  }

  void endDumpVars() { file << "$end\n"; }

  void writeTime(uint64_t time) { file << "#" << time << "\n"; }

  void writeValue(char identifier, uint64_t value, uint32_t width) {
    if (width == 1) {
      file << (value ? "1" : "0") << identifier << "\n";
    } else {
      file << "b";
      for (int i = width - 1; i >= 0; --i) {
        file << ((value >> i) & 1 ? "1" : "0");
      }
      file << " " << identifier << "\n";
    }
  }

  void writeUnknown(char identifier, uint32_t width) {
    if (width == 1) {
      file << "x" << identifier << "\n";
    } else {
      file << "b";
      for (uint32_t i = 0; i < width; ++i) {
        file << "x";
      }
      file << " " << identifier << "\n";
    }
  }

  bool isOpen() const { return file.is_open(); }

private:
  std::string filename;
  std::ofstream file;
  llvm::StringMap<char> signalIds;
};

//===----------------------------------------------------------------------===//
// Simulation Context
//===----------------------------------------------------------------------===//

/// The main simulation context that holds all simulation state.
class SimulationContext {
public:
  SimulationContext()
      : scheduler(ProcessScheduler::Config()),
        control(SimulationControl::Config()),
        profiler(nullptr), vcdWriter(nullptr), parallelScheduler(nullptr) {}

  ~SimulationContext() {
    stopWatchdogThread();
    if (parallelScheduler) {
      parallelScheduler->stopWorkers();
    }
  }

  /// Initialize the simulation from an MLIR module with multiple top modules.
  /// This is the primary interface for UVM testbenches (hdl_top + hvl_top).
  LogicalResult initialize(mlir::ModuleOp module,
                           const llvm::SmallVector<std::string, 4> &tops);

  /// Run the simulation.
  LogicalResult run();

  /// Configure the maximum delta cycles per time step.
  void setMaxDeltaCycles(size_t maxDeltaCycles) {
    scheduler.setMaxDeltaCycles(maxDeltaCycles);
  }

  /// Configure the maximum operations per process activation.
  void setMaxProcessSteps(size_t maxSteps) {
    maxProcessSteps = maxSteps;
    if (llhdInterpreter)
      llhdInterpreter->setMaxProcessSteps(maxSteps);
  }

  /// Attach JIT compile manager for compile-mode thunk/deopt accounting.
  void setJITCompileManager(JITCompileManager *manager) {
    jitCompileManager = manager;
    if (llhdInterpreter)
      llhdInterpreter->setJITCompileManager(jitCompileManager);
  }

  /// Enable runtime func.call_indirect target-set profiling.
  void setJitRuntimeIndirectProfileEnabled(bool enable) {
    jitRuntimeIndirectProfileEnabled = enable;
    if (llhdInterpreter)
      llhdInterpreter->setJitRuntimeIndirectProfileEnabled(enable);
  }

  /// Dump signals that changed in the last delta cycle.
  void dumpLastDeltaSignals(llvm::raw_ostream &os) const {
    scheduler.dumpLastDeltaSignals(os);
  }

  /// Dump processes executed in the last delta cycle.
  void dumpLastDeltaProcesses(llvm::raw_ostream &os) const {
    scheduler.dumpLastDeltaProcesses(os);
  }

  /// Print statistics.
  void printStatistics(llvm::raw_ostream &os) const;

  /// Get the final simulation time.
  const SimTime &getFinalTime() const { return scheduler.getCurrentTime(); }

  /// Get the exit code.
  int getExitCode() const { return control.getExitCode(); }

  /// Get scheduler statistics collected during simulation.
  const ProcessScheduler::Statistics &getSchedulerStats() const {
    return scheduler.getStatistics();
  }

  /// Get simulation-control statistics collected during simulation.
  const SimulationControl::Statistics &getControlStats() const {
    return control.getStatistics();
  }

  /// Get the total error count.
  size_t getErrorCount() const { return control.getErrorCount(); }

  /// Get the total warning count.
  size_t getWarningCount() const { return control.getWarningCount(); }

  /// Get the LLHD process interpreter, if initialized.
  const LLHDProcessInterpreter *getInterpreter() const {
    return llhdInterpreter.get();
  }

private:
  /// Set up waveform tracing.
  LogicalResult setupWaveformTracing();

  /// Set up parallel simulation.
  LogicalResult setupParallelSimulation();

  /// Set up profiling.
  LogicalResult setupProfiling();

  /// Record a value change for waveform output.
  void recordValueChange(SignalId signal, const SignalValue &value);

  /// Register a signal for tracing in the VCD file.
  void registerTracedSignal(SignalId signalId, llvm::StringRef name);

  /// Register traces for requested signals (trace-all or --trace).
  void registerRequestedTraces();

  /// Find the top module in the design.
  hw::HWModuleOp findTopModule(mlir::ModuleOp module, const std::string &name);

  /// Build the simulation model from HW module.
  LogicalResult buildSimulationModel(hw::HWModuleOp hwModule);

  /// Request an abort from another thread or signal handler.
  void requestAbort(llvm::StringRef reason);

  /// Handle a pending abort request on the simulation thread.
  void handleAbort();

  /// Start a wall-clock watchdog thread if timeout is enabled.
  void startWatchdogThread();

  /// Stop and join the watchdog thread.
  void stopWatchdogThread();

  ProcessScheduler scheduler;
  SimulationControl control;
  std::unique_ptr<PerformanceProfiler> profiler;
  std::unique_ptr<VCDWriter> vcdWriter;
  std::unique_ptr<ParallelScheduler> parallelScheduler;
  size_t maxProcessSteps = 50000;

  // Maps from MLIR values to simulation signals
  llvm::DenseMap<mlir::Value, SignalId> valueToSignal;
  llvm::StringMap<SignalId> nameToSignal;
  llvm::StringMap<SignalId> inputPortSignals; // Input-only subset for interpreter

  // Traced signals for VCD output
  llvm::SmallVector<std::pair<SignalId, char>, 64> tracedSignals;
  llvm::DenseSet<SignalId> tracedSignalIds;
  char nextVCDId = '!';
  uint64_t lastVCDTime = 0;
  bool vcdTimeInitialized = false;
  bool vcdReady = false;

  // Module information
  llvm::SmallVector<std::string, 4> topModuleNames;
  llvm::SmallVector<hw::HWModuleOp, 4> topHWModules;
  mlir::ModuleOp rootModule;

  // LLHD Process interpreter
  std::unique_ptr<LLHDProcessInterpreter> llhdInterpreter;
  JITCompileManager *jitCompileManager = nullptr;
  bool jitRuntimeIndirectProfileEnabled = false;

  std::atomic<bool> abortRequested{false};
  std::atomic<bool> abortHandled{false};
  std::atomic<bool> stopWatchdog{false};
  bool inInitializationPhase = true;  // Track if we're still initializing
  std::mutex abortMutex;
  std::string abortReason;
  std::thread watchdogThread;

  // hw.output process IDs for deferred re-evaluation after VPI callbacks.
  // Firreg changes are NOT wired to hw.output via sensitivity (would cause
  // race conditions with cocotb). Instead, these processes are explicitly
  // re-evaluated at the end of each cbAfterDelay cycle.
  llvm::SmallVector<ProcessId, 8> hwOutputProcessIds;
};

LogicalResult SimulationContext::initialize(
    mlir::ModuleOp module,
    const llvm::SmallVector<std::string, 4> &tops) {
  startWatchdogThread();
  rootModule = module;
  scheduler.setShouldAbortCallback([this]() {
    if (interruptRequested.load())
      requestAbort("Interrupt signal received");
    return abortRequested.load();
  });
  // Note: signal change callback is set up AFTER buildSimulationModel()
  // below, because the interpreter's initialize() also sets a callback
  // which would overwrite one set here.

  // Collect all top modules to simulate
  llvm::SmallVector<hw::HWModuleOp, 4> hwModules;

  if (tops.empty()) {
    // Auto-detect split UVM tops when both hdl_top and hvl_top are present.
    // This lets hdl_top interface/config_db setup run alongside hvl_top
    // run_test() without requiring explicit --top flags.
    auto findModuleByName = [&](llvm::StringRef target) -> hw::HWModuleOp {
      hw::HWModuleOp found;
      module.walk([&](hw::HWModuleOp hwModule) {
        if (!found && hwModule.getName() == target)
          found = hwModule;
      });
      return found;
    };

    auto hdlTop = findModuleByName("hdl_top");
    auto hvlTop = findModuleByName("hvl_top");
    if (hdlTop && hvlTop) {
      hwModules.push_back(hdlTop);
      topModuleNames.push_back(hdlTop.getName().str());
      hwModules.push_back(hvlTop);
      topModuleNames.push_back(hvlTop.getName().str());
    } else {
      // No top modules specified - find the last module (typically the top)
      auto hwModule = findTopModule(module, "");
      if (!hwModule) {
        return failure();
      }
      hwModules.push_back(hwModule);
      topModuleNames.push_back(hwModule.getName().str());
    }
  } else {
    // Find all specified top modules
    for (const auto &top : tops) {
      auto hwModule = findTopModule(module, top);
      if (!hwModule) {
        return failure();
      }
      hwModules.push_back(hwModule);
      topModuleNames.push_back(hwModule.getName().str());
    }
  }

  // Save hw.module ops for parameter extraction in VPI initialization.
  topHWModules = hwModules;

  // Report what we're simulating
  if (topModuleNames.size() > 1) {
    llvm::outs() << "[circt-sim] Simulating " << topModuleNames.size()
                 << " top modules: ";
    for (size_t i = 0; i < topModuleNames.size(); ++i) {
      if (i > 0)
        llvm::outs() << ", ";
      llvm::outs() << topModuleNames[i];
    }
    llvm::outs() << "\n";
  }

  // Set up waveform tracing if requested
  if (failed(setupWaveformTracing()))
    return failure();

  // Set up profiling if requested
  if (failed(setupProfiling()))
    return failure();

  // Build the simulation model for all top modules
  // All modules share the same scheduler and interpreter, so signals and
  // processes from all modules run together in the same simulation timeline.
  for (auto hwModule : hwModules) {
    if (failed(buildSimulationModel(hwModule)))
      return failure();
  }

  // Finalize initialization: execute global constructors (UVM init) AFTER all
  // modules' module-level ops have run. This ensures that hdl_top's initial
  // blocks (which call config_db::set) complete before hvl_top's UVM
  // build_phase (which calls config_db::get) starts.
  if (llhdInterpreter) {
    if (failed(llhdInterpreter->finalizeInit()))
      return failure();
  }

  // Set up the unified signal change callback AFTER finalizeInit() completes.
  // finalizeInit() sets its own callback, so we must overwrite it here to
  // include all handlers: module drive re-evaluation, VIF forward propagation,
  // VCD recording, and VPI value-change callbacks.
  scheduler.setSignalChangeCallback(
      [this](SignalId signal, const SignalValue &value) {
        recordValueChange(signal, value);
        if (llhdInterpreter) {
          // Forward-propagate parent interface signal changes to child BFM copies.
          llhdInterpreter->forwardPropagateOnSignalChange(signal, value);
          // Re-evaluate synthetic interface tri-state rules when their source
          // or condition signals change (including non-store updates).
          llhdInterpreter->reevaluateInterfaceTriState(signal);
          // Re-evaluate combinational module drives that depend on this signal.
          // Critical for VPI: when cocotb writes an input via vpi_put_value,
          // continuous assignments must re-compute dependent outputs.
          llhdInterpreter->executeModuleDrivesForSignal(signal);
        }
        // Fire VPI cbValueChange callbacks for cocotb RisingEdge/FallingEdge.
        // Suppress callbacks when the previous value was X (unknown).  Standard
        // simulators don't fire cbValueChange for the initial X→known
        // transition; since VPI returns 0 for X in vpiIntVal format, cocotb
        // would misinterpret X→1 as 0→1 (= spurious rising edge at t=0).
        if (!vpiLibrary.empty()) {
          const SignalValue &prev = scheduler.getSignalPreviousValue(signal);
          if (!prev.isUnknown())
            VPIRuntime::getInstance().fireValueChangeCallbacks(signal);
        }
      });

  if (traceAll || !traceSignals.empty())
    registerRequestedTraces();

  // Set up parallel simulation if multiple threads requested
  if (numThreads > 1) {
    if (failed(setupParallelSimulation()))
      return failure();
  }

  // Configure simulation control
  control.setVerbosity(verbosity);
  if (maxTime > 0) {
    control.setGlobalTimeout(maxTime);
  }

  // Set up watchdog if timeout specified
  if (timeout > 0) {
    control.enableWatchdog(timeout * 1000000000000ULL); // Convert to fs
  }

  // Mark initialization complete - terminations during init are ignored
  inInitializationPhase = false;

  return success();
}

hw::HWModuleOp SimulationContext::findTopModule(mlir::ModuleOp module,
                                                 const std::string &name) {
  hw::HWModuleOp foundModule;
  hw::HWModuleOp lastModule;

  module.walk([&](hw::HWModuleOp hwModule) {
    lastModule = hwModule;
    if (!name.empty() && hwModule.getName() == name) {
      foundModule = hwModule;
    }
  });

  if (foundModule) {
    return foundModule;
  }

  if (!name.empty()) {
    llvm::errs() << "Error: Could not find top module '" << name << "'\n";
    return nullptr;
  }

  // If no name specified, use the last module (typically the top)
  if (lastModule) {
    llvm::outs() << "Using module '" << lastModule.getName()
                 << "' as top module\n";
    return lastModule;
  }

  llvm::errs() << "Error: No hw.module found in input\n";
  return nullptr;
}

LogicalResult SimulationContext::buildSimulationModel(hw::HWModuleOp hwModule) {
  bool tracePortsOnly = !traceAll && traceSignals.empty();

  // Register signals for all ports
  for (auto portInfo : hwModule.getPortList()) {
    // Determine bit width - use getTypeWidth to handle all types including
    // structs, arrays, and other complex types
    unsigned bitWidth;
    if (isa<seq::ClockType>(portInfo.type)) {
      bitWidth = 1;
    } else {
      bitWidth = LLHDProcessInterpreter::getTypeWidth(portInfo.type);
    }

    SignalEncoding encoding =
        LLHDProcessInterpreter::getSignalEncoding(portInfo.type);
    auto signalId = scheduler.registerSignal(portInfo.getName().str(),
                                             bitWidth, encoding);
    nameToSignal[portInfo.getName()] = signalId;

    // Record logical width for VPI (strips 4-state overhead from nested
    // structs).
    unsigned logicalWidth =
        LLHDProcessInterpreter::getLogicalWidth(portInfo.type);
    if (logicalWidth != bitWidth)
      scheduler.setSignalLogicalWidth(signalId, logicalWidth);

    // Detect unpacked array types and record array info for VPI.
    // Only arrays with explicit vpi.array_bounds entries are unpacked —
    // packed multi-dimensional arrays (e.g., logic [3:0][7:0]) are
    // lowered to hw::ArrayType but should appear as flat vectors in VPI.
    if (auto arrayType = dyn_cast<hw::ArrayType>(portInfo.type)) {
      auto boundsDict =
          hwModule->getAttrOfType<DictionaryAttr>("vpi.array_bounds");
      auto sigBounds = boundsDict
          ? boundsDict.getAs<DictionaryAttr>(portInfo.getName())
          : DictionaryAttr();
      // Only create SignalArrayInfo for signals with explicit unpacked
      // bounds. Signals without bounds are packed arrays — treat as flat.
      if (sigBounds) {
        uint32_t numElems = arrayType.getNumElements();
        mlir::Type elemType = arrayType.getElementType();
        uint32_t elemPhysW =
            LLHDProcessInterpreter::getTypeWidth(elemType);
        uint32_t elemLogW =
            LLHDProcessInterpreter::getLogicalWidth(elemType);
        ProcessScheduler::SignalArrayInfo info{numElems, elemPhysW, elemLogW};
        if (sigBounds) {
          if (auto leftAttr = sigBounds.getAs<IntegerAttr>("left"))
            info.leftBound = leftAttr.getInt();
          if (auto rightAttr = sigBounds.getAs<IntegerAttr>("right"))
            info.rightBound = rightAttr.getInt();
        } else {
          // No explicit bounds; default to 0-based indexing.
          info.leftBound = 0;
          info.rightBound = static_cast<int32_t>(numElems - 1);
        }

        // Check for nested unpacked dimensions (e.g., logic x [3][3]).
        // If the element type is also hw::ArrayType and the signal has
        // unpacked depth >= 2, recursively create inner array info.
        auto depthDict =
            hwModule->getAttrOfType<DictionaryAttr>("vpi.unpacked_depth");
        int unpackedDepth = 1;
        if (depthDict) {
          if (auto depthAttr =
                  depthDict.getAs<IntegerAttr>(portInfo.getName()))
            unpackedDepth = depthAttr.getInt();
        }
        if (unpackedDepth >= 2) {
          // Read inner dimension bounds from vpi.array_bounds attribute.
          ArrayAttr innerBoundsAttr = sigBounds
              ? sigBounds.getAs<ArrayAttr>("inner_bounds")
              : ArrayAttr();
          // Build nested inner array info by peeling hw::ArrayType layers.
          auto buildInner =
              [&innerBoundsAttr](mlir::Type type, int depth, int dimIdx,
                 auto &self) -> std::shared_ptr<ProcessScheduler::SignalArrayInfo> {
            if (depth <= 0)
              return nullptr;
            auto innerArray = dyn_cast<hw::ArrayType>(type);
            if (!innerArray)
              return nullptr;
            auto inner = std::make_shared<ProcessScheduler::SignalArrayInfo>();
            inner->numElements = innerArray.getNumElements();
            mlir::Type innerElem = innerArray.getElementType();
            inner->elementPhysWidth =
                LLHDProcessInterpreter::getTypeWidth(innerElem);
            inner->elementLogicalWidth =
                LLHDProcessInterpreter::getLogicalWidth(innerElem);
            // Use stored inner bounds if available.
            if (innerBoundsAttr &&
                dimIdx < static_cast<int>(innerBoundsAttr.size())) {
              auto dimBounds =
                  cast<DictionaryAttr>(innerBoundsAttr[dimIdx]);
              if (auto l = dimBounds.getAs<IntegerAttr>("left"))
                inner->leftBound = l.getInt();
              if (auto r = dimBounds.getAs<IntegerAttr>("right"))
                inner->rightBound = r.getInt();
            } else {
              inner->leftBound = 0;
              inner->rightBound =
                  static_cast<int32_t>(inner->numElements - 1);
            }
            inner->innerArrayInfo =
                self(innerElem, depth - 1, dimIdx + 1, self);
            return inner;
          };
          info.innerArrayInfo =
              buildInner(elemType, unpackedDepth - 1, 0, buildInner);
        }

        scheduler.setSignalArrayInfo(signalId, info);
      }
    }

    // Detect unpacked struct types and record struct field info for VPI.
    if (auto structFieldsDict =
            hwModule->getAttrOfType<DictionaryAttr>("vpi.struct_fields")) {
      if (auto fieldListAttr =
              structFieldsDict.getAs<ArrayAttr>(portInfo.getName())) {
        std::vector<ProcessScheduler::SignalStructFieldInfo> fields;
        for (auto fieldAttr : fieldListAttr) {
          auto fieldDict = cast<DictionaryAttr>(fieldAttr);
          auto name = fieldDict.getAs<StringAttr>("name").getValue().str();
          auto width = fieldDict.getAs<IntegerAttr>("width").getInt();
          ProcessScheduler::SignalStructFieldInfo fi;
          fi.name = name;
          fi.logicalWidth = static_cast<uint32_t>(width);
          fi.physicalWidth = static_cast<uint32_t>(width * 2);
          if (auto isArr = fieldDict.getAs<BoolAttr>("is_array"))
            if (isArr.getValue()) {
              fi.isArray = true;
              if (auto n = fieldDict.getAs<IntegerAttr>("num_elements"))
                fi.numElements = n.getInt();
              if (auto l = fieldDict.getAs<IntegerAttr>("left_bound"))
                fi.leftBound = l.getInt();
              if (auto r = fieldDict.getAs<IntegerAttr>("right_bound"))
                fi.rightBound = r.getInt();
              if (auto ew = fieldDict.getAs<IntegerAttr>("element_width"))
                fi.elementLogicalWidth = ew.getInt();
              fi.elementPhysicalWidth = fi.elementLogicalWidth * 2;
            }
          fields.push_back(std::move(fi));
        }
        scheduler.setSignalStructFields(signalId, std::move(fields));
      }
    }

    // Set up default tracing (ports-only) if enabled.
    if (tracePortsOnly)
      registerTracedSignal(signalId, portInfo.getName().str());
  }

  // Check if this module contains LLHD processes, seq.initial blocks,
  // hw.instance ops, seq.firreg, or llhd.combinational ops
  bool hasLLHDProcesses = false;
  bool hasSeqInitial = false;
  bool hasInstances = false;
  bool hasFirRegs = false;
  bool hasCombinationals = false;
  bool hasLLHDSignals = false;
  size_t llhdProcessCount = 0;
  size_t seqInitialCount = 0;
  size_t instanceCount = 0;
  size_t totalOpsCount = 0;

  // Walk all operations in the module body
  // Use walk to recursively find all operations
  hwModule.getOperation()->walk([&](Operation *op) -> WalkResult {
    if (abortRequested.load())
      return WalkResult::interrupt();
    totalOpsCount++;
    if (isa<llhd::ProcessOp>(op)) {
      hasLLHDProcesses = true;
      llhdProcessCount++;
    } else if (isa<seq::InitialOp>(op)) {
      hasSeqInitial = true;
      seqInitialCount++;
    } else if (isa<hw::InstanceOp>(op)) {
      hasInstances = true;
      instanceCount++;
    } else if (isa<seq::FirRegOp>(op)) {
      hasFirRegs = true;
    } else if (isa<llhd::CombinationalOp>(op)) {
      hasCombinationals = true;
    } else if (isa<llhd::SignalOp, llhd::DriveOp>(op)) {
      hasLLHDSignals = true;
    } else if (auto outOp = dyn_cast<hw::OutputOp>(op)) {
      // If hw.output has non-trivial operands (not just block args),
      // we need the interpreter to evaluate them.
      for (auto val : outOp.getOperands()) {
        if (!isa<mlir::BlockArgument>(val)) {
          hasCombinationals = true;
          break;
        }
      }
    }
    return WalkResult::advance();
  });
  if (abortRequested.load())
    return failure();

  llvm::outs() << "[circt-sim] Found " << llhdProcessCount << " LLHD processes"
               << ", " << seqInitialCount << " seq.initial blocks"
               << ", and " << instanceCount << " hw.instance ops"
               << " (out of " << totalOpsCount << " total ops) in module\n";

  // Initialize the interpreter if we have processes, initial blocks,
  // instances, firreg, combinational ops, or LLHD signals/drives
  if (hasLLHDProcesses || hasSeqInitial || hasInstances || hasFirRegs ||
      hasCombinationals || hasLLHDSignals) {
    // Create the interpreter if it doesn't exist yet (supports multiple top
    // modules - the interpreter accumulates signals and processes across calls)
    if (!llhdInterpreter) {
      llhdInterpreter = std::make_unique<LLHDProcessInterpreter>(scheduler);
      llhdInterpreter->setMaxProcessSteps(maxProcessSteps);
      llhdInterpreter->setCollectOpStats(printOpStats);
      llhdInterpreter->setShouldAbortCallback([this]() {
        if (interruptRequested.load())
          requestAbort("Interrupt signal received");
        return abortRequested.load();
      });
      llhdInterpreter->setAbortCallback([this]() { handleAbort(); });
      llhdInterpreter->setJitRuntimeIndirectProfileEnabled(
          jitRuntimeIndirectProfileEnabled);

      // Set up terminate callback to signal SimulationControl (only once)
      llhdInterpreter->setTerminateCallback(
          [this](bool success, bool verbose) {
            // During initialization phase (global constructors), ignore terminations.
            // UVM's global init may call sim.terminate (via $finish in error paths)
            // but we want the actual simulation to run.
            if (inInitializationPhase) {
              // Note: Using verbosity check instead of LLVM_DEBUG to avoid
              // needing DEBUG_TYPE definition in lambda
              if (verbosity >= 2)
                llvm::errs() << "[circt-sim] Ignoring sim.terminate during "
                             << "initialization phase\n";
              return;
            }
            // Always print termination info for debugging - this helps diagnose
            // silent terminations from fatal errors (e.g., UVM die() -> $finish)
            llvm::errs() << "[circt-sim] Simulation terminated at time "
                         << scheduler.getCurrentTime().realTime << " fs"
                         << " (success=" << (success ? "true" : "false")
                         << ", verbose=" << (verbose ? "true" : "false") << ")\n";
            if (verbose) {
              llvm::outs() << "[circt-sim] Simulation "
                           << (success ? "finished" : "failed") << " at time "
                           << scheduler.getCurrentTime().realTime << " fs\n";
            }
            int code = success ? 0 : 1;
            // UVM's die() calls bare $finish (success=true) even after
            // UVM_FATAL.  Check the error count to catch this case.
            if (code == 0 && control.getErrorCount() > 0)
              code = 1;
            control.finish(code);
          });

      // Set up $dumpfile callback to create VCD file at runtime.
      llhdInterpreter->setDumpfileCallback(
          [this](llvm::StringRef filename) {
            if (vcdWriter) return; // already open
            vcdWriter = std::make_unique<VCDWriter>(std::string(filename));
            if (!vcdWriter->open()) {
              llvm::errs() << "[circt-sim] Warning: $dumpfile could not open '"
                           << filename << "'\n";
              vcdWriter.reset();
            }
          });

      // Set up $dumpvars callback to register signals and write VCD header.
      llhdInterpreter->setDumpvarsCallback(
          [this]() {
            if (!vcdWriter || vcdReady) return;
            // Register all signals for tracing.
            const auto &signalNames = scheduler.getSignalNames();
            for (const auto &entry : signalNames)
              registerTracedSignal(entry.first, entry.second);
            // Write VCD header.
            std::string vcdTopName =
                topModuleNames.empty() ? "top" : topModuleNames[0];
            vcdWriter->writeHeader(vcdTopName);
            for (auto &traced : tracedSignals) {
              auto it = signalNames.find(traced.first);
              if (it == signalNames.end()) continue;
              const auto &value = scheduler.getSignalValue(traced.first);
              vcdWriter->declareSignal(it->second, value.getWidth(),
                                       traced.second);
            }
            vcdWriter->endHeader();
            vcdWriter->writeTime(0);
            lastVCDTime = 0;
            vcdTimeInitialized = true;
            for (auto &traced : tracedSignals) {
              const auto &value = scheduler.getSignalValue(traced.first);
              if (value.isUnknown())
                vcdWriter->writeUnknown(traced.second, value.getWidth());
              else
                vcdWriter->writeValue(traced.second, value.getValue(),
                                      value.getWidth());
            }
            vcdWriter->endDumpVars();
            vcdReady = true;
          });
    }

    llhdInterpreter->setCompileModeEnabled(runMode == RunMode::Compile);
    llhdInterpreter->setJITCompileManager(jitCompileManager);
    llhdInterpreter->setJitRuntimeIndirectProfileEnabled(
        jitRuntimeIndirectProfileEnabled);

    // Provide port signal mappings so the interpreter can map non-ref-type
    // block arguments (e.g., four-state struct ports) to signal IDs.
    // IMPORTANT: Only include INPUT ports. Output ports share argNum indices
    // with input ports but refer to hw.output operands, not block arguments.
    // Including output ports would overwrite input block arg → signal mappings.
    inputPortSignals.clear();
    for (auto portInfo : hwModule.getPortList()) {
      if (portInfo.isOutput())
        continue;
      auto it = nameToSignal.find(portInfo.getName());
      if (it != nameToSignal.end())
        inputPortSignals[portInfo.getName()] = it->second;
    }
    llhdInterpreter->setPortSignalMap(inputPortSignals);

    // Initialize this module (will add to existing signals and processes)
    if (failed(llhdInterpreter->initialize(hwModule))) {
      llvm::errs() << "Error: Failed to initialize LLHD process interpreter\n";
      return failure();
    }

    llvm::outs() << "[circt-sim] Registered " << llhdInterpreter->getNumSignals()
                 << " LLHD signals and " << llhdInterpreter->getNumProcesses()
                 << " LLHD processes/initial blocks\n";

    // Register combinational processes for hw.output operands of the top-level
    // module. The interpreter handles llhd.drv via registerContinuousAssignments
    // but hw.output is the module terminator that defines output port values.
    // Without this, output port signals are only computed once during init and
    // never re-evaluated when input signals change (e.g., via VPI writes).
    if (auto *bodyBlock = hwModule.getBodyBlock()) {
      if (auto outputOp = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator())) {
        // Collect input port signal IDs for sensitivity
        llvm::SmallVector<SignalId, 4> inputSigIds;
        for (auto portInfo : hwModule.getPortList()) {
          if (!portInfo.isOutput()) {
            auto it = nameToSignal.find(portInfo.getName());
            if (it != nameToSignal.end())
              inputSigIds.push_back(it->second);
          }
        }

        // Also collect internal LLHD signal IDs for sensitivity
        llvm::SmallVector<SignalId, 16> internalSigIds;
        hwModule.walk([&](llhd::SignalOp sigOp) {
          SignalId sigId = llhdInterpreter->getSignalId(sigOp.getResult());
          if (sigId != 0)
            internalSigIds.push_back(sigId);
        });
        // NOTE: firreg signal IDs are NOT added to the sensitivity list.
        // Making hw.output sensitive to firreg causes race conditions with
        // cocotb's RisingEdge callbacks: firreg captures the new value and
        // hw.output updates the port in the same executeCurrentTime() call
        // as the clock write, before cocotb reads the port. Instead, firreg
        // changes are propagated to ports via a post-callback hook that runs
        // AFTER all VPI callbacks for the current time step.

        // Build output port name → signal ID mapping
        auto outputPorts = hwModule.getPortList();
        unsigned outputIdx = 0;
        for (auto portInfo : outputPorts) {
          if (!portInfo.isOutput())
            continue;
          if (outputIdx >= outputOp.getNumOperands())
            break;
          mlir::Value outputValue = outputOp.getOperand(outputIdx);
          auto sigIt = nameToSignal.find(portInfo.getName());
          if (sigIt == nameToSignal.end()) {
            ++outputIdx;
            continue;
          }
          SignalId outSigId = sigIt->second;

          // Create a combinational process that re-evaluates this output
          std::string procName = "hw_output_" + portInfo.getName().str();
          auto procId = scheduler.registerProcess(
              procName,
              [this, outputValue, outSigId]() {
                InterpretedValue val =
                    llhdInterpreter->evaluateContinuousValue(outputValue);
                scheduler.updateSignal(outSigId, val.toSignalValue());
              });

          hwOutputProcessIds.push_back(procId);

          // Register a top-level output update so that when the
          // llhd.process yields fresh values, executeInstanceOutputUpdates
          // immediately re-evaluates and drives the output signal.  This
          // is critical for multi-block process CFGs where the inline
          // combinational evaluation falls back to reading the process
          // valueMap — without this, the hw_output_* process may read a
          // stale valueMap if it runs before the process in the same
          // delta cycle.
          llhdInterpreter->registerTopLevelOutputUpdate(outSigId,
                                                        outputValue);

          auto *proc = scheduler.getProcess(procId);
          if (proc) {
            proc->setCombinational(true);
            // Sensitive to all input ports
            for (SignalId inSig : inputSigIds)
              scheduler.addSensitivity(procId, inSig);
            // Sensitive to all internal LLHD signals
            for (SignalId intSig : internalSigIds)
              scheduler.addSensitivity(procId, intSig);
          }
          ++outputIdx;
        }
      }
    }

    // Execute hw.output processes once after registration to set initial
    // output values. The module-level llhd.drv ops schedule changes for 1
    // epsilon during initialize(), but the hw_output processes are registered
    // AFTER initialize() returns, so they miss the initial signal changes.
    // Without this, output ports stay at their uninitialized values (Z/X)
    // until an input actually changes (which never happens if VPI writes
    // the same value the signal was initialized to).
    if (!hwOutputProcessIds.empty()) {
      // First, flush any pending drives from module-level initialization.
      scheduler.executeCurrentTime();
      for (ProcessId pid : hwOutputProcessIds) {
        auto *proc = scheduler.getProcess(pid);
        if (proc)
          proc->execute();
      }
      scheduler.executeCurrentTime();
    }
  } else {
    // For modules without LLHD processes, register combinational processes
    // for hw.output operands directly. Since the interpreter is not created,
    // we can only handle direct input→output wiring (hw.output %inputPort).
    if (auto *bodyBlock = hwModule.getBodyBlock()) {
      if (auto outputOp = dyn_cast<hw::OutputOp>(bodyBlock->getTerminator())) {
        // Build a block-argument-index to input signal map.
        llvm::DenseMap<unsigned, SignalId> argIdxToSignal;
        for (auto portInfo : hwModule.getPortList()) {
          if (portInfo.isOutput())
            continue;
          auto it = nameToSignal.find(portInfo.getName());
          if (it != nameToSignal.end())
            argIdxToSignal[portInfo.argNum] = it->second;
        }

        auto outputPorts = hwModule.getPortList();
        unsigned outputIdx = 0;
        for (auto portInfo : outputPorts) {
          if (!portInfo.isOutput())
            continue;
          if (outputIdx >= outputOp.getNumOperands())
            break;
          mlir::Value outputValue = outputOp.getOperand(outputIdx);
          auto sigIt = nameToSignal.find(portInfo.getName());
          if (sigIt == nameToSignal.end()) {
            ++outputIdx;
            continue;
          }
          SignalId outSigId = sigIt->second;

          // Check if the output operand is a block argument (direct wiring).
          if (auto blockArg = dyn_cast<mlir::BlockArgument>(outputValue)) {
            auto inIt = argIdxToSignal.find(blockArg.getArgNumber());
            if (inIt != argIdxToSignal.end()) {
              SignalId inSigId = inIt->second;
              std::string procName = "hw_output_" + portInfo.getName().str();
              auto procId = scheduler.registerProcess(
                  procName,
                  [this, inSigId, outSigId]() {
                    const SignalValue &sv =
                        scheduler.getSignalValue(inSigId);
                    scheduler.updateSignal(outSigId, sv);
                  });
              auto *proc = scheduler.getProcess(procId);
              if (proc) {
                proc->setCombinational(true);
                proc->setState(ProcessState::Suspended);
                scheduler.addSensitivity(procId, inSigId);
              }
            }
          }
          ++outputIdx;
        }
      }
    }
  }

  return success();
}

void SimulationContext::requestAbort(llvm::StringRef reason) {
  std::lock_guard<std::mutex> lock(abortMutex);
  if (abortRequested.load())
    return;
  abortReason = reason.str();
  abortRequested.store(true);
}

void SimulationContext::handleAbort() {
  if (!abortRequested.load())
    return;
  if (abortHandled.exchange(true))
    return;
  std::string reason;
  {
    std::lock_guard<std::mutex> lock(abortMutex);
    reason = abortReason;
  }
  if (reason.empty())
    reason = "Abort requested";
  llvm::errs() << "[circt-sim] " << reason << "\n";
  if (llhdInterpreter)
    llhdInterpreter->dumpProcessStates(llvm::errs());
  control.finish(1);
}

void SimulationContext::startWatchdogThread() {
  if (timeout == 0)
    return;
  if (watchdogThread.joinable())
    return;
  stopWatchdog.store(false);
  watchdogThread = std::thread([this]() {
    auto start = std::chrono::steady_clock::now();
    while (!stopWatchdog.load()) {
      if (abortRequested.load() || interruptRequested.load())
        break;
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - start);
      if (static_cast<uint64_t>(elapsed.count()) >= timeout) {
        requestAbort("Wall-clock timeout reached");
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
}

void SimulationContext::stopWatchdogThread() {
  stopWatchdog.store(true);
  if (watchdogThread.joinable())
    watchdogThread.join();
}

LogicalResult SimulationContext::setupWaveformTracing() {
  if (vcdFile.empty())
    return success();

  vcdWriter = std::make_unique<VCDWriter>(vcdFile);
  if (!vcdWriter->open())
    return failure();

  return success();
}

void SimulationContext::registerTracedSignal(SignalId signalId,
                                             llvm::StringRef name) {
  if (!vcdWriter)
    return;
  if (nextVCDId > '~')
    return;
  if (tracedSignalIds.count(signalId))
    return;

  tracedSignals.push_back({signalId, nextVCDId++});
  tracedSignalIds.insert(signalId);
}

void SimulationContext::registerRequestedTraces() {
  if (!vcdWriter)
    return;

  const auto &signalNames = scheduler.getSignalNames();

  if (traceAll) {
    for (const auto &entry : signalNames)
      registerTracedSignal(entry.first, entry.second);
    if (nextVCDId > '~')
      llvm::errs() << "[circt-sim] Warning: VCD signal limit reached; "
                      "additional signals are not traced\n";
    return;
  }

  if (traceSignals.empty())
    return;

  llvm::SmallVector<std::string, 4> topPrefixes;
  topPrefixes.reserve(topModuleNames.size());
  for (const auto &topName : topModuleNames)
    topPrefixes.push_back(topName + ".");

  for (const auto &requested : traceSignals) {
    llvm::StringRef requestedRef(requested);
    bool matched = false;
    for (const auto &entry : signalNames) {
      if (entry.second == requested) {
        registerTracedSignal(entry.first, entry.second);
        matched = true;
      }
    }
    if (!matched) {
      for (const auto &prefix : topPrefixes) {
        if (!requestedRef.starts_with(prefix))
          continue;
        llvm::StringRef stripped = requestedRef.drop_front(prefix.size());
        for (const auto &entry : signalNames) {
          if (entry.second == stripped) {
            registerTracedSignal(entry.first, entry.second);
            matched = true;
          }
        }
      }
    }
    if (!matched) {
      llvm::errs() << "[circt-sim] Warning: trace signal '" << requested
                   << "' not found\n";
    }
  }
}

LogicalResult SimulationContext::setupProfiling() {
  if (!profile)
    return success();

  PerformanceProfiler::Config config;
  config.enabled = true;
  config.profileProcesses = true;
  config.profileSignals = true;
  config.collectHistograms = true;

  profiler = std::make_unique<PerformanceProfiler>(config);
  profiler->startSession("circt-sim");

  return success();
}

LogicalResult SimulationContext::setupParallelSimulation() {
  auto isTruthyEnv = [](const char *value) {
    if (!value)
      return false;
    llvm::StringRef v(value);
    return v.equals_insensitive("1") || v.equals_insensitive("true") ||
           v.equals_insensitive("yes") || v.equals_insensitive("on");
  };

  // The current parallel scheduler remains experimental and may deadlock on
  // LLHD-heavy workloads. Keep --parallel CLI compatibility, but default to
  // stable sequential execution unless explicitly opted in.
  const bool enableExperimentalParallel =
      isTruthyEnv(std::getenv("CIRCT_SIM_EXPERIMENTAL_PARALLEL"));
  if (!enableExperimentalParallel) {
    llvm::errs()
        << "[circt-sim] Warning: parallel scheduler is temporarily disabled "
           "by default due stability issues; running sequentially. Set "
           "CIRCT_SIM_EXPERIMENTAL_PARALLEL=1 to force-enable.\n";
    return success();
  }

  ParallelScheduler::Config config;
  config.numThreads = numThreads;
  config.enableWorkStealing = enableWorkStealing;
  config.enableDynamicBalancing = true;
  config.debugLevel = verbosity;

  parallelScheduler =
      std::make_unique<ParallelScheduler>(scheduler, config);

  if (autoPartition) {
    parallelScheduler->autoPartition();
  }

  parallelScheduler->startWorkers();
  return success();
}

void SimulationContext::recordValueChange(SignalId signal,
                                           const SignalValue &value) {
  if (!vcdWriter || !vcdReady)
    return;

  uint64_t currentTime = scheduler.getCurrentTime().realTime;
  if (!vcdTimeInitialized) {
    lastVCDTime = currentTime;
    vcdTimeInitialized = true;
  }
  if (currentTime != lastVCDTime) {
    vcdWriter->writeTime(currentTime);
    lastVCDTime = currentTime;
  }

  for (auto &traced : tracedSignals) {
    if (traced.first == signal) {
      if (value.isUnknown()) {
        vcdWriter->writeUnknown(traced.second, value.getWidth());
      } else {
        vcdWriter->writeValue(traced.second, value.getValue(), value.getWidth());
      }
      break;
    }
  }
}

LogicalResult SimulationContext::run() {
  startWatchdogThread();
  simulationStarted.store(true);

  // Write VCD header
  if (vcdWriter) {
    // Use combined name for multiple top modules
    std::string vcdTopName = topModuleNames.empty() ? "top" : topModuleNames[0];
    if (topModuleNames.size() > 1) {
      vcdTopName = "multi_top";  // Indicate multi-top simulation
    }
    vcdWriter->writeHeader(vcdTopName);
    const auto &signalNames = scheduler.getSignalNames();
    for (auto &traced : tracedSignals) {
      auto it = signalNames.find(traced.first);
      if (it == signalNames.end()) {
        llvm::errs() << "[circt-sim] Warning: missing name for signal "
                     << traced.first << " in VCD trace\n";
        continue;
      }
      const auto &value = scheduler.getSignalValue(traced.first);
      vcdWriter->declareSignal(it->second, value.getWidth(), traced.second);
    }
    vcdWriter->endHeader();
    vcdWriter->writeTime(0);
    lastVCDTime = 0;
    vcdTimeInitialized = true;
    // Write initial unknown values
    for (auto &traced : tracedSignals) {
      const auto &value = scheduler.getSignalValue(traced.first);
      if (value.isUnknown()) {
        vcdWriter->writeUnknown(traced.second, value.getWidth());
      } else {
        vcdWriter->writeValue(traced.second, value.getValue(), value.getWidth());
      }
    }
    vcdWriter->endDumpVars();
    vcdReady = true;
  }

  // Start profiling
  auto simulationStartTime = std::chrono::high_resolution_clock::now();

  // Initialize the scheduler
  scheduler.initialize();

  // Initialize VPI runtime if a VPI library was specified.
  if (!vpiLibrary.empty()) {
    auto &vpiRuntime = VPIRuntime::getInstance();
    vpiRuntime.setScheduler(&scheduler);
    vpiRuntime.setSimulationControl(&control);
    vpiRuntime.setTopModuleNames(topModuleNames);

    // Pass command-line args (including plusargs) to VPI for vpi_get_vlog_info.
    {
      std::vector<std::string> vlogArgs;
      vlogArgs.push_back("circt-sim");
      for (const auto &pa : vlogPlusargs)
        vlogArgs.push_back(pa);
      vpiRuntime.setVlogArgs(vlogArgs);
    }

    // Populate signal type metadata (integer/string/real) from hw.module attrs.
    // These must be set before buildHierarchy() so that signal type
    // classification is available during VPI object creation.
    for (auto hwMod : topHWModules) {
      if (!hwMod)
        continue;
      if (auto intVars = hwMod->getAttrOfType<mlir::ArrayAttr>(
              "vpi.integer_vars")) {
        for (auto attr : intVars)
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
            vpiRuntime.addIntegerVar(strAttr.getValue().str());
      }
      if (auto strVars = hwMod->getAttrOfType<mlir::ArrayAttr>(
              "vpi.string_vars")) {
        for (auto attr : strVars)
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
            vpiRuntime.addStringVar(strAttr.getValue().str());
      }
      if (auto realVars = hwMod->getAttrOfType<mlir::ArrayAttr>(
              "vpi.real_vars")) {
        for (auto attr : realVars)
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
            vpiRuntime.addRealVar(strAttr.getValue().str());
      }
    }

    // Register synthetic ProcessScheduler signals for module-level variables
    // that were removed by Sig2Reg (zero users) but still need to be
    // discoverable via VPI.  We do this BEFORE buildHierarchy() so that
    // array elements are automatically created.
    {
      // Build a set of existing signal names for dedup.
      llvm::StringSet<> existingSignals;
      for (const auto &entry : scheduler.getSignalNames())
        existingSignals.insert(entry.second);

      for (auto hwMod : topHWModules) {
        if (!hwMod)
          continue;
        auto allVars =
            hwMod->getAttrOfType<mlir::DictionaryAttr>("vpi.all_vars");
        if (!allVars)
          continue;
        auto boundsDict =
            hwMod->getAttrOfType<mlir::DictionaryAttr>("vpi.array_bounds");
        auto depthDict =
            hwMod->getAttrOfType<mlir::DictionaryAttr>("vpi.unpacked_depth");

        for (auto namedAttr : allVars) {
          llvm::StringRef varName = namedAttr.getName().getValue();
          // Skip if already registered as a real signal.
          if (existingSignals.count(varName))
            continue;

          uint32_t logicalWidth = 0;
          if (auto intAttr =
                  dyn_cast<mlir::IntegerAttr>(namedAttr.getValue()))
            logicalWidth = intAttr.getInt();
          if (logicalWidth == 0)
            continue;

          // Register a 4-state signal in ProcessScheduler.
          uint32_t physWidth = logicalWidth * 2;
          auto signalId = scheduler.registerSignal(
              varName.str(), physWidth, SignalEncoding::FourStateStruct);
          scheduler.setSignalLogicalWidth(signalId, logicalWidth);

          // Set up array info if this is an unpacked array.
          auto sigBounds = boundsDict
              ? boundsDict.getAs<DictionaryAttr>(varName)
              : DictionaryAttr();
          if (sigBounds) {
            int32_t left = 0, right = 0;
            if (auto leftAttr = sigBounds.getAs<IntegerAttr>("left"))
              left = leftAttr.getInt();
            if (auto rightAttr = sigBounds.getAs<IntegerAttr>("right"))
              right = rightAttr.getInt();
            uint32_t numElems =
                static_cast<uint32_t>(std::abs(right - left) + 1);
            uint32_t elemLogW =
                (numElems > 0) ? logicalWidth / numElems : logicalWidth;
            uint32_t elemPhysW = elemLogW * 2;
            ProcessScheduler::SignalArrayInfo info{numElems, elemPhysW,
                                                   elemLogW};
            info.leftBound = left;
            info.rightBound = right;

            // Handle nested unpacked dimensions.
            int unpackedDepth = 1;
            if (depthDict) {
              if (auto depthAttr = depthDict.getAs<IntegerAttr>(varName))
                unpackedDepth = depthAttr.getInt();
            }
            if (unpackedDepth >= 2) {
              ArrayAttr innerBoundsAttr =
                  sigBounds.getAs<ArrayAttr>("inner_bounds");
              auto buildInner =
                  [&innerBoundsAttr](
                      uint32_t totalLogW, int depth, int dimIdx,
                      auto &self)
                  -> std::shared_ptr<ProcessScheduler::SignalArrayInfo> {
                if (depth <= 0 || totalLogW == 0)
                  return nullptr;
                // Use inner bounds to compute dimension size.
                int32_t iLeft = 0, iRight = 0;
                if (innerBoundsAttr &&
                    dimIdx < static_cast<int>(innerBoundsAttr.size())) {
                  auto dimBounds =
                      cast<DictionaryAttr>(innerBoundsAttr[dimIdx]);
                  if (auto l = dimBounds.getAs<IntegerAttr>("left"))
                    iLeft = l.getInt();
                  if (auto r = dimBounds.getAs<IntegerAttr>("right"))
                    iRight = r.getInt();
                }
                uint32_t n = static_cast<uint32_t>(
                    std::abs(iRight - iLeft) + 1);
                if (n == 0)
                  return nullptr;
                auto inner = std::make_shared<
                    ProcessScheduler::SignalArrayInfo>();
                inner->numElements = n;
                uint32_t innerElemLogW = totalLogW / n;
                inner->elementLogicalWidth = innerElemLogW;
                inner->elementPhysWidth = innerElemLogW * 2;
                inner->leftBound = iLeft;
                inner->rightBound = iRight;
                inner->innerArrayInfo =
                    self(innerElemLogW, depth - 1, dimIdx + 1, self);
                return inner;
              };
              info.innerArrayInfo =
                  buildInner(elemLogW, unpackedDepth - 1, 0, buildInner);
            }

            scheduler.setSignalArrayInfo(signalId, info);
          }

          // Set struct field info for synthetic signals if available.
          auto structFieldsDict =
              hwMod->getAttrOfType<mlir::DictionaryAttr>("vpi.struct_fields");
          if (structFieldsDict) {
            if (auto fieldListAttr =
                    structFieldsDict.getAs<ArrayAttr>(varName)) {
              std::vector<ProcessScheduler::SignalStructFieldInfo> fields;
              for (auto fieldAttr : fieldListAttr) {
                auto fieldDict = cast<DictionaryAttr>(fieldAttr);
                auto name =
                    fieldDict.getAs<StringAttr>("name").getValue().str();
                auto width = fieldDict.getAs<IntegerAttr>("width").getInt();
                ProcessScheduler::SignalStructFieldInfo fi;
                fi.name = name;
                fi.logicalWidth = static_cast<uint32_t>(width);
                fi.physicalWidth = static_cast<uint32_t>(width * 2);
                if (auto isArr = fieldDict.getAs<BoolAttr>("is_array"))
                  if (isArr.getValue()) {
                    fi.isArray = true;
                    if (auto n =
                            fieldDict.getAs<IntegerAttr>("num_elements"))
                      fi.numElements = n.getInt();
                    if (auto l =
                            fieldDict.getAs<IntegerAttr>("left_bound"))
                      fi.leftBound = l.getInt();
                    if (auto r =
                            fieldDict.getAs<IntegerAttr>("right_bound"))
                      fi.rightBound = r.getInt();
                    if (auto ew =
                            fieldDict.getAs<IntegerAttr>("element_width"))
                      fi.elementLogicalWidth = ew.getInt();
                    fi.elementPhysicalWidth = fi.elementLogicalWidth * 2;
                  }
                fields.push_back(std::move(fi));
              }
              scheduler.setSignalStructFields(signalId, std::move(fields));
            }
          }

          existingSignals.insert(varName);
        }
      }
    }

    vpiRuntime.buildHierarchy();

    // Register module parameters from hw.module ops so that cocotb can
    // access them via vpi_handle_by_name / vpi_iterate(vpiParameter, ...).
    // Parameters are stored as a "vpi.parameters" dictionary attribute on
    // hw.module ops after elaboration (the standard "parameters" attribute
    // is consumed during parameter specialization).
    for (auto hwMod : topHWModules) {
      if (!hwMod)
        continue;
      std::string modName = hwMod.getName().str();
      // Find the VPI module object for this hw.module.
      auto *vpiMod = vpiRuntime.findByName(modName);
      uint32_t modId = vpiMod ? vpiMod->id : 0;

      // Try vpi.parameters dictionary attribute first (post-elaboration).
      if (auto vpiParams = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.parameters")) {
        for (auto namedAttr : vpiParams) {
          llvm::StringRef paramName = namedAttr.getName().getValue();
          mlir::Attribute valueAttr = namedAttr.getValue();
          int64_t paramValue = 0;
          uint32_t paramWidth = 32;
          bool isRealParam = false;
          bool isStringParam = false;
          std::string stringParamValue;
          if (auto strAttr = dyn_cast<mlir::StringAttr>(valueAttr)) {
            isStringParam = true;
            stringParamValue = strAttr.getValue().str();
          } else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(valueAttr)) {
            paramValue = intAttr.getValue().getSExtValue();
            paramWidth = intAttr.getValue().getBitWidth();
          } else if (auto floatAttr = dyn_cast<mlir::FloatAttr>(valueAttr)) {
            double dval = floatAttr.getValueAsDouble();
            std::memcpy(&paramValue, &dval, sizeof(double));
            paramWidth = 64;
            isRealParam = true;
          }
          std::string qualifiedName = modName + "." + paramName.str();
          if (isStringParam) {
            // Register as a string variable so VPI reports vpiStringVar.
            uint32_t paramId = vpiRuntime.registerStringVariable(
                paramName.str(), qualifiedName, stringParamValue, modId);
            auto *paramObj = vpiRuntime.findById(paramId);
            if (paramObj)
              paramObj->paramConstType = 3; // vpiStringConst
          } else {
          uint32_t paramId = vpiRuntime.registerParameter(
              paramName.str(), qualifiedName, paramValue, paramWidth, modId);
          if (isRealParam) {
            auto *paramObj = vpiRuntime.findById(paramId);
            if (paramObj)
              paramObj->paramConstType = 2; // vpiRealConst
          }
          }
          // Also register under unqualified name for vpi_handle_by_name.
          auto *existingUnqual = vpiRuntime.findByName(paramName.str());
          if (!existingUnqual) {
            auto *paramObj = vpiRuntime.findByName(qualifiedName);
            if (paramObj)
              vpiRuntime.addNameMapping(paramName.str(), paramObj->id);
          }
        }
      }

      // Also try standard parameters attribute (pre-elaboration modules).
      auto params = hwMod.getParameters();
      if (params && !params.empty()) {
        for (auto paramAttr : params) {
          auto paramDecl = dyn_cast<hw::ParamDeclAttr>(paramAttr);
          if (!paramDecl)
            continue;
          llvm::StringRef paramName = paramDecl.getName().getValue();
          // Skip if already registered from vpi.parameters.
          std::string qualifiedName = modName + "." + paramName.str();
          if (vpiRuntime.findByName(qualifiedName))
            continue;
          mlir::Attribute valueAttr = paramDecl.getValue();
          int64_t paramValue = 0;
          uint32_t paramWidth = 32;
          if (auto intAttr =
                  dyn_cast_or_null<mlir::IntegerAttr>(valueAttr)) {
            paramValue = intAttr.getValue().getSExtValue();
            paramWidth = intAttr.getValue().getBitWidth();
          }
          vpiRuntime.registerParameter(paramName.str(), qualifiedName,
                                       paramValue, paramWidth, modId);
          auto *existingUnqual = vpiRuntime.findByName(paramName.str());
          if (!existingUnqual) {
            auto *paramObj = vpiRuntime.findByName(qualifiedName);
            if (paramObj)
              vpiRuntime.addNameMapping(paramName.str(), paramObj->id);
          }
        }
      }

      // Register synthetic string variables from vpi.string_var_values.
      // These are string-typed SV variables that don't have backing signals
      // but need to be discoverable via VPI for cocotb.
      if (auto strVarVals = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.string_var_values")) {
        for (auto namedAttr : strVarVals) {
          llvm::StringRef varName = namedAttr.getName().getValue();
          std::string qualifiedName = modName + "." + varName.str();
          // Skip if already registered as a signal-backed string var.
          if (vpiRuntime.findByName(qualifiedName))
            continue;
          std::string initVal;
          if (auto strAttr = dyn_cast<mlir::StringAttr>(namedAttr.getValue()))
            initVal = strAttr.getValue().str();
          vpiRuntime.registerStringVariable(
              varName.str(), qualifiedName, initVal, modId);
        }
      }

      // Register generate scopes from vpi.gen_scopes attribute.
      if (auto genScopes = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.gen_scopes")) {
        for (auto namedAttr : genScopes) {
          llvm::StringRef scopeName = namedAttr.getName().getValue();
          auto dictAttr =
              dyn_cast<mlir::DictionaryAttr>(namedAttr.getValue());
          if (!dictAttr)
            continue;
          std::string scopeType;
          if (auto typeAttr = dictAttr.getAs<mlir::StringAttr>("type"))
            scopeType = typeAttr.getValue().str();

          std::string qualifiedName = modName + "." + scopeName.str();
          // Skip if already created by buildHierarchy from signal names.
          if (vpiRuntime.findByName(qualifiedName))
            continue;

          if (scopeType == "array") {
            // Generate array (for-generate): create GenScopeArray + children.
            int32_t left = 0, right = 0;
            if (auto leftAttr = dictAttr.getAs<mlir::IntegerAttr>("left"))
              left = leftAttr.getInt();
            if (auto rightAttr = dictAttr.getAs<mlir::IntegerAttr>("right"))
              right = rightAttr.getInt();
            uint32_t arrayId = vpiRuntime.registerModule(
                scopeName.str(), qualifiedName, modId);
            auto *arrayObj = vpiRuntime.findById(arrayId);
            if (arrayObj) {
              arrayObj->type = VPIObjectType::GenScopeArray;
              arrayObj->leftBound = left;
              arrayObj->rightBound = right;
            }
            // Create GenScope children for each index.
            int32_t lo = std::min(left, right);
            int32_t hi = std::max(left, right);
            for (int32_t idx = lo; idx <= hi; ++idx) {
              std::string childName =
                  scopeName.str() + "[" + std::to_string(idx) + "]";
              std::string childFull = modName + "." + childName;
              if (!vpiRuntime.findByName(childFull)) {
                uint32_t childId = vpiRuntime.registerModule(
                    childName, childFull, arrayId);
                auto *childObj = vpiRuntime.findById(childId);
                if (childObj)
                  childObj->type = VPIObjectType::GenScope;
              }
            }
          } else {
            // Conditional generate scope: create single GenScope.
            uint32_t scopeId = vpiRuntime.registerModule(
                scopeName.str(), qualifiedName, modId);
            auto *scopeObj = vpiRuntime.findById(scopeId);
            if (scopeObj)
              scopeObj->type = VPIObjectType::GenScope;
          }
        }
      }

      // Register parameters inside generate scopes from
      // vpi.gen_scope_params.
      if (auto genParams = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.gen_scope_params")) {
        for (auto namedAttr : genParams) {
          // Qualified name like "cond_scope.scoped_param" or
          // "outer_scope[1].outer_param".
          llvm::StringRef qualParamName = namedAttr.getName().getValue();
          std::string fullParamName =
              modName + "." + qualParamName.str();
          if (vpiRuntime.findByName(fullParamName))
            continue;

          int64_t paramValue = 0;
          uint32_t paramWidth = 32;
          if (auto intAttr =
                  dyn_cast<mlir::IntegerAttr>(namedAttr.getValue())) {
            paramValue = intAttr.getValue().getSExtValue();
            paramWidth = intAttr.getValue().getBitWidth();
          }

          // Find the parent scope: everything before the last '.'.
          // Auto-create intermediate scopes if they don't exist
          // (handles nested generate arrays like
          // outer_scope[1].inner_scope[1]).
          auto lastDot = qualParamName.rfind('.');
          uint32_t parentScopeId = modId;
          if (lastDot != llvm::StringRef::npos) {
            llvm::StringRef parentPath =
                qualParamName.substr(0, lastDot);
            std::string parentFull =
                modName + "." + parentPath.str();
            auto *parentObj = vpiRuntime.findByName(parentFull);
            if (!parentObj) {
              // Auto-create intermediate scope hierarchy.
              uint32_t curParent = modId;
              llvm::SmallVector<llvm::StringRef> parts;
              parentPath.split(parts, '.');
              std::string accumulated = modName;
              for (auto part : parts) {
                accumulated += "." + part.str();
                auto *existing = vpiRuntime.findByName(accumulated);
                if (existing) {
                  curParent = existing->id;
                } else {
                  // Check if this looks like a GenScopeArray parent.
                  auto bracket = part.find('[');
                  if (bracket != llvm::StringRef::npos) {
                    // Ensure array parent exists.
                    llvm::StringRef arrayName = part.substr(0, bracket);
                    std::string arrayFull =
                        accumulated.substr(
                            0, accumulated.size() - part.size()) +
                        arrayName.str();
                    auto *arrObj =
                        vpiRuntime.findByName(arrayFull);
                    uint32_t arrId = curParent;
                    if (!arrObj) {
                      arrId = vpiRuntime.registerModule(
                          arrayName.str(), arrayFull, curParent);
                      auto *a = vpiRuntime.findById(arrId);
                      if (a)
                        a->type = VPIObjectType::GenScopeArray;
                    } else {
                      arrId = arrObj->id;
                    }
                    curParent = vpiRuntime.registerModule(
                        part.str(), accumulated, arrId);
                    auto *s = vpiRuntime.findById(curParent);
                    if (s)
                      s->type = VPIObjectType::GenScope;
                  } else {
                    curParent = vpiRuntime.registerModule(
                        part.str(), accumulated, curParent);
                    auto *s = vpiRuntime.findById(curParent);
                    if (s)
                      s->type = VPIObjectType::GenScope;
                  }
                }
              }
              parentScopeId = curParent;
            } else {
              parentScopeId = parentObj->id;
            }
          }

          llvm::StringRef shortName = qualParamName;
          if (lastDot != llvm::StringRef::npos)
            shortName = qualParamName.substr(lastDot + 1);

          vpiRuntime.registerParameter(shortName.str(), fullParamName,
                                       paramValue, paramWidth,
                                       parentScopeId);
        }
      }

      // Register gate primitive instances from vpi.gate_instances.
      // These appear as named hierarchy objects accessible via
      // vpi_handle_by_name.
      if (auto gateInsts = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.gate_instances")) {
        for (auto namedAttr : gateInsts) {
          llvm::StringRef gateName = namedAttr.getName().getValue();
          std::string qualifiedName = modName + "." + gateName.str();
          if (vpiRuntime.findByName(qualifiedName))
            continue;
          // Register as a GenScope so it appears in hierarchy iteration.
          uint32_t gateId = vpiRuntime.registerModule(
              gateName.str(), qualifiedName, modId);
          auto *gateObj = vpiRuntime.findById(gateId);
          if (gateObj)
            gateObj->type = VPIObjectType::GenScope;
        }
      }

      // Register interface instance arrays from vpi.interface_arrays.
      // cocotb expects these as GenScopeArray + GenScope children,
      // similar to generate arrays.
      if (auto ifaceArrays = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.interface_arrays")) {
        for (auto namedAttr : ifaceArrays) {
          llvm::StringRef arrName = namedAttr.getName().getValue();
          std::string qualifiedName = modName + "." + arrName.str();
          if (vpiRuntime.findByName(qualifiedName))
            continue;
          auto dictAttr =
              dyn_cast<mlir::DictionaryAttr>(namedAttr.getValue());
          if (!dictAttr)
            continue;
          int32_t left = 0, right = 0;
          if (auto leftAttr = dictAttr.getAs<mlir::IntegerAttr>("left"))
            left = leftAttr.getInt();
          if (auto rightAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("right"))
            right = rightAttr.getInt();

          uint32_t arrayId = vpiRuntime.registerModule(
              arrName.str(), qualifiedName, modId);
          auto *arrayObj = vpiRuntime.findById(arrayId);
          if (arrayObj) {
            arrayObj->type = VPIObjectType::GenScopeArray;
            arrayObj->leftBound = left;
            arrayObj->rightBound = right;
          }
          // Create GenScope children for each index.
          int32_t lo = std::min(left, right);
          int32_t hi = std::max(left, right);
          for (int32_t idx = lo; idx <= hi; ++idx) {
            std::string childName =
                arrName.str() + "[" + std::to_string(idx) + "]";
            std::string childFull = modName + "." + childName;
            if (!vpiRuntime.findByName(childFull)) {
              uint32_t childId = vpiRuntime.registerModule(
                  childName, childFull, arrayId);
              auto *childObj = vpiRuntime.findById(childId);
              if (childObj)
                childObj->type = VPIObjectType::GenScope;
            }
          }
        }
      }

      // Read interface definitions (signal names/widths) for creating
      // synthetic signal children under interface scope objects.
      mlir::DictionaryAttr ifaceDefsAttr =
          hwMod->getAttrOfType<mlir::DictionaryAttr>("vpi.interface_defs");

      // Helper: register synthetic signal children for an interface scope.
      auto registerIfaceSignals = [&](uint32_t scopeId,
                                      const std::string &scopeFull,
                                      llvm::StringRef defName) {
        if (!ifaceDefsAttr)
          return;
        auto defAttr = ifaceDefsAttr.getAs<mlir::DictionaryAttr>(defName);
        if (!defAttr)
          return;
        for (auto sigAttr : defAttr) {
          llvm::StringRef sigName = sigAttr.getName().getValue();
          uint32_t sigWidth = 1;
          if (auto intAttr =
                  dyn_cast<mlir::IntegerAttr>(sigAttr.getValue()))
            sigWidth = intAttr.getInt();
          std::string sigFull = scopeFull + "." + sigName.str();
          if (!vpiRuntime.findByName(sigFull)) {
            // Register as a zero-backed signal (no simulation backing).
            vpiRuntime.registerSignal(
                sigName.str(), sigFull, /*signalId=*/0, sigWidth,
                VPIObjectType::Net, scopeId);
          }
        }
      };

      // Register single interface instances.
      if (auto ifaceInsts = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.interface_instances")) {
        for (auto namedAttr : ifaceInsts) {
          llvm::StringRef instName = namedAttr.getName().getValue();
          std::string qualifiedName = modName + "." + instName.str();
          if (vpiRuntime.findByName(qualifiedName))
            continue;
          uint32_t scopeId = vpiRuntime.registerModule(
              instName.str(), qualifiedName, modId);
          auto *scopeObj = vpiRuntime.findById(scopeId);
          if (scopeObj)
            scopeObj->type = VPIObjectType::GenScope;

          // Register signal children from interface definition.
          if (auto defStr =
                  dyn_cast<mlir::StringAttr>(namedAttr.getValue()))
            registerIfaceSignals(scopeId, qualifiedName,
                                 defStr.getValue());
        }
      }

      // Add signal children to interface array elements.
      if (auto ifaceArrays = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.interface_arrays")) {
        for (auto namedAttr : ifaceArrays) {
          auto dictAttr =
              dyn_cast<mlir::DictionaryAttr>(namedAttr.getValue());
          if (!dictAttr)
            continue;
          llvm::StringRef arrName = namedAttr.getName().getValue();
          llvm::StringRef defName;
          if (auto defAttr = dictAttr.getAs<mlir::StringAttr>("def"))
            defName = defAttr.getValue();
          if (defName.empty())
            continue;
          int32_t left = 0, right = 0;
          if (auto leftAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("left"))
            left = leftAttr.getInt();
          if (auto rightAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("right"))
            right = rightAttr.getInt();
          int32_t lo = std::min(left, right);
          int32_t hi = std::max(left, right);
          for (int32_t idx = lo; idx <= hi; ++idx) {
            std::string childFull = modName + "." + arrName.str() +
                                    "[" + std::to_string(idx) + "]";
            auto *childObj = vpiRuntime.findByName(childFull);
            if (childObj)
              registerIfaceSignals(childObj->id, childFull, defName);
          }
        }
      }

      // Register VPI packages from vpi.packages attribute.
      // Packages are top-level objects accessible via
      // vpi_iterate(vpiPackage, NULL) and their members via
      // vpi_handle_by_name with :: separator.
      if (auto pkgsAttr = hwMod->getAttrOfType<mlir::DictionaryAttr>(
              "vpi.packages")) {
        for (auto pkgAttr : pkgsAttr) {
          llvm::StringRef pkgName = pkgAttr.getName().getValue();
          auto membersDict =
              dyn_cast<mlir::DictionaryAttr>(pkgAttr.getValue());
          if (!membersDict)
            continue;
          // Skip if already registered.
          if (vpiRuntime.findByName(pkgName.str()))
            continue;
          // Register the package object (no parent).
          uint32_t pkgId = vpiRuntime.registerModule(
              pkgName.str(), pkgName.str(), /*parentId=*/0);
          auto *pkgObj = vpiRuntime.findById(pkgId);
          if (pkgObj)
            pkgObj->type = VPIObjectType::Package;

          // Register each member as a Net (LogicArrayObject in cocotb).
          // Store the constant value as a hex string for getValue.
          for (auto memberAttr : membersDict) {
            llvm::StringRef memberName =
                memberAttr.getName().getValue();
            // Use :: separator for full path.
            std::string memberFull =
                pkgName.str() + "::" + memberName.str();
            // Also register with . separator for handleByName.
            std::string memberDot =
                pkgName.str() + "." + memberName.str();

            uint32_t width = 32;
            std::string hexValue = "0";
            if (auto intAttr = dyn_cast<mlir::IntegerAttr>(
                    memberAttr.getValue())) {
              width = intAttr.getValue().getBitWidth();
              // Store unsigned hex representation.
              llvm::SmallString<128> hexStr;
              intAttr.getValue().toStringUnsigned(hexStr, 16);
              hexValue = std::string(hexStr.begin(), hexStr.end());
            }

            // Register as Net so cocotb creates LogicArrayObject.
            uint32_t sigId = vpiRuntime.registerSignal(
                memberName.str(), memberFull, /*signalId=*/0,
                width, VPIObjectType::Net, pkgId);
            auto *sigObj = vpiRuntime.findById(sigId);
            if (sigObj) {
              // Store hex value in stringValue for getValue.
              sigObj->stringValue = hexValue;
              // Package members always have explicit type ranges, so cocotb
              // creates LogicArrayObject (not LogicObject) even for 1-bit.
              sigObj->hasExplicitRange = true;
            }
            // Also add . mapping for handleByName lookup.
            vpiRuntime.addNameMapping(memberDot, sigId);
          }
        }
      }
    }

    vpiRuntime.installDispatchTable();
    if (!vpiRuntime.loadVPILibrary(vpiLibrary)) {
      llvm::errs() << "[circt-sim] Failed to load VPI library: " << vpiLibrary
                   << "\n";
      return failure();
    }
    vpiRuntime.fireStartOfSimulation();

    // Set up the post-callback hook to propagate firreg changes to hw.output
    // port signals. This runs AFTER all VPI callbacks for each cbAfterDelay,
    // ensuring cocotb reads old port values during RisingEdge handlers before
    // ports are updated with newly captured firreg values.
    //
    // Keep this opt-in because enabling it globally can perturb existing
    // non-cocotb scheduling behavior.
    bool enablePostCallbackHwOutputHook =
        std::getenv("CIRCT_SIM_ENABLE_POST_CALLBACK_HW_OUTPUT_HOOK") != nullptr;
    if (enablePostCallbackHwOutputHook && !hwOutputProcessIds.empty()) {
      vpiRuntime.setPostCallbackHook([this]() {
        static bool traceHook = std::getenv("CIRCT_TRACE_POST_HOOK");
        if (traceHook) {
          SimTime now = scheduler.getCurrentTime();
          llvm::errs() << "[postCallbackHook] t=" << now.realTime
                       << " re-eval " << hwOutputProcessIds.size()
                       << " hw.output processes\n";
        }
        for (ProcessId procId : hwOutputProcessIds)
          scheduler.scheduleProcess(procId, SchedulingRegion::Active);
        scheduler.executeCurrentTime();
      });
    }
  }

  llvm::outs() << "[circt-sim] Starting simulation\n";
  llvm::outs().flush();

  // Main simulation loop
  uint64_t loopIterations = 0;
  auto startWallTime = std::chrono::steady_clock::now();

  // Track consecutive zero-delta iterations at the same time for loop detection
  uint64_t lastZeroIterTime = std::numeric_limits<uint64_t>::max();
  uint64_t lastDeltaTime = std::numeric_limits<uint64_t>::max();
  uint64_t zeroIterationsAtSameTime = 0;
  uint64_t deltaCyclesAtSameTime = 0;
  constexpr uint64_t maxZeroIterations = 1000;

  while (control.shouldContinue()) {
    if (interruptRequested.load())
      requestAbort("Interrupt signal received");
    if (abortRequested.load()) {
      handleAbort();
      break;
    }
    ++loopIterations;

    // Check wall-clock timeout
    if (timeout > 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - startWallTime);
      if (static_cast<uint64_t>(elapsed.count()) >= timeout) {
        requestAbort("Wall-clock timeout reached");
        handleAbort();
        break;
      }
    }

    // Check simulation time limit
    const auto &currentTime = scheduler.getCurrentTime();
    if (maxTime > 0 && currentTime.realTime >= maxTime) {
      llvm::errs() << "[circt-sim] Main loop exit: maxTime reached ("
                   << currentTime.realTime << " >= " << maxTime
                   << " fs), iter=" << loopIterations << "\n";
      break;
    }

    if (llhdInterpreter)
      llhdInterpreter->pollRegisteredMonitor();

    // Execute delta cycles
    size_t deltasExecuted;
    static bool traceExec =
        std::getenv("CIRCT_SIM_TRACE_VPI_TIMING") != nullptr;
    auto execStart = std::chrono::steady_clock::now();
    if (parallelScheduler) {
      deltasExecuted = parallelScheduler->executeCurrentTimeParallel();
    } else {
      deltasExecuted = scheduler.executeCurrentTime();
    }
    if (traceExec) {
      auto execEnd = std::chrono::steady_clock::now();
      auto execUs = std::chrono::duration_cast<
          std::chrono::microseconds>(execEnd - execStart).count();
      if (execUs > 100000 || loopIterations <= 5) { // > 100ms or first 5
        llvm::errs() << "[MAIN-EXEC] iter=" << loopIterations
                     << " execTime=" << execUs << "us deltas="
                     << deltasExecuted << " t="
                     << currentTime.realTime << "\n";
      }
    }

    LLVM_DEBUG({
      static uint64_t lastDiagTime = UINT64_MAX;
      bool timeChanged = (currentTime.realTime != lastDiagTime);
      if (timeChanged || loopIterations <= 5 ||
          (loopIterations % 500000 == 0)) {
        llvm::dbgs() << "[MAINLOOP] iter=" << loopIterations
                     << " time=" << currentTime.realTime
                     << " deltas=" << deltasExecuted
                     << " ready=" << scheduler.hasReadyProcesses()
                     << "\n";
        lastDiagTime = currentTime.realTime;
      }
    });

    // Fire VPI scheduling-region callbacks after each iteration.
    // cocotb defers signal writes (vpi_put_value) to the ReadWriteSynch region,
    // so we must fire these callbacks every iteration — not only when
    // deltasExecuted > 0 — to ensure deferred writes are flushed.
    if (!vpiLibrary.empty()) {
      auto &vpi = VPIRuntime::getInstance();
      static bool traceMainVpi =
          std::getenv("CIRCT_SIM_TRACE_VPI_TIMING") != nullptr;
      auto mRwStart = std::chrono::steady_clock::now();
      int mRwCount = 0;
      for (int rwIter = 0; rwIter < 100; ++rwIter) {
        vpi.fireCallbacks(cbReadWriteSynch);
        ++mRwCount;
        if (!vpi.hasActiveCallbacks(cbReadWriteSynch))
          break;
      }
      auto mRwEnd = std::chrono::steady_clock::now();
      vpi.fireCallbacks(cbReadOnlySynch);
      auto mRoEnd = std::chrono::steady_clock::now();
      if (traceMainVpi) {
        static uint64_t mainVpiCount = 0;
        ++mainVpiCount;
        auto rwUs = std::chrono::duration_cast<
            std::chrono::microseconds>(mRwEnd - mRwStart).count();
        auto roUs = std::chrono::duration_cast<
            std::chrono::microseconds>(mRoEnd - mRwEnd).count();
        if (mainVpiCount <= 20 || rwUs + roUs > 1000) {
          llvm::errs() << "[MAIN-VPI] #" << mainVpiCount
                       << " rw=" << rwUs << "us(x" << mRwCount
                       << ") ro=" << roUs << "us t="
                       << scheduler.getCurrentTime().realTime
                       << " deltas=" << deltasExecuted << "\n";
        }
      }
    }

    if (llhdInterpreter)
      llhdInterpreter->pollRegisteredMonitor();

    // Check if simulation was terminated during execution (e.g., $finish called).
    // This must be checked immediately after executeCurrentTime() to ensure
    // termination is honored before any further processing.
    if (!control.shouldContinue()) {
      llvm::errs() << "[circt-sim] Main loop exit: shouldContinue()=false at time "
                   << currentTime.realTime << " fs, iter=" << loopIterations
                   << ", deltas=" << deltasExecuted << "\n";
      break;
    }

    // Check $finish grace period (wall-clock). This handles the case where
    // UVM runs entirely at simulation time 0 in delta cycles - advanceTime()
    // is never called, so the post-advanceTime check wouldn't fire.
    if (llhdInterpreter && llhdInterpreter->checkFinishGracePeriod()) {
      llvm::errs()
          << "[circt-sim] Main loop exit: $finish grace period expired at time "
          << scheduler.getCurrentTime().realTime
          << " fs, iter=" << loopIterations << "\n";
      control.finish(0);
      break;
    }

    bool hasReadyProcesses = scheduler.hasReadyProcesses();

    if (deltasExecuted == 0) {
      // Track zero-delta iterations at the same time for loop detection
      if (currentTime.realTime == lastZeroIterTime) {
        ++zeroIterationsAtSameTime;
        if (zeroIterationsAtSameTime >= maxZeroIterations) {
          // Print diagnostic information
          llvm::errs() << "[circt-sim] WARNING: Possible infinite loop detected!\n";
          llvm::errs() << "[circt-sim] " << maxZeroIterations
                       << " iterations with 0 delta cycles at time "
                       << currentTime.realTime << " fs\n";

          // Get scheduler statistics
          const auto &stats = scheduler.getStatistics();
          llvm::errs() << "[circt-sim] Processes registered: "
                       << stats.processesRegistered << "\n";
          llvm::errs() << "[circt-sim] Processes executed: "
                       << stats.processesExecuted << "\n";
          llvm::errs() << "[circt-sim] Total delta cycles: "
                       << stats.deltaCyclesExecuted << "\n";

          control.error("ZERO_DELTA_LOOP",
                        "Too many zero-delta iterations at same time - "
                        "possible infinite loop with no progress");
          if (llhdInterpreter)
            llhdInterpreter->dumpProcessStates(llvm::errs());
          dumpLastDeltaSignals(llvm::errs());
          dumpLastDeltaProcesses(llvm::errs());
          control.finish(1);
          break;
        }
      } else {
        lastZeroIterTime = currentTime.realTime;
        zeroIterationsAtSameTime = 0;
      }
    } else {
      if (currentTime.realTime == lastDeltaTime) {
        deltaCyclesAtSameTime += deltasExecuted;
      } else {
        deltaCyclesAtSameTime = deltasExecuted;
      }

      if (deltaCyclesAtSameTime > maxDeltas) {
        control.error("DELTA_OVERFLOW",
                      "Too many delta cycles at same time - possible infinite loop");
        if (llhdInterpreter)
          llhdInterpreter->dumpProcessStates(llvm::errs());
        dumpLastDeltaSignals(llvm::errs());
        dumpLastDeltaProcesses(llvm::errs());
        control.finish(1);
        break;
      }

      // Reset counter when we actually execute deltas
      zeroIterationsAtSameTime = 0;
      lastZeroIterTime = currentTime.realTime;
      lastDeltaTime = currentTime.realTime;
    }

    if (!hasReadyProcesses) {
      LLVM_DEBUG({
        static uint64_t lastAdvDiagTime = UINT64_MAX;
        if (currentTime.realTime != lastAdvDiagTime ||
            loopIterations <= 5) {
          llvm::dbgs() << "[MAINLOOP] advanceTime at time="
                       << currentTime.realTime
                       << " iter=" << loopIterations << "\n";
          lastAdvDiagTime = currentTime.realTime;
        }
      });

      // Check if simulation should stop before advancing time.
      // This ensures we report the correct termination time when $finish
      // is called (rather than advancing to the next scheduled event first).
      if (!control.shouldContinue())
        break;

      // No more events at current time, advance to next event
      uint64_t preAdvTime = currentTime.realTime;
      if (!scheduler.advanceTime()) {
        LLVM_DEBUG(llvm::dbgs() << "[MAINLOOP] advanceTime returned FALSE at time="
                     << currentTime.realTime << " iter=" << loopIterations
                     << "\n");
        // Always print why the sim stopped — this is critical diagnostic info.
        llvm::errs() << "[circt-sim] advanceTime() returned false at time "
                     << currentTime.realTime << " fs, iter=" << loopIterations
                     << " — no more scheduled events\n";
        const auto &stats = scheduler.getStatistics();
        llvm::errs() << "[circt-sim] Processes registered: "
                     << stats.processesRegistered
                     << ", executed: " << stats.processesExecuted
                     << ", delta cycles: " << stats.deltaCyclesExecuted << "\n";
        // Dump process states to show what's alive/dead/waiting
        if (llhdInterpreter)
          llhdInterpreter->dumpProcessStates(llvm::errs());
        // No more events
        break;
      }

      LLVM_DEBUG({
        auto newTime = scheduler.getCurrentTime().realTime;
        if (newTime != preAdvTime) {
          llvm::dbgs() << "[MAINLOOP] advanceTime: " << preAdvTime
                       << " -> " << newTime
                       << " iter=" << loopIterations << "\n";
        }
      });

      // Fire VPI cbNextSimTime callbacks when simulation time advances.
      // cocotb's NextTimeStep trigger registers a cbNextSimTime callback.
      if (!vpiLibrary.empty() &&
          scheduler.getCurrentTime().realTime != preAdvTime) {
        VPIRuntime::getInstance().fireCallbacks(cbNextSimTime);
      }

      // Check if the $finish grace period has expired. When UVM calls
      // $finish(success) with active forked children (phase hopper), we
      // allow a wall-clock grace period for cleanup phases to run before
      // forcing exit.
      if (llhdInterpreter && llhdInterpreter->checkFinishGracePeriod()) {
        if (verbosity >= 1) {
          llvm::errs()
              << "[circt-sim] $finish grace period expired at time "
              << scheduler.getCurrentTime().realTime
              << " fs - forcing termination\n";
        }
        control.finish(0);
        break;
      }
    }

    // Check for excessive delta cycles (infinite loop detection)
    if (deltasExecuted > maxDeltas) {
      control.error("DELTA_OVERFLOW",
                    "Too many delta cycles - possible infinite loop");
      if (llhdInterpreter)
        llhdInterpreter->dumpProcessStates(llvm::errs());
      dumpLastDeltaSignals(llvm::errs());
      dumpLastDeltaProcesses(llvm::errs());
      control.finish(1);
      break;
    }

    // Update control time for message timestamps
    control.setCurrentTime(currentTime);
  }

  // End profiling
  if (profiler) {
    auto simulationEndTime = std::chrono::high_resolution_clock::now();
    auto duration = simulationEndTime - simulationStartTime;
    profiler->endOperation(ProfileCategory::Custom, duration, "simulation_main");
  }

  // Close VCD file
  if (vcdWriter) {
    vcdWriter->close();
    llvm::outs() << "[circt-sim] Wrote waveform to " << vcdFile << "\n";
  }

  // Fire VPI end-of-simulation callback.
  if (!vpiLibrary.empty())
    VPIRuntime::getInstance().fireEndOfSimulation();

  // Report completion
  const auto &finalTime = scheduler.getCurrentTime();
  llvm::outs() << "[circt-sim] Simulation completed at time "
               << finalTime.realTime << " fs\n";

  stopWatchdogThread();

  return success();
}

void SimulationContext::printStatistics(llvm::raw_ostream &os) const {
  os << "\n=== Simulation Statistics ===\n";

  const auto &schedStats = scheduler.getStatistics();
  os << "Processes registered: " << schedStats.processesRegistered << "\n";
  os << "Processes executed:   " << schedStats.processesExecuted << "\n";
  os << "Delta cycles:         " << schedStats.deltaCyclesExecuted << "\n";
  os << "Signal updates:       " << schedStats.signalUpdates << "\n";
  os << "Edges detected:       " << schedStats.edgesDetected << "\n";

  const auto &ctrlStats = control.getStatistics();
  os << "Messages reported:    " << ctrlStats.messagesReported << "\n";
  os << "Errors:               " << control.getErrorCount() << "\n";
  os << "Warnings:             " << control.getWarningCount() << "\n";

  if (parallelScheduler) {
    os << "\n--- Parallel Statistics ---\n";
    parallelScheduler->printStatistics(os);
  }

  if (printOpStats && llhdInterpreter) {
    llhdInterpreter->dumpOpStats(os, opStatsTop);
  }

  if (printProcessStats && llhdInterpreter) {
    llhdInterpreter->dumpProcessStats(os, processStatsTop);
  }

  if (profiler) {
    os << "\n--- Profiling Statistics ---\n";
    // profiler->printReport(os);
  }

  os << "=============================\n";
}

static std::string resolveJitReportPath() {
  if (!jitReportPath.empty())
    return jitReportPath;
  if (const char *envPath = std::getenv("CIRCT_SIM_JIT_REPORT_PATH"))
    return std::string(envPath);
  return {};
}

static uint64_t resolveUint64OptionOrEnv(const llvm::cl::opt<uint64_t> &option,
                                         const char *envName) {
  if (option.getNumOccurrences() > 0)
    return option;
  if (const char *env = std::getenv(envName)) {
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == 0 && end != env)
      return static_cast<uint64_t>(parsed);
    llvm::errs() << "[circt-sim] Warning: ignoring invalid " << envName
                 << "='" << env << "'\n";
  }
  return option;
}

static int64_t resolveInt64OptionOrEnv(const llvm::cl::opt<int64_t> &option,
                                       const char *envName) {
  if (option.getNumOccurrences() > 0)
    return option;
  if (const char *env = std::getenv(envName)) {
    char *end = nullptr;
    errno = 0;
    long long parsed = std::strtoll(env, &end, 10);
    if (errno == 0 && end != env)
      return static_cast<int64_t>(parsed);
    llvm::errs() << "[circt-sim] Warning: ignoring invalid " << envName
                 << "='" << env << "'\n";
  }
  return option;
}

static bool resolveBoolOptionOrEnv(const llvm::cl::opt<bool> &option,
                                   const char *envName) {
  if (option.getNumOccurrences() > 0)
    return option;
  if (const char *env = std::getenv(envName)) {
    char c = env[0];
    if (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T')
      return true;
    if (c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F')
      return false;
    llvm::errs() << "[circt-sim] Warning: ignoring invalid " << envName
                 << "='" << env << "'\n";
  }
  return option;
}

static std::string resolveJitCachePolicy() {
  std::string policy = jitCachePolicy;
  llvm::StringRef source = "--jit-cache-policy";
  if (jitCachePolicy.getNumOccurrences() == 0) {
    if (const char *env = std::getenv("CIRCT_SIM_JIT_CACHE_POLICY")) {
      policy = env;
      source = "CIRCT_SIM_JIT_CACHE_POLICY";
    }
  }

  std::string normalized = llvm::StringRef(policy).trim().lower();
  if (normalized.empty() || normalized == "memory")
    return "memory";
  if (normalized == "none")
    return "none";

  llvm::errs() << "[circt-sim] Warning: invalid " << source << "='" << policy
               << "' (expected 'memory' or 'none'); using 'memory'\n";
  return "memory";
}

static JITCompileManager::Config resolveJitCompileManagerConfig() {
  JITCompileManager::Config config;
  config.hotThreshold =
      resolveUint64OptionOrEnv(jitHotThreshold, "CIRCT_SIM_JIT_HOT_THRESHOLD");
  config.compileBudget = resolveInt64OptionOrEnv(
      jitCompileBudget, "CIRCT_SIM_JIT_COMPILE_BUDGET");
  config.cachePolicy = resolveJitCachePolicy();
  config.failOnDeopt = resolveBoolOptionOrEnv(
      jitFailOnDeopt, "CIRCT_SIM_JIT_FAIL_ON_DEOPT");
  return config;
}

static LogicalResult emitJitReport(const SimulationContext &simContext,
                                   const JITCompileManager &jitCompileManager,
                                   uint64_t runWallMs, uint64_t totalWallMs) {
  std::string reportPath = resolveJitReportPath();
  if (reportPath.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream os(reportPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "[circt-sim] Failed to open JIT report file '"
                 << reportPath << "': " << ec.message() << "\n";
    return failure();
  }

  const auto &schedStats = simContext.getSchedulerStats();
  const auto &ctrlStats = simContext.getControlStats();
  const auto &jitConfig = jitCompileManager.getConfig();
  const auto &jitStats = jitCompileManager.getStatistics();
  const LLHDProcessInterpreter *interpreter = simContext.getInterpreter();

  uint64_t uvmFastPathHitsTotal = 0;
  uint64_t uvmFastPathActionKeysTotal = 0;
  uint64_t uvmJitPromotedActionsTotal = 0;
  uint64_t uvmJitHotThreshold = 0;
  int64_t uvmJitPromotionBudgetRemaining = 0;
  struct JitDeoptProcessEntry {
    uint64_t processId;
    std::string processName;
    std::string reason;
    std::string detail;
  };
  llvm::SmallVector<JitDeoptProcessEntry, 8> jitDeoptProcesses;
  std::vector<LLHDProcessInterpreter::JitRuntimeIndirectSiteProfile>
      jitRuntimeIndirectSites;
  uint64_t jitRuntimeIndirectCallsTotal = 0;
  uint64_t jitRuntimeIndirectUnresolvedTotal = 0;
  if (interpreter) {
    const auto &deoptDetails = interpreter->getJitDeoptDetailByProcess();
    uvmFastPathHitsTotal = interpreter->getUvmFastPathHitsTotal();
    uvmFastPathActionKeysTotal = interpreter->getUvmFastPathActionKeyCount();
    uvmJitPromotedActionsTotal = interpreter->getUvmJitPromotedActionCount();
    uvmJitHotThreshold = interpreter->getUvmJitHotThreshold();
    uvmJitPromotionBudgetRemaining =
        interpreter->getUvmJitPromotionBudgetRemaining();
    jitRuntimeIndirectSites = interpreter->getJitRuntimeIndirectSiteProfiles();
    for (const auto &site : jitRuntimeIndirectSites) {
      jitRuntimeIndirectCallsTotal += site.callsTotal;
      jitRuntimeIndirectUnresolvedTotal += site.unresolvedCalls;
    }
    for (const auto &entry : interpreter->getJitDeoptReasonByProcess()) {
      std::string detail;
      auto detailIt = deoptDetails.find(entry.first);
      if (detailIt != deoptDetails.end())
        detail = detailIt->second;
      jitDeoptProcesses.push_back({entry.first,
                                   interpreter->getJitDeoptProcessName(
                                       entry.first),
                                   entry.second, std::move(detail)});
    }
    llvm::sort(jitDeoptProcesses, [](const auto &lhs, const auto &rhs) {
      return lhs.processId < rhs.processId;
    });
  }

  auto jos = llvm::json::OStream(os, 2);
  jos.object([&] {
    jos.attribute("schema_version", 1);
    jos.attribute("mode", getRunModeName(runMode));
    jos.attribute("exit_code", simContext.getExitCode());
    jos.attribute("final_time_fs", simContext.getFinalTime().realTime);
    jos.attribute("run_wall_ms", runWallMs);
    jos.attribute("total_wall_ms", totalWallMs);

    jos.attributeObject("jit_config", [&] {
      jos.attribute("hot_threshold", jitConfig.hotThreshold);
      jos.attribute("compile_budget", jitConfig.compileBudget);
      jos.attribute("cache_policy", jitConfig.cachePolicy);
      jos.attribute("fail_on_deopt", jitConfig.failOnDeopt ? 1 : 0);
    });

    jos.attributeObject("scheduler", [&] {
      jos.attribute("processes_registered", schedStats.processesRegistered);
      jos.attribute("processes_executed", schedStats.processesExecuted);
      jos.attribute("delta_cycles_executed", schedStats.deltaCyclesExecuted);
      jos.attribute("signal_updates", schedStats.signalUpdates);
      jos.attribute("edges_detected", schedStats.edgesDetected);
      jos.attribute("max_delta_cycles_reached", schedStats.maxDeltaCyclesReached);
    });

    jos.attributeObject("control", [&] {
      jos.attribute("messages_reported", ctrlStats.messagesReported);
      jos.attribute("messages_filtered", ctrlStats.messagesFiltered);
      jos.attribute("finish_calls", ctrlStats.finishCalls);
      jos.attribute("stop_calls", ctrlStats.stopCalls);
      jos.attribute("errors", simContext.getErrorCount());
      jos.attribute("warnings", simContext.getWarningCount());
    });

    jos.attributeObject("jit", [&] {
      jos.attribute("jit_compiles_total", jitStats.jitCompilesTotal);
      jos.attribute("jit_cache_hits_total", jitStats.jitCacheHitsTotal);
      jos.attribute("jit_exec_hits_total", jitStats.jitExecHitsTotal);
      jos.attribute("jit_deopts_total", jitStats.jitDeoptsTotal);
      jos.attribute("jit_deopt_reason_unknown", jitStats.jitDeoptReasonUnknown);
      jos.attribute("jit_deopt_reason_interpreter_fallback",
                    jitStats.jitDeoptReasonInterpreterFallback);
      jos.attribute("jit_deopt_reason_guard_failed",
                    jitStats.jitDeoptReasonGuardFailed);
      jos.attribute("jit_deopt_reason_unsupported_operation",
                    jitStats.jitDeoptReasonUnsupportedOperation);
      jos.attribute("jit_deopt_reason_missing_thunk",
                    jitStats.jitDeoptReasonMissingThunk);
      jos.attribute("jit_compile_wall_ms", jitStats.jitCompileWallMs);
      jos.attribute("jit_exec_wall_ms", jitStats.jitExecWallMs);
      jos.attribute("jit_strict_violations_total",
                    jitStats.jitStrictViolationsTotal);
      jos.attribute("jit_call_indirect_sites_total",
                    static_cast<uint64_t>(jitRuntimeIndirectSites.size()));
      jos.attribute("jit_call_indirect_calls_total",
                    jitRuntimeIndirectCallsTotal);
      jos.attribute("jit_call_indirect_unresolved_total",
                    jitRuntimeIndirectUnresolvedTotal);
      jos.attributeArray("jit_deopt_processes", [&] {
        for (const auto &entry : jitDeoptProcesses) {
          jos.object([&] {
            jos.attribute("process_id", entry.processId);
            jos.attribute("process_name", entry.processName);
            jos.attribute("reason", entry.reason);
            if (!entry.detail.empty())
              jos.attribute("detail", entry.detail);
          });
        }
      });
      jos.attributeArray("jit_call_indirect_sites", [&] {
        for (const auto &site : jitRuntimeIndirectSites) {
          jos.object([&] {
            jos.attribute("site_id", site.siteId);
            jos.attribute("owner", site.owner);
            jos.attribute("location", site.location);
            jos.attribute("calls_total", site.callsTotal);
            jos.attribute("unresolved_calls", site.unresolvedCalls);
            jos.attribute("targets_total",
                          static_cast<uint64_t>(site.targets.size()));
            jos.attribute("target_set_version", site.targetSetVersion);
            jos.attribute(
                "target_set_hash",
                (Twine("0x") + llvm::utohexstr(site.targetSetHash)).str());
            jos.attributeArray("targets", [&] {
              for (const auto &target : site.targets) {
                jos.object([&] {
                  jos.attribute("target_name", target.targetName);
                  jos.attribute("calls", target.calls);
                });
              }
            });
          });
        }
      });
    });

    jos.attributeObject("uvm_fast_path", [&] {
      jos.attribute("hits_total", uvmFastPathHitsTotal);
      jos.attribute("action_keys_total", uvmFastPathActionKeysTotal);
      jos.attribute("jit_promoted_actions_total", uvmJitPromotedActionsTotal);
      jos.attribute("jit_hot_threshold", uvmJitHotThreshold);
      jos.attribute("jit_promotion_budget_remaining",
                    uvmJitPromotionBudgetRemaining);
    });
  });

  os.flush();
  return success();
}

//===----------------------------------------------------------------------===//
// Main Processing Pipeline
//===----------------------------------------------------------------------===//

static LogicalResult processInput(MLIRContext &context,
                                   llvm::SourceMgr &sourceMgr) {
  auto startTime = std::chrono::steady_clock::now();
  JITCompileManager jitCompileManager(resolveJitCompileManagerConfig());
  auto lastStageTime = startTime;
  auto reportStage = [&lastStageTime, &startTime](llvm::StringRef stage) {
    auto now = std::chrono::steady_clock::now();
    if (verbosity >= 1) {
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                lastStageTime)
              .count();
      auto total =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime)
              .count();
      llvm::errs() << "[circt-sim] Stage: " << stage << " (prev: " << elapsed
                    << "ms, total: " << total << "ms)\n";
    }
    lastStageTime = now;
  };

  std::unique_ptr<WallClockTimeout> wallClockTimeout;
  if (timeout > 0) {
    wallClockTimeout = std::make_unique<WallClockTimeout>(
        std::chrono::seconds(timeout), []() {
          llvm::errs()
              << "[circt-sim] Wall-clock timeout reached (global guard)\n";
          interruptRequested.store(true);
          if (!simulationStarted.load())
            std::_Exit(1);
        });
  }

  reportStage("parse");
  // Parse the input file
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: Failed to parse input\n";
    return failure();
  }

  auto countRegionOps = [](mlir::Region &region) -> size_t {
    llvm::SmallVector<mlir::Region *, 16> regionWorklist;
    regionWorklist.push_back(&region);
    size_t count = 0;
    while (!regionWorklist.empty()) {
      mlir::Region *curRegion = regionWorklist.pop_back_val();
      for (mlir::Block &block : *curRegion) {
        for (mlir::Operation &op : block) {
          ++count;
          for (mlir::Region &nested : op.getRegions())
            regionWorklist.push_back(&nested);
        }
      }
    }
    return count;
  };

  auto countRegionOpNames = [](mlir::Region &region) {
    llvm::SmallVector<mlir::Region *, 16> regionWorklist;
    regionWorklist.push_back(&region);
    llvm::StringMap<size_t> counts;
    while (!regionWorklist.empty()) {
      mlir::Region *curRegion = regionWorklist.pop_back_val();
      for (mlir::Block &block : *curRegion) {
        for (mlir::Operation &op : block) {
          ++counts[op.getName().getStringRef()];
          for (mlir::Region &nested : op.getRegions())
            regionWorklist.push_back(&nested);
        }
      }
    }
    return counts;
  };

  auto countExtractBuckets = [](mlir::Region &region) {
    struct Buckets {
      size_t low1 = 0;
      size_t high1 = 0;
      size_t lowWide = 0;
      size_t highWide = 0;
    } buckets;

    llvm::SmallVector<mlir::Region *, 16> regionWorklist;
    regionWorklist.push_back(&region);
    while (!regionWorklist.empty()) {
      mlir::Region *curRegion = regionWorklist.pop_back_val();
      for (mlir::Block &block : *curRegion) {
        for (mlir::Operation &op : block) {
          if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
            auto intTy = dyn_cast<IntegerType>(extract.getType());
            if (!intTy)
              continue;
            unsigned width = intTy.getWidth();
            unsigned lowBit = extract.getLowBit();
            if (width == 1) {
              if (lowBit < 64)
                ++buckets.low1;
              else
                ++buckets.high1;
            } else {
              if (lowBit + width <= 64)
                ++buckets.lowWide;
              else
                ++buckets.highWide;
            }
          }
          for (mlir::Region &nested : op.getRegions())
            regionWorklist.push_back(&nested);
        }
      }
    }
    return buckets;
  };

  // Handle analyze mode - just print statistics and exit
  if (runMode == RunMode::Analyze) {
    llvm::outs() << "=== Design Analysis ===\n";

    size_t moduleCount = 0;
    size_t portCount = 0;
    size_t instanceCount = 0;

    module->walk([&](Operation *op) {
      if (auto hwModule = dyn_cast<hw::HWModuleOp>(op)) {
        moduleCount++;
        portCount += hwModule.getPortList().size();
      } else if (isa<hw::InstanceOp>(op)) {
        instanceCount++;
      }
    });

    llvm::outs() << "Modules:   " << moduleCount << "\n";
    llvm::outs() << "Ports:     " << portCount << "\n";
    llvm::outs() << "Instances: " << instanceCount << "\n";
    llvm::outs() << "========================\n";

    if (printProcessOpCounts) {
      struct ProcEntry {
        std::string kind;
        std::string loc;
        size_t opCount;
        Operation *op;
      };
      llvm::SmallVector<ProcEntry, 16> entries;
      module->walk([&](llhd::ProcessOp op) {
        std::string locStr;
        llvm::raw_string_ostream locStream(locStr);
        op.getLoc().print(locStream);
        entries.push_back(
            {"process", locStream.str(), countRegionOps(op.getBody()),
             op.getOperation()});
      });
      module->walk([&](seq::InitialOp op) {
        std::string locStr;
        llvm::raw_string_ostream locStream(locStr);
        op.getLoc().print(locStream);
        entries.push_back(
            {"initial", locStream.str(), countRegionOps(op.getBody()),
             op.getOperation()});
      });

      llvm::sort(entries, [](const ProcEntry &lhs, const ProcEntry &rhs) {
        if (lhs.opCount != rhs.opCount)
          return lhs.opCount > rhs.opCount;
        if (lhs.kind != rhs.kind)
          return lhs.kind < rhs.kind;
        return lhs.loc < rhs.loc;
      });

      llvm::outs() << "\n=== Process Op Counts (top "
                   << processOpCountsTop << ") ===\n";
      size_t limit = std::min<size_t>(processOpCountsTop, entries.size());
      for (size_t i = 0; i < limit; ++i) {
        llvm::outs() << entries[i].kind << " opCount=" << entries[i].opCount;
        if (!entries[i].loc.empty())
          llvm::outs() << " loc=" << entries[i].loc;
        llvm::outs() << "\n";
      }
      llvm::outs() << "===============================\n";

      if (printProcessOpDumps) {
        llvm::outs() << "\n=== Process Op Dumps (top " << processOpCountsTop
                     << ") ===\n";
        for (size_t i = 0; i < limit; ++i) {
          llvm::outs() << "--- " << entries[i].kind << " opCount="
                       << entries[i].opCount;
          if (!entries[i].loc.empty())
            llvm::outs() << " loc=" << entries[i].loc;
          llvm::outs() << " ---\n";
          if (entries[i].op)
            entries[i].op->print(llvm::outs());
          llvm::outs() << "\n";
        }
        llvm::outs() << "=============================\n";
      }

      if (printProcessOpBreakdown) {
        llvm::outs() << "\n=== Process Op Breakdown (top "
                     << processOpCountsTop << ") ===\n";
        for (size_t i = 0; i < limit; ++i) {
          llvm::outs() << "--- " << entries[i].kind << " opCount="
                       << entries[i].opCount;
          if (!entries[i].loc.empty())
            llvm::outs() << " loc=" << entries[i].loc;
          llvm::outs() << " ---\n";
          if (!entries[i].op)
            continue;
          llvm::StringMap<size_t> counts;
          if (auto proc = dyn_cast<llhd::ProcessOp>(entries[i].op))
            counts = countRegionOpNames(proc.getBody());
          else if (auto init = dyn_cast<seq::InitialOp>(entries[i].op))
            counts = countRegionOpNames(init.getBody());

          struct OpEntry {
            llvm::StringRef name;
            size_t count;
          };
          llvm::SmallVector<OpEntry, 16> opEntries;
          opEntries.reserve(counts.size());
          for (const auto &entry : counts)
            opEntries.push_back({entry.getKey(), entry.getValue()});
          llvm::sort(opEntries, [](const OpEntry &lhs, const OpEntry &rhs) {
            if (lhs.count != rhs.count)
              return lhs.count > rhs.count;
            return lhs.name < rhs.name;
          });

          size_t opLimit =
              std::min<size_t>(processOpBreakdownTop, opEntries.size());
          for (size_t idx = 0; idx < opLimit; ++idx) {
            llvm::outs() << opEntries[idx].name << ": " << opEntries[idx].count
                         << "\n";
          }
          if (printProcessOpExtractBreakdown) {
            decltype(countExtractBuckets(std::declval<mlir::Region &>()))
                buckets;
            if (auto proc = dyn_cast<llhd::ProcessOp>(entries[i].op))
              buckets = countExtractBuckets(proc.getBody());
            else if (auto init = dyn_cast<seq::InitialOp>(entries[i].op))
              buckets = countExtractBuckets(init.getBody());
            llvm::outs() << "comb.extract.low1<64: " << buckets.low1 << "\n";
            llvm::outs() << "comb.extract.high1>=64: " << buckets.high1 << "\n";
            llvm::outs() << "comb.extract.lowWide<64: " << buckets.lowWide
                         << "\n";
            llvm::outs() << "comb.extract.highWide>=64: " << buckets.highWide
                         << "\n";
          }
        }
        llvm::outs() << "===============================\n";
      }
    }

    return success();
  }

  reportStage("passes");
  if (!skipPasses) {
    // Fix loop partial-width NBA accumulation before running passes.
    // This corrects a compiler lowering issue where partial-width non-blocking
    // assignments in for-loops (e.g., reg[i*W +: W] <= data[i]) don't
    // accumulate correctly because the read-modify-write base is a stale
    // signal probe instead of the loop-carried accumulated value.
    int loopFixes = fixLoopPartialDriveAccumulation(module->getOperation());
    LLVM_DEBUG(if (loopFixes) llvm::dbgs()
               << "[circt-sim] Fixed " << loopFixes
               << " loop partial-drive accumulation issues\n");

    // Run preprocessing passes if needed
    PassManager pm(&context);
    pm.enableVerifier(verifyPasses);

    // Add passes to lower to simulation-friendly form.
    // Use the CIRCT bottom-up simple canonicalizer (region simplification
    // disabled, max 200k rewrites) to prevent OOM when canonicalization
    // patterns interact to cause exponential blowup on LLHD+comb IR.
    pm.addPass(circt::createBottomUpSimpleCanonicalizerPass());

    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass pipeline failed\n";
      return failure();
    }
  }

  // Create and initialize simulation context
  // Convert topModules from cl::list to SmallVector
  llvm::SmallVector<std::string, 4> tops;
  for (const auto &top : topModules) {
    tops.push_back(top);
  }

  reportStage("init");
  SimulationContext simContext;
  simContext.setJITCompileManager(&jitCompileManager);
  simContext.setJitRuntimeIndirectProfileEnabled(
      runMode == RunMode::Compile && !resolveJitReportPath().empty());
  simContext.setMaxDeltaCycles(maxDeltas);
  simContext.setMaxProcessSteps(maxProcessSteps);
  if (failed(simContext.initialize(*module, tops))) {
    return failure();
  }

  // Run the simulation
  reportStage("run");
  auto runStartTime = std::chrono::steady_clock::now();
  LogicalResult runResult = simContext.run();
  uint64_t runWallMs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - runStartTime)
          .count());
  if (runMode == RunMode::Compile)
    jitCompileManager.addExecWallMs(runWallMs);

  uint64_t totalWallMs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - startTime)
          .count());
  if (failed(runResult)) {
    if (failed(
            emitJitReport(simContext, jitCompileManager, runWallMs, totalWallMs)))
      return failure();
    return failure();
  }

  // Print statistics if requested
  if (printStats) {
    simContext.printStatistics(llvm::outs());
  }
  if (failed(
          emitJitReport(simContext, jitCompileManager, runWallMs, totalWallMs)))
    return failure();

  // Use _exit() here, before returning, to skip the expensive
  // SimulationContext destructor.  For UVM designs with millions of
  // operations, the destructor chain (DenseMap/StringMap/vector cleanup)
  // can take minutes and provides no user-visible benefit after a
  // successful simulation.  The _exit() must be here (not in main())
  // because SimulationContext is stack-allocated and its destructor
  // runs when this function returns.
  int exitCode = simContext.getExitCode();
  if (runMode == RunMode::Compile &&
      jitCompileManager.getStatistics().jitStrictViolationsTotal > 0) {
    llvm::errs() << "[circt-sim] Strict JIT policy violation: deopts_total="
                 << jitCompileManager.getStatistics().jitDeoptsTotal
                 << " (native-thunk coverage is not complete yet)\n";
    if (const LLHDProcessInterpreter *interpreter = simContext.getInterpreter()) {
      struct StrictDeoptProcessEntry {
        uint64_t processId;
        std::string processName;
        std::string reason;
        std::string detail;
      };
      llvm::SmallVector<StrictDeoptProcessEntry, 8> strictDeoptProcesses;
      const auto &deoptDetails = interpreter->getJitDeoptDetailByProcess();
      for (const auto &entry : interpreter->getJitDeoptReasonByProcess()) {
        std::string detail;
        auto detailIt = deoptDetails.find(entry.first);
        if (detailIt != deoptDetails.end())
          detail = detailIt->second;
        strictDeoptProcesses.push_back(
            {entry.first, interpreter->getJitDeoptProcessName(entry.first),
             entry.second, std::move(detail)});
      }
      llvm::sort(strictDeoptProcesses, [](const auto &lhs, const auto &rhs) {
        return lhs.processId < rhs.processId;
      });

      constexpr size_t kStrictDeoptLogLimit = 20;
      size_t emitCount = std::min(strictDeoptProcesses.size(),
                                  kStrictDeoptLogLimit);
      for (size_t i = 0; i < emitCount; ++i) {
        const auto &entry = strictDeoptProcesses[i];
        llvm::StringRef name =
            entry.processName.empty() ? llvm::StringRef("-")
                                      : llvm::StringRef(entry.processName);
        llvm::errs() << "[circt-sim] Strict JIT deopt process: id="
                     << entry.processId << " name=" << name
                     << " reason=" << entry.reason;
        if (!entry.detail.empty())
          llvm::errs() << " detail=" << entry.detail;
        llvm::errs() << "\n";
      }
      if (strictDeoptProcesses.size() > kStrictDeoptLogLimit) {
        llvm::errs() << "[circt-sim] Strict JIT deopt process: omitted="
                     << (strictDeoptProcesses.size() - kStrictDeoptLogLimit)
                     << " (log limit=" << kStrictDeoptLogLimit << ")\n";
      }
    }
    exitCode = 1;
  }
  // Check for SVA clocked assertion failures.
  if (exitCode == 0) {
    if (const auto *interp = simContext.getInterpreter()) {
      size_t assertionFailures = interp->getClockedAssertionFailures();
      if (assertionFailures > 0) {
        llvm::errs() << "[circt-sim] " << assertionFailures
                     << " SVA assertion failure(s)\n";
        exitCode = 1;
      }
    }
  }

  if (exitCode == 0)
    llvm::outs() << "[circt-sim] Simulation completed\n";
  else
    llvm::outs() << "[circt-sim] Simulation finished with exit code "
                 << exitCode << "\n";
  // Print coverage report if any covergroups were registered.
  __moore_coverage_report();
  llvm::outs().flush();
  llvm::errs().flush();
  std::fflush(stdout);
  std::fflush(stderr);
  _exit(exitCode);
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Increase stack size to 64 MB to handle deep UVM call chains.
  // The interpreter uses C++ recursion for func.call/call_indirect, and UVM
  // patterns (test → virtual_seq → sub_seq → start → body → ...) can nest
  // very deeply, overflowing the default 8 MB stack.
  {
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
      constexpr rlim_t kDesiredStack = 64ULL * 1024 * 1024; // 64 MB
      if (rl.rlim_cur < kDesiredStack) {
        rl.rlim_cur = std::min(kDesiredStack, rl.rlim_max);
        setrlimit(RLIMIT_STACK, &rl);
      }
    }
  }

  // Set up signal handling for clean shutdown
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  // Crash diagnostic: print last LLVM callee and current op before abort
  std::signal(SIGABRT, [](int) {
    extern const char *g_lastLLVMCallCallee;
    extern const char *g_lastOpName;
    extern unsigned g_lastProcId;
    extern char g_lastFuncName[256];
    extern unsigned g_lastFuncProcId;
    if (g_lastFuncName[0])
      fprintf(stderr, "[CRASH-DIAG] Last func body: %s proc=%u\n",
              g_lastFuncName, g_lastFuncProcId);
    if (g_lastLLVMCallCallee)
      fprintf(stderr, "[CRASH-DIAG] Last LLVM callee: %s\n",
              g_lastLLVMCallCallee);
    if (g_lastOpName)
      fprintf(stderr, "[CRASH-DIAG] Last op: %s proc=%u\n",
              g_lastOpName, g_lastProcId);
    std::signal(SIGABRT, SIG_DFL);
    std::raise(SIGABRT);
  });

  // Hide default LLVM options
  llvm::cl::HideUnrelatedOptions(
      {&mainCategory, &simCategory, &waveCategory, &parallelCategory,
       &debugCategory, &circt::getResourceGuardCategory()});

  // Register pass command line options
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Add version printer
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Extract Verilog plusargs (+key, +key=value) from argv before LLVM parsing.
  // LLVM's option parser rejects unknown arguments, so we pre-filter them.
  // These are stored and returned via vpi_get_vlog_info().
  extern std::vector<std::string> vlogPlusargs;
  std::vector<char *> filteredArgv;
  for (int i = 0; i < argc; ++i) {
    if (argv[i][0] == '+') {
      vlogPlusargs.push_back(argv[i]);
    } else {
      filteredArgv.push_back(argv[i]);
    }
  }
  int filteredArgc = static_cast<int>(filteredArgv.size());
  char **filteredArgvPtr = filteredArgv.data();

  // Moore runtime helpers ($test$plusargs/$value$plusargs and UVM run_test
  // command-line test selection) read arguments from CIRCT_UVM_ARGS/UVM_ARGS.
  // Merge command-line plusargs into CIRCT_UVM_ARGS so filtered '+' arguments
  // remain visible after LLVM option parsing.
  if (!vlogPlusargs.empty()) {
    std::string mergedUvmArgs;
    if (const char *envArgs = std::getenv("CIRCT_UVM_ARGS")) {
      mergedUvmArgs = envArgs;
    } else if (const char *legacyArgs = std::getenv("UVM_ARGS")) {
      mergedUvmArgs = legacyArgs;
    }
    for (const std::string &plusarg : vlogPlusargs) {
      if (!mergedUvmArgs.empty())
        mergedUvmArgs.push_back(' ');
      mergedUvmArgs.append(plusarg);
    }
    ::setenv("CIRCT_UVM_ARGS", mergedUvmArgs.c_str(), /*overwrite=*/1);
  }

  // Parse command line (with plusargs filtered out)
  llvm::cl::ParseCommandLineOptions(
      filteredArgc, filteredArgvPtr,
      "CIRCT Event-Driven Simulation Tool\n\n"
      "This tool simulates hardware designs using CIRCT's event-driven\n"
      "simulation infrastructure with IEEE 1800 scheduling semantics.\n");
  // circt-sim-specific: apply tighter resource limits than the generic 10 GB
  // defaults.  Multiple circt-sim instances may be launched in parallel (e.g.
  // by lit), so each instance must stay well below the system total.  Using
  // setenv with overwrite=0 means explicit user settings (env vars or CLI
  // flags) always take precedence.
  ::setenv("CIRCT_MAX_RSS_MB", "4096", /*overwrite=*/0);   // 4 GB RSS
  ::setenv("CIRCT_MAX_VMEM_MB", "8192", /*overwrite=*/0);  // 8 GB virtual
  ::setenv("CIRCT_MAX_WALL_MS", "300000", /*overwrite=*/0); // 5 min timeout
  circt::installResourceGuard();

  // Set up MLIR context with required dialects
  MLIRContext context;
  DialectRegistry registry;

  // Register dialects
  registry.insert<
      arc::ArcDialect,
      comb::CombDialect,
      emit::EmitDialect,
      hw::HWDialect,
      llhd::LLHDDialect,
      ltl::LTLDialect,
      mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect,
      mlir::DLTIDialect,
      mlir::func::FuncDialect,
      mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect,
      mlir::math::MathDialect,
      mlir::scf::SCFDialect,
      moore::MooreDialect,
      om::OMDialect,
      seq::SeqDialect,
      sim::SimDialect,
      sv::SVDialect,
      verif::VerifDialect>();

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  // Open input file
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // Set up source manager
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Process the input
  if (failed(processInput(context, sourceMgr))) {
    return 1;
  }

  // processInput() calls _exit(0) on success, so this is unreachable.
  return 0;
}
