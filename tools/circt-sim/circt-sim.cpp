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
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Support/WallClockTimeout.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

using namespace mlir;
using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Signal Handler for Clean Shutdown
//===----------------------------------------------------------------------===//

static std::atomic<bool> interruptRequested(false);
static std::atomic<bool> simulationStarted(false);
static void signalHandler(int) { interruptRequested.store(true); }

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
                    llvm::cl::desc("Maximum operations per process activation (0 = no limit)"),
                    llvm::cl::init(50000), llvm::cl::cat(simCategory));

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

private:
  /// Set up waveform tracing.
  LogicalResult setupWaveformTracing();

  /// Set up parallel simulation.
  LogicalResult setupParallelSimulation();

  /// Set up profiling.
  LogicalResult setupProfiling();

  /// Record a value change for waveform output.
  void recordValueChange(SignalId signal, const SignalValue &value);

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

  // Traced signals for VCD output
  llvm::SmallVector<std::pair<SignalId, char>, 64> tracedSignals;
  char nextVCDId = '!';

  // Module information
  llvm::SmallVector<std::string, 4> topModuleNames;
  mlir::ModuleOp rootModule;

  // LLHD Process interpreter
  std::unique_ptr<LLHDProcessInterpreter> llhdInterpreter;

  std::atomic<bool> abortRequested{false};
  std::atomic<bool> abortHandled{false};
  std::atomic<bool> stopWatchdog{false};
  std::mutex abortMutex;
  std::string abortReason;
  std::thread watchdogThread;
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

  // Collect all top modules to simulate
  llvm::SmallVector<hw::HWModuleOp, 4> hwModules;

  if (tops.empty()) {
    // No top modules specified - find the last module (typically the top)
    auto hwModule = findTopModule(module, "");
    if (!hwModule) {
      return failure();
    }
    hwModules.push_back(hwModule);
    topModuleNames.push_back(hwModule.getName().str());
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

    auto signalId = scheduler.registerSignal(portInfo.getName().str(),
                                              bitWidth);
    nameToSignal[portInfo.getName()] = signalId;

    // Set up tracing if enabled
    if (traceAll || traceSignals.empty()) {
      if (vcdWriter && nextVCDId <= '~') {
        vcdWriter->declareSignal(portInfo.getName().str(),
                                 bitWidth,
                                 nextVCDId);
        tracedSignals.push_back({signalId, nextVCDId++});
      }
    }
  }

  // Check if this module contains LLHD processes, seq.initial blocks, or
  // hw.instance ops (which may contain processes in submodules)
  bool hasLLHDProcesses = false;
  bool hasSeqInitial = false;
  bool hasInstances = false;
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
    }
    return WalkResult::advance();
  });
  if (abortRequested.load())
    return failure();

  llvm::outs() << "[circt-sim] Found " << llhdProcessCount << " LLHD processes"
               << ", " << seqInitialCount << " seq.initial blocks"
               << ", and " << instanceCount << " hw.instance ops"
               << " (out of " << totalOpsCount << " total ops) in module\n";

  // Initialize the interpreter if we have processes, initial blocks, or
  // instances (instances may contain processes in submodules that need
  // recursive initialization)
  if (hasLLHDProcesses || hasSeqInitial || hasInstances) {
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

      // Set up terminate callback to signal SimulationControl (only once)
      llhdInterpreter->setTerminateCallback(
          [this](bool success, bool verbose) {
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
            control.finish(success ? 0 : 1);
          });
    }

    // Initialize this module (will add to existing signals and processes)
    if (failed(llhdInterpreter->initialize(hwModule))) {
      llvm::errs() << "Error: Failed to initialize LLHD process interpreter\n";
      return failure();
    }

    llvm::outs() << "[circt-sim] Registered " << llhdInterpreter->getNumSignals()
                 << " LLHD signals and " << llhdInterpreter->getNumProcesses()
                 << " LLHD processes/initial blocks\n";
  } else {
    // For modules without LLHD processes, create a simple placeholder process
    auto topProcessId = scheduler.registerProcess(
        "top_eval", [this]() {
          // Placeholder evaluation function
        });

    // Mark as combinational (sensitive to all inputs)
    auto *process = scheduler.getProcess(topProcessId);
    if (process) {
      process->setCombinational(true);
      for (auto &entry : nameToSignal) {
        scheduler.addSensitivity(topProcessId, entry.second);
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
  if (!vcdWriter)
    return;

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
    vcdWriter->endHeader();
    vcdWriter->writeTime(0);
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
  }

  // Start profiling
  auto simulationStartTime = std::chrono::high_resolution_clock::now();

  // Initialize the scheduler
  scheduler.initialize();

  llvm::outs() << "[circt-sim] Starting simulation\n";
  llvm::outs().flush();

  // Main simulation loop
  uint64_t lastVCDTime = 0;
  uint64_t loopIterations = 0;
  auto startWallTime = std::chrono::steady_clock::now();

  // Track consecutive zero-delta iterations at the same time for loop detection
  uint64_t lastSimTime = 0;
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
    if (verbosity >= 3 && loopIterations <= 10) {
      llvm::outs() << "[circt-sim] Loop iteration " << loopIterations
                   << " at time " << scheduler.getCurrentTime().realTime << " fs\n";
      llvm::outs().flush();
    }

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
      break;
    }

    // Execute delta cycles
    if (verbosity >= 3) {
      llvm::outs() << "[circt-sim] Calling executeCurrentTime()...\n";
      llvm::outs().flush();
    }
    size_t deltasExecuted;
    if (parallelScheduler) {
      deltasExecuted = parallelScheduler->executeCurrentTimeParallel();
    } else {
      deltasExecuted = scheduler.executeCurrentTime();
    }
    if (verbosity >= 3) {
      llvm::outs() << "[circt-sim] executeCurrentTime() returned "
                   << deltasExecuted << " delta cycles\n";
      llvm::outs().flush();
    }

    // Check if simulation was terminated during execution (e.g., $finish called).
    // This must be checked immediately after executeCurrentTime() to ensure
    // termination is honored before any further processing.
    if (!control.shouldContinue())
      break;

    bool hasReadyProcesses = scheduler.hasReadyProcesses();

    if (deltasExecuted == 0) {
      // Track zero-delta iterations at the same time for loop detection
      if (currentTime.realTime == lastSimTime) {
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
        lastSimTime = currentTime.realTime;
        zeroIterationsAtSameTime = 0;
      }
    } else {
      if (currentTime.realTime == lastSimTime) {
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
      lastSimTime = currentTime.realTime;
    }

    if (!hasReadyProcesses) {
      // Check if simulation should stop before advancing time.
      // This ensures we report the correct termination time when $finish
      // is called (rather than advancing to the next scheduled event first).
      if (!control.shouldContinue())
        break;

      // No more events at current time, advance to next event
      if (!scheduler.advanceTime()) {
        // Print diagnostic information about why simulation can't advance
        if (verbosity >= 2) {
          llvm::outs() << "[circt-sim] advanceTime() returned false at time "
                       << currentTime.realTime << " fs\n";
          const auto &stats = scheduler.getStatistics();
          llvm::outs() << "[circt-sim] Processes registered: "
                       << stats.processesRegistered << "\n";
          llvm::outs() << "[circt-sim] Delta cycles executed: "
                       << stats.deltaCyclesExecuted << "\n";
        }
        // No more events
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

    // Write VCD time change
    if (vcdWriter && currentTime.realTime != lastVCDTime) {
      vcdWriter->writeTime(currentTime.realTime);
      lastVCDTime = currentTime.realTime;
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

//===----------------------------------------------------------------------===//
// Main Processing Pipeline
//===----------------------------------------------------------------------===//

static LogicalResult processInput(MLIRContext &context,
                                   llvm::SourceMgr &sourceMgr) {
  auto reportStage = [](llvm::StringRef stage) {
    if (verbosity >= 1)
      llvm::errs() << "[circt-sim] Stage: " << stage << "\n";
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
  // Run preprocessing passes if needed
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);

  // Add passes to lower to simulation-friendly form
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass pipeline failed\n";
    return failure();
  }

  // Create and initialize simulation context
  // Convert topModules from cl::list to SmallVector
  llvm::SmallVector<std::string, 4> tops;
  for (const auto &top : topModules) {
    tops.push_back(top);
  }

  reportStage("init");
  SimulationContext simContext;
  simContext.setMaxDeltaCycles(maxDeltas);
  simContext.setMaxProcessSteps(maxProcessSteps);
  if (failed(simContext.initialize(*module, tops))) {
    return failure();
  }

  // Run the simulation
  reportStage("run");
  if (failed(simContext.run())) {
    return failure();
  }

  // Print statistics if requested
  if (printStats) {
    simContext.printStatistics(llvm::outs());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Set up signal handling for clean shutdown
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // Hide default LLVM options
  llvm::cl::HideUnrelatedOptions(
      {&mainCategory, &simCategory, &waveCategory, &parallelCategory,
       &debugCategory});

  // Register pass command line options
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Add version printer
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse command line
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "CIRCT Event-Driven Simulation Tool\n\n"
      "This tool simulates hardware designs using CIRCT's event-driven\n"
      "simulation infrastructure with IEEE 1800 scheduling semantics.\n");

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

  llvm::outs() << "[circt-sim] Simulation finished successfully\n";
  return 0;
}
