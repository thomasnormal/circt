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

// Include CIRCT dialect ops BEFORE conversion passes to avoid namespace conflicts
// with LLVM ops (e.g., mlir::LLVM::AddOp vs circt::comb::AddOp)
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <memory>

using namespace mlir;
using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

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

// Top module selection
static llvm::cl::opt<std::string>
    topModule("top", llvm::cl::desc("Name of the top module"),
              llvm::cl::value_desc("name"), llvm::cl::init(""),
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
// MLIR Interpreter for HW/Comb/Seq Operations
//===----------------------------------------------------------------------===//

/// An MLIR interpreter that evaluates hw/comb/seq operations.
/// This provides an interpretive execution model for hardware simulation.
class MLIRInterpreter {
public:
  MLIRInterpreter() = default;

  /// Set a value for an MLIR Value in the interpreter state.
  void setValue(mlir::Value v, uint64_t val) {
    values[v] = val;
    valueIsX[v] = false;
  }

  /// Set a value as unknown (X).
  void setValueX(mlir::Value v) {
    values[v] = 0;
    valueIsX[v] = true;
  }

  /// Get the value of an MLIR Value, returning true if the value is known.
  bool getValue(mlir::Value v, uint64_t &result) const {
    auto it = values.find(v);
    if (it == values.end())
      return false;
    auto itX = valueIsX.find(v);
    if (itX != valueIsX.end() && itX->second)
      return false;
    result = it->second;
    return true;
  }

  /// Check if a value is X (unknown).
  bool isX(mlir::Value v) const {
    auto it = valueIsX.find(v);
    return it != valueIsX.end() && it->second;
  }

  /// Get the bit width of a value's type.
  static unsigned getBitWidth(mlir::Value v) {
    auto ty = v.getType();
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return intTy.getWidth();
    if (isa<seq::ClockType>(ty))
      return 1;
    return 0;
  }

  /// Evaluate a single operation and update the interpreter state.
  /// Returns true if the operation was successfully evaluated.
  bool evaluateOp(Operation *op) {
    // Handle hw.constant
    if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
      setValue(constOp.getResult(), constOp.getValue().getZExtValue());
      return true;
    }

    // Handle comb.add (variadic)
    if (auto addOp = dyn_cast<circt::comb::AddOp>(op)) {
      uint64_t result = 0;
      for (auto input : addOp.getInputs()) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        result += val;
      }
      unsigned width = getBitWidth(addOp.getResult());
      result &= (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
      setValue(addOp.getResult(), result);
      return true;
    }

    // Handle comb.sub
    if (auto subOp = dyn_cast<circt::comb::SubOp>(op)) {
      uint64_t lhs, rhs;
      if (!getValue(subOp.getLhs(), lhs) || !getValue(subOp.getRhs(), rhs))
        return setResultX(op), true;
      unsigned width = getBitWidth(subOp.getResult());
      uint64_t result = (lhs - rhs) & ((width < 64) ? ((1ULL << width) - 1) : ~0ULL);
      setValue(subOp.getResult(), result);
      return true;
    }

    // Handle comb.mul (variadic)
    if (auto mulOp = dyn_cast<circt::comb::MulOp>(op)) {
      uint64_t result = 1;
      for (auto input : mulOp.getInputs()) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        result *= val;
      }
      unsigned width = getBitWidth(mulOp.getResult());
      result &= (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
      setValue(mulOp.getResult(), result);
      return true;
    }

    // Handle comb.and (variadic)
    if (auto andOp = dyn_cast<circt::comb::AndOp>(op)) {
      uint64_t result = ~0ULL;
      for (auto input : andOp.getInputs()) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        result &= val;
      }
      unsigned width = getBitWidth(andOp.getResult());
      result &= (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
      setValue(andOp.getResult(), result);
      return true;
    }

    // Handle comb.or (variadic)
    if (auto orOp = dyn_cast<circt::comb::OrOp>(op)) {
      uint64_t result = 0;
      for (auto input : orOp.getInputs()) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        result |= val;
      }
      setValue(orOp.getResult(), result);
      return true;
    }

    // Handle comb.xor (variadic)
    if (auto xorOp = dyn_cast<circt::comb::XorOp>(op)) {
      uint64_t result = 0;
      for (auto input : xorOp.getInputs()) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        result ^= val;
      }
      setValue(xorOp.getResult(), result);
      return true;
    }

    // Handle comb.shl
    if (auto shlOp = dyn_cast<circt::comb::ShlOp>(op)) {
      uint64_t lhs, rhs;
      if (!getValue(shlOp.getLhs(), lhs) || !getValue(shlOp.getRhs(), rhs))
        return setResultX(op), true;
      unsigned width = getBitWidth(shlOp.getResult());
      uint64_t result = (lhs << rhs) & ((width < 64) ? ((1ULL << width) - 1) : ~0ULL);
      setValue(shlOp.getResult(), result);
      return true;
    }

    // Handle comb.shru (logical right shift)
    if (auto shruOp = dyn_cast<circt::comb::ShrUOp>(op)) {
      uint64_t lhs, rhs;
      if (!getValue(shruOp.getLhs(), lhs) || !getValue(shruOp.getRhs(), rhs))
        return setResultX(op), true;
      setValue(shruOp.getResult(), lhs >> rhs);
      return true;
    }

    // Handle comb.shrs (arithmetic right shift)
    if (auto shrsOp = dyn_cast<circt::comb::ShrSOp>(op)) {
      uint64_t lhs, rhs;
      if (!getValue(shrsOp.getLhs(), lhs) || !getValue(shrsOp.getRhs(), rhs))
        return setResultX(op), true;
      unsigned width = getBitWidth(shrsOp.getLhs());
      // Sign extend then arithmetic shift
      int64_t signedVal = static_cast<int64_t>(lhs);
      if (width < 64 && (lhs & (1ULL << (width - 1))))
        signedVal |= ~((1ULL << width) - 1);
      uint64_t result = static_cast<uint64_t>(signedVal >> rhs);
      result &= (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
      setValue(shrsOp.getResult(), result);
      return true;
    }

    // Handle comb.mux
    if (auto muxOp = dyn_cast<circt::comb::MuxOp>(op)) {
      uint64_t cond;
      if (!getValue(muxOp.getCond(), cond))
        return setResultX(op), true;
      mlir::Value selected = cond ? muxOp.getTrueValue() : muxOp.getFalseValue();
      uint64_t result;
      if (!getValue(selected, result))
        return setResultX(op), true;
      setValue(muxOp.getResult(), result);
      return true;
    }

    // Handle comb.extract
    if (auto extractOp = dyn_cast<circt::comb::ExtractOp>(op)) {
      uint64_t input;
      if (!getValue(extractOp.getInput(), input))
        return setResultX(op), true;
      unsigned lowBit = extractOp.getLowBit();
      unsigned width = getBitWidth(extractOp.getResult());
      uint64_t mask = (width < 64) ? ((1ULL << width) - 1) : ~0ULL;
      uint64_t result = (input >> lowBit) & mask;
      setValue(extractOp.getResult(), result);
      return true;
    }

    // Handle comb.concat
    if (auto concatOp = dyn_cast<circt::comb::ConcatOp>(op)) {
      uint64_t result = 0;
      unsigned totalBits = 0;
      // Concat goes from MSB to LSB in operand order
      for (auto input : llvm::reverse(concatOp.getInputs())) {
        uint64_t val;
        if (!getValue(input, val))
          return setResultX(op), true;
        unsigned width = getBitWidth(input);
        result |= (val << totalBits);
        totalBits += width;
      }
      setValue(concatOp.getResult(), result);
      return true;
    }

    // Handle comb.replicate
    if (auto replicateOp = dyn_cast<circt::comb::ReplicateOp>(op)) {
      uint64_t input;
      if (!getValue(replicateOp.getInput(), input))
        return setResultX(op), true;
      unsigned inputWidth = getBitWidth(replicateOp.getInput());
      unsigned outputWidth = getBitWidth(replicateOp.getResult());
      unsigned multiple = outputWidth / inputWidth;
      uint64_t result = 0;
      for (unsigned i = 0; i < multiple; ++i) {
        result |= (input << (i * inputWidth));
      }
      setValue(replicateOp.getResult(), result);
      return true;
    }

    // Handle comb.icmp
    if (auto icmpOp = dyn_cast<circt::comb::ICmpOp>(op)) {
      uint64_t lhs, rhs;
      if (!getValue(icmpOp.getLhs(), lhs) || !getValue(icmpOp.getRhs(), rhs))
        return setResultX(op), true;

      bool result = false;
      unsigned width = getBitWidth(icmpOp.getLhs());
      int64_t signedLhs = static_cast<int64_t>(lhs);
      int64_t signedRhs = static_cast<int64_t>(rhs);
      // Sign extend for signed comparisons
      if (width < 64) {
        if (lhs & (1ULL << (width - 1)))
          signedLhs |= ~((1ULL << width) - 1);
        if (rhs & (1ULL << (width - 1)))
          signedRhs |= ~((1ULL << width) - 1);
      }

      switch (icmpOp.getPredicate()) {
      case circt::comb::ICmpPredicate::eq:
      case circt::comb::ICmpPredicate::ceq:
      case circt::comb::ICmpPredicate::weq:
        result = (lhs == rhs);
        break;
      case circt::comb::ICmpPredicate::ne:
      case circt::comb::ICmpPredicate::cne:
      case circt::comb::ICmpPredicate::wne:
        result = (lhs != rhs);
        break;
      case circt::comb::ICmpPredicate::slt:
        result = (signedLhs < signedRhs);
        break;
      case circt::comb::ICmpPredicate::sle:
        result = (signedLhs <= signedRhs);
        break;
      case circt::comb::ICmpPredicate::sgt:
        result = (signedLhs > signedRhs);
        break;
      case circt::comb::ICmpPredicate::sge:
        result = (signedLhs >= signedRhs);
        break;
      case circt::comb::ICmpPredicate::ult:
        result = (lhs < rhs);
        break;
      case circt::comb::ICmpPredicate::ule:
        result = (lhs <= rhs);
        break;
      case circt::comb::ICmpPredicate::ugt:
        result = (lhs > rhs);
        break;
      case circt::comb::ICmpPredicate::uge:
        result = (lhs >= rhs);
        break;
      }
      setValue(icmpOp.getResult(), result ? 1 : 0);
      return true;
    }

    // Handle comb.parity
    if (auto parityOp = dyn_cast<circt::comb::ParityOp>(op)) {
      uint64_t input;
      if (!getValue(parityOp.getInput(), input))
        return setResultX(op), true;
      unsigned width = getBitWidth(parityOp.getInput());
      bool parity = false;
      for (unsigned i = 0; i < width; ++i) {
        parity ^= ((input >> i) & 1);
      }
      setValue(parityOp.getResult(), parity ? 1 : 0);
      return true;
    }

    // Handle seq.to_clock (convert i1 to clock type)
    if (auto toClockOp = dyn_cast<seq::ToClockOp>(op)) {
      uint64_t input;
      if (!getValue(toClockOp.getInput(), input))
        return setResultX(op), true;
      setValue(toClockOp.getResult(), input & 1);
      return true;
    }

    // Handle seq.from_clock (convert clock to i1)
    if (auto fromClockOp = dyn_cast<seq::FromClockOp>(op)) {
      uint64_t input;
      if (!getValue(fromClockOp.getInput(), input))
        return setResultX(op), true;
      setValue(fromClockOp.getResult(), input & 1);
      return true;
    }

    // Handle hw.output - nothing to evaluate, just indicates end of module
    if (isa<hw::OutputOp>(op)) {
      return true;
    }

    // Unhandled operation - mark results as X
    for (auto result : op->getResults()) {
      setValueX(result);
    }
    return true;
  }

  /// Set all results of an operation to X.
  void setResultX(Operation *op) {
    for (auto result : op->getResults()) {
      setValueX(result);
    }
  }

  /// Clear all stored values.
  void clear() {
    values.clear();
    valueIsX.clear();
  }

private:
  llvm::DenseMap<mlir::Value, uint64_t> values;
  llvm::DenseMap<mlir::Value, bool> valueIsX;
};

//===----------------------------------------------------------------------===//
// Register State for Sequential Simulation
//===----------------------------------------------------------------------===//

/// Tracks the state of sequential elements (registers) across clock cycles.
struct RegisterState {
  mlir::Value dataInput;      // The input value to the register
  mlir::Value clock;          // The clock signal
  mlir::Value reset;          // Optional reset signal
  mlir::Value resetValue;     // Optional reset value
  uint64_t currentValue = 0;  // Current stored value
  uint64_t nextValue = 0;     // Next value to be latched
  bool isX = true;            // Whether the current value is X
  bool isAsync = false;       // Whether reset is asynchronous
  unsigned width = 0;         // Bit width of the register
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
    if (parallelScheduler) {
      parallelScheduler->stopWorkers();
    }
  }

  /// Initialize the simulation from an MLIR module.
  LogicalResult initialize(mlir::ModuleOp module, const std::string &top);

  /// Run the simulation.
  LogicalResult run();

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

  ProcessScheduler scheduler;
  SimulationControl control;
  std::unique_ptr<PerformanceProfiler> profiler;
  std::unique_ptr<VCDWriter> vcdWriter;
  std::unique_ptr<ParallelScheduler> parallelScheduler;

  // Maps from MLIR values to simulation signals
  llvm::DenseMap<mlir::Value, SignalId> valueToSignal;
  llvm::StringMap<SignalId> nameToSignal;

  // Traced signals for VCD output
  llvm::SmallVector<std::pair<SignalId, char>, 64> tracedSignals;
  char nextVCDId = '!';

  // Module information
  std::string topModuleName;
  mlir::ModuleOp rootModule;
  hw::HWModuleOp topHWModule;

  // MLIR Interpreter for evaluating operations
  MLIRInterpreter interpreter;

  // Register states for sequential simulation
  llvm::SmallVector<RegisterState, 16> registerStates;

  // Clock edge tracking
  llvm::DenseMap<SignalId, bool> previousClockValues;
};

LogicalResult SimulationContext::initialize(mlir::ModuleOp module,
                                            const std::string &top) {
  rootModule = module;

  // Find the top module
  auto hwModule = findTopModule(module, top);
  if (!hwModule) {
    return failure();
  }

  topModuleName = hwModule.getName().str();

  // Set up waveform tracing if requested
  if (failed(setupWaveformTracing()))
    return failure();

  // Set up profiling if requested
  if (failed(setupProfiling()))
    return failure();

  // Build the simulation model
  if (failed(buildSimulationModel(hwModule)))
    return failure();

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
  topHWModule = hwModule;

  // Get the module body
  Block &body = hwModule.getBody().front();

  // Map block arguments (input ports) to signals
  auto portList = hwModule.getPortList();
  unsigned inputIdx = 0;
  for (auto portInfo : portList) {
    // Determine bit width - clock types are treated as 1-bit signals
    unsigned bitWidth;
    if (isa<seq::ClockType>(portInfo.type)) {
      bitWidth = 1;
    } else {
      bitWidth = portInfo.type.getIntOrFloatBitWidth();
    }

    auto signalId = scheduler.registerSignal(portInfo.getName().str(), bitWidth);
    nameToSignal[portInfo.getName()] = signalId;

    // Map input ports to block arguments for the interpreter
    if (portInfo.dir == hw::ModulePort::Direction::Input) {
      if (inputIdx < body.getNumArguments()) {
        valueToSignal[body.getArgument(inputIdx)] = signalId;
      }
      inputIdx++;
    }

    // Set up tracing if enabled
    if (traceAll || traceSignals.empty()) {
      if (vcdWriter && nextVCDId <= '~') {
        vcdWriter->declareSignal(portInfo.getName().str(), bitWidth, nextVCDId);
        tracedSignals.push_back({signalId, nextVCDId++});
      }
    }
  }

  // Walk the module body and find all sequential elements (registers)
  hwModule.walk([&](Operation *op) {
    // Handle seq.firreg (FIRRTL-style register)
    if (auto firRegOp = dyn_cast<seq::FirRegOp>(op)) {
      RegisterState regState;
      regState.dataInput = firRegOp.getNext();
      regState.clock = firRegOp.getClk();
      regState.reset = firRegOp.getReset();
      regState.resetValue = firRegOp.getResetValue();
      regState.width = MLIRInterpreter::getBitWidth(firRegOp.getData());
      regState.isX = true;
      regState.isAsync = firRegOp.getIsAsync();

      // Map the register output value to its state index
      valueToSignal[firRegOp.getData()] =
          scheduler.registerSignal(firRegOp.getNameAttr()
                                       ? firRegOp.getNameAttr().str()
                                       : "reg",
                                   regState.width);
      registerStates.push_back(regState);
    }

    // Handle seq.compreg
    if (auto compRegOp = dyn_cast<seq::CompRegOp>(op)) {
      RegisterState regState;
      regState.dataInput = compRegOp.getInput();
      regState.clock = compRegOp.getClk();
      regState.reset = compRegOp.getReset();
      regState.resetValue = compRegOp.getResetValue();
      regState.width = MLIRInterpreter::getBitWidth(compRegOp.getData());
      regState.isX = true;
      regState.isAsync = false;  // compreg uses sync reset

      valueToSignal[compRegOp.getData()] =
          scheduler.registerSignal(compRegOp.getNameAttr()
                                       ? compRegOp.getNameAttr().str()
                                       : "compreg",
                                   regState.width);
      registerStates.push_back(regState);
    }
  });

  // Create the combinational evaluation process
  auto combProcessId = scheduler.registerProcess(
      "comb_eval", [this, &body]() {
        // Set input values from signals to interpreter
        for (auto arg : body.getArguments()) {
          auto it = valueToSignal.find(arg);
          if (it != valueToSignal.end()) {
            const auto &sigVal = scheduler.getSignalValue(it->second);
            if (sigVal.isUnknown()) {
              interpreter.setValueX(arg);
            } else {
              interpreter.setValue(arg, sigVal.getValue());
            }
          }
        }

        // Set register output values from register states
        for (size_t i = 0; i < registerStates.size(); ++i) {
          auto &regState = registerStates[i];
          // Find the Value corresponding to this register
          for (auto &op : body) {
            if (auto firRegOp = dyn_cast<seq::FirRegOp>(&op)) {
              if (&registerStates[i] == &regState) {
                if (regState.isX) {
                  interpreter.setValueX(firRegOp.getData());
                } else {
                  interpreter.setValue(firRegOp.getData(), regState.currentValue);
                }
              }
            }
            if (auto compRegOp = dyn_cast<seq::CompRegOp>(&op)) {
              if (&registerStates[i] == &regState) {
                if (regState.isX) {
                  interpreter.setValueX(compRegOp.getData());
                } else {
                  interpreter.setValue(compRegOp.getData(), regState.currentValue);
                }
              }
            }
          }
        }

        // Evaluate all operations in order
        for (auto &op : body) {
          // Skip register ops (their outputs are set above)
          if (isa<seq::FirRegOp, seq::CompRegOp>(&op))
            continue;
          interpreter.evaluateOp(&op);
        }

        // Update output signals from interpreter
        if (auto outputOp = dyn_cast<hw::OutputOp>(body.getTerminator())) {
          unsigned outputIdx = 0;
          for (auto portInfo : topHWModule.getPortList()) {
            if (portInfo.dir == hw::ModulePort::Direction::Output) {
              auto it = nameToSignal.find(portInfo.getName());
              if (it != nameToSignal.end() && outputIdx < outputOp.getNumOperands()) {
                uint64_t val;
                auto outputVal = outputOp.getOperand(outputIdx);
                if (interpreter.getValue(outputVal, val)) {
                  scheduler.updateSignal(it->second,
                      SignalValue(val, MLIRInterpreter::getBitWidth(outputVal)));
                } else {
                  scheduler.updateSignal(it->second,
                      SignalValue::makeX(MLIRInterpreter::getBitWidth(outputVal)));
                }
              }
              outputIdx++;
            }
          }
        }
      });

  // Create clock edge detection process for sequential elements
  if (!registerStates.empty()) {
    auto seqProcessId = scheduler.registerProcess(
        "seq_eval", [this, &body]() {
          // For each register, check for clock edges and update state
          size_t regIdx = 0;
          for (auto &op : body) {
            if (auto firRegOp = dyn_cast<seq::FirRegOp>(&op)) {
              auto &regState = registerStates[regIdx++];

              // Get clock value
              uint64_t clockVal;
              bool clockKnown = interpreter.getValue(firRegOp.getClk(), clockVal);

              // Check for reset (async resets take priority)
              if (regState.isAsync && firRegOp.getReset()) {
                uint64_t resetVal;
                if (interpreter.getValue(firRegOp.getReset(), resetVal) && resetVal) {
                  uint64_t rstValue = 0;
                  if (firRegOp.getResetValue()) {
                    interpreter.getValue(firRegOp.getResetValue(), rstValue);
                  }
                  regState.currentValue = rstValue;
                  regState.isX = false;
                  interpreter.setValue(firRegOp.getData(), regState.currentValue);
                  continue;
                }
              }

              // Check for posedge clock
              auto clkSignalIt = valueToSignal.find(firRegOp.getClk());
              SignalId clkSignalId = clkSignalIt != valueToSignal.end()
                                         ? clkSignalIt->second
                                         : 0;
              bool prevClk = previousClockValues.count(clkSignalId)
                                 ? previousClockValues[clkSignalId]
                                 : false;
              bool currClk = clockKnown && (clockVal & 1);

              if (!prevClk && currClk) {
                // Posedge detected
                // Check sync reset
                if (!regState.isAsync && firRegOp.getReset()) {
                  uint64_t resetVal;
                  if (interpreter.getValue(firRegOp.getReset(), resetVal) && resetVal) {
                    uint64_t rstValue = 0;
                    if (firRegOp.getResetValue()) {
                      interpreter.getValue(firRegOp.getResetValue(), rstValue);
                    }
                    regState.currentValue = rstValue;
                    regState.isX = false;
                    interpreter.setValue(firRegOp.getData(), regState.currentValue);
                    previousClockValues[clkSignalId] = currClk;
                    continue;
                  }
                }

                // Latch the next value
                uint64_t nextVal;
                if (interpreter.getValue(firRegOp.getNext(), nextVal)) {
                  regState.currentValue = nextVal;
                  regState.isX = false;
                } else {
                  regState.isX = true;
                }
                interpreter.setValue(firRegOp.getData(), regState.currentValue);
              }

              previousClockValues[clkSignalId] = currClk;
            }

            if (auto compRegOp = dyn_cast<seq::CompRegOp>(&op)) {
              auto &regState = registerStates[regIdx++];

              // Get clock value
              uint64_t clockVal;
              bool clockKnown = interpreter.getValue(compRegOp.getClk(), clockVal);

              // Check for posedge clock
              auto clkSignalIt = valueToSignal.find(compRegOp.getClk());
              SignalId clkSignalId = clkSignalIt != valueToSignal.end()
                                         ? clkSignalIt->second
                                         : 0;
              bool prevClk = previousClockValues.count(clkSignalId)
                                 ? previousClockValues[clkSignalId]
                                 : false;
              bool currClk = clockKnown && (clockVal & 1);

              if (!prevClk && currClk) {
                // Posedge detected - check sync reset
                if (compRegOp.getReset()) {
                  uint64_t resetVal;
                  if (interpreter.getValue(compRegOp.getReset(), resetVal) && resetVal) {
                    uint64_t rstValue = 0;
                    if (compRegOp.getResetValue()) {
                      interpreter.getValue(compRegOp.getResetValue(), rstValue);
                    }
                    regState.currentValue = rstValue;
                    regState.isX = false;
                    interpreter.setValue(compRegOp.getData(), regState.currentValue);
                    previousClockValues[clkSignalId] = currClk;
                    continue;
                  }
                }

                // Latch the next value
                uint64_t nextVal;
                if (interpreter.getValue(compRegOp.getInput(), nextVal)) {
                  regState.currentValue = nextVal;
                  regState.isX = false;
                } else {
                  regState.isX = true;
                }
                interpreter.setValue(compRegOp.getData(), regState.currentValue);
              }

              previousClockValues[clkSignalId] = currClk;
            }
          }
        });

    // Sequential process is sensitive to clock edges
    auto *seqProcess = scheduler.getProcess(seqProcessId);
    if (seqProcess) {
      for (auto &entry : nameToSignal) {
        if (entry.getKey().contains("clk") || entry.getKey().contains("clock")) {
          scheduler.addSensitivity(seqProcessId, entry.second, EdgeType::Posedge);
        }
      }
      // Also sensitive to all inputs for async reset
      for (auto &entry : nameToSignal) {
        scheduler.addSensitivity(seqProcessId, entry.second);
      }
    }
  }

  // Mark combinational process as sensitive to all inputs
  auto *combProcess = scheduler.getProcess(combProcessId);
  if (combProcess) {
    combProcess->setCombinational(true);
    for (auto &entry : nameToSignal) {
      scheduler.addSensitivity(combProcessId, entry.second);
    }
  }

  // Create clock driver for ports named "clk" or "clock"
  for (auto &entry : nameToSignal) {
    if (entry.getKey().contains("clk") || entry.getKey().contains("clock")) {
      SignalId clockSignal = entry.second;

      // Initialize clock to 0
      scheduler.updateSignal(clockSignal, SignalValue(0, 1));

      // Create a clock generator using scheduled events
      // Clock period: 10 ns (5 ns high, 5 ns low)
      constexpr uint64_t halfPeriodFs = 5000000000000ULL; // 5 ns in fs

      // Use shared_ptr to enable self-referential lambda for recursive scheduling
      auto toggleClock = std::make_shared<std::function<void()>>();
      *toggleClock = [this, clockSignal, halfPeriodFs, toggleClock]() {
        auto &val = scheduler.getSignalValue(clockSignal);
        uint64_t newVal = val.isUnknown() ? 1 : (val.getValue() ^ 1);
        scheduler.updateSignal(clockSignal, SignalValue(newVal, 1));

        // Schedule next toggle
        uint64_t nextTime = scheduler.getCurrentTime().realTime + halfPeriodFs;
        uint64_t maxTimeFs = static_cast<uint64_t>(::maxTime);
        if (maxTimeFs == 0 || nextTime <= maxTimeFs) {
          scheduler.getEventScheduler().schedule(
              SimTime(nextTime, 0, 0), SchedulingRegion::Active,
              Event(*toggleClock));
        }
      };

      // Schedule first clock edge
      scheduler.getEventScheduler().schedule(
          SimTime(halfPeriodFs, 0, 0), SchedulingRegion::Active,
          Event(*toggleClock));

      if (verbosity > 0) {
        llvm::outs() << "[circt-sim] Clock driver scheduled for signal '"
                     << entry.getKey() << "' (period=10ns)\n";
      }
      break; // Only one clock driver
    }
  }

  // Create reset driver process for ports named "rst" or "reset"
  for (auto &entry : nameToSignal) {
    StringRef portName = entry.getKey();
    if (portName.contains("rst") || portName.contains("reset")) {
      SignalId resetSignal = entry.second;
      bool activeHigh = !portName.contains("n"); // rstn, resetn are active-low

      // Assert reset initially
      scheduler.updateSignal(resetSignal, SignalValue(activeHigh ? 1 : 0, 1));

      // Deassert reset after 100 ns
      constexpr uint64_t resetDelayFs = 100000000000000ULL; // 100 ns in fs
      scheduler.getEventScheduler().schedule(
          SimTime(resetDelayFs, 0, 0), SchedulingRegion::Active,
          Event([this, resetSignal, activeHigh]() {
            scheduler.updateSignal(resetSignal, SignalValue(activeHigh ? 0 : 1, 1));
          }));

      if (verbosity > 0) {
        llvm::outs() << "[circt-sim] Reset driver scheduled for signal '"
                     << portName << "' (active-"
                     << (activeHigh ? "high" : "low") << ")\n";
      }
      break; // Only one reset driver
    }
  }

  return success();
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
  // Write VCD header
  if (vcdWriter) {
    vcdWriter->writeHeader(topModuleName);
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

  // Main simulation loop
  uint64_t lastVCDTime = 0;
  auto startWallTime = std::chrono::steady_clock::now();

  while (control.shouldContinue()) {
    // Check wall-clock timeout
    if (timeout > 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - startWallTime);
      if (static_cast<uint64_t>(elapsed.count()) >= timeout) {
        control.warning("TIMEOUT", "Wall-clock timeout reached");
        break;
      }
    }

    // Check simulation time limit
    const auto &currentTime = scheduler.getCurrentTime();
    if (maxTime > 0 && currentTime.realTime >= maxTime) {
      break;
    }

    // Execute delta cycles
    size_t deltasExecuted;
    if (parallelScheduler) {
      deltasExecuted = parallelScheduler->executeCurrentTimeParallel();
    } else {
      deltasExecuted = scheduler.executeCurrentTime();
    }

    if (verbosity > 1 && deltasExecuted > 0) {
      llvm::outs() << "[circt-sim] Executed " << deltasExecuted
                   << " deltas at time " << currentTime.realTime << " fs\n";
    }

    if (deltasExecuted == 0) {
      // No more events at current time, try to advance to next event
      auto &eventSched = scheduler.getEventScheduler();

      // Check if there are any events left
      if (eventSched.isComplete()) {
        if (verbosity > 1) {
          llvm::outs() << "[circt-sim] No more events, exiting\n";
        }
        break;
      }

      // Use runUntil to process events and advance time
      // Run up to 1 time step (next real time event)
      uint64_t nextStepTime = currentTime.realTime + 1;
      if (maxTime > 0 && nextStepTime > maxTime)
        nextStepTime = maxTime;
      eventSched.runUntil(nextStepTime);
    }

    // Check for excessive delta cycles (infinite loop detection)
    if (deltasExecuted > maxDeltas) {
      control.error("DELTA_OVERFLOW",
                    "Too many delta cycles - possible infinite loop");
      control.finish(1);
      break;
    }

    // Write VCD time and value changes
    if (vcdWriter && currentTime.realTime != lastVCDTime) {
      vcdWriter->writeTime(currentTime.realTime);
      lastVCDTime = currentTime.realTime;

      // Write all traced signal values
      for (const auto &traced : tracedSignals) {
        const auto &value = scheduler.getSignalValue(traced.first);
        if (value.isUnknown()) {
          vcdWriter->writeUnknown(traced.second, value.getWidth());
        } else {
          vcdWriter->writeValue(traced.second, value.getValue(), value.getWidth());
        }
      }
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

  if (profiler) {
    os << "\n--- Profiling Statistics ---\n";
    // profiler->printReport(os);
  }

  os << "=============================\n";
}

//===----------------------------------------------------------------------===//
// Signal Handler for Clean Shutdown
//===----------------------------------------------------------------------===//

static std::atomic<bool> interruptRequested(false);

static void signalHandler(int) { interruptRequested.store(true); }

//===----------------------------------------------------------------------===//
// Main Processing Pipeline
//===----------------------------------------------------------------------===//

static LogicalResult processInput(MLIRContext &context,
                                   llvm::SourceMgr &sourceMgr) {
  // Parse the input file
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: Failed to parse input\n";
    return failure();
  }

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

    return success();
  }

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
  SimulationContext simContext;
  if (failed(simContext.initialize(*module, topModule))) {
    return failure();
  }

  // Run the simulation
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
