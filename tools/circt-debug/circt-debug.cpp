//===- circt-debug.cpp - Interactive Hardware Debugger --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-debug' tool, an interactive debugger for
// hardware designs. It provides a GDB-like interface for stepping through
// simulations, setting breakpoints, and inspecting signal values.
//
// Commands:
//   run [cycles]     - Run simulation for N cycles
//   step [n]         - Step N clock cycles
//   stepi            - Step one delta cycle
//   continue         - Run until breakpoint or end
//   break <loc>      - Set breakpoint (file:line)
//   break -sig <s>   - Break on signal change
//   watch <signal>   - Watch for signal changes
//   print <expr>     - Print signal/expression value
//   info signals     - List signals in current scope
//   scope <path>     - Change hierarchical scope
//   dump vcd <file>  - Start VCD waveform dump
//   help             - Show help
//   quit             - Exit debugger
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-debug/Debug.h"
#include "circt/Tools/circt-debug/DebugSession.h"
#include "circt/Tools/circt-debug/DAPServer.h"
#include "circt/Tools/circt-debug/VCDWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace circt::debug;

namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-debug Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> topModule("top", cl::desc("Top module name"),
                                      cl::value_desc("module"),
                                      cl::init("top"), cl::cat(mainCategory));

static cl::opt<std::string> vcdFile("vcd", cl::desc("VCD output file"),
                                    cl::value_desc("filename"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::opt<std::string> outputDir("output-dir",
                                      cl::desc("Output directory"),
                                      cl::value_desc("directory"), cl::init("."),
                                      cl::cat(mainCategory));

static cl::opt<uint64_t> maxCycles("max-cycles",
                                   cl::desc("Maximum simulation cycles"),
                                   cl::value_desc("N"), cl::init(0),
                                   cl::cat(mainCategory));

static cl::opt<bool> verbose("verbose", cl::desc("Verbose output"),
                             cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string>
    commandFile("x", cl::desc("Execute commands from file"),
                cl::value_desc("filename"), cl::init(""),
                cl::cat(mainCategory));

static cl::opt<bool> batch("batch", cl::desc("Run in batch mode (non-interactive)"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> dapMode("dap", cl::desc("Run as Debug Adapter Protocol server"),
                             cl::init(false), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Mock Simulation Backend
//===----------------------------------------------------------------------===//

namespace {

/// A simple mock simulation backend for testing the debugger infrastructure.
/// In a real implementation, this would connect to Verilator, LLHD, or another
/// simulation engine.
class MockSimulationBackend : public SimulationBackend {
public:
  MockSimulationBackend() = default;

  bool initialize(const DebugConfig &config) override {
    this->config = config;

    // Create a simple mock design hierarchy
    auto root = std::make_unique<Scope>(config.topModule);

    // Add some mock signals
    SignalInfo clk;
    clk.name = "clk";
    clk.fullPath = config.topModule + ".clk";
    clk.type = SignalType::Input;
    clk.width = 1;
    clk.isSigned = false;
    root->addSignal(clk);

    SignalInfo rst;
    rst.name = "rst";
    rst.fullPath = config.topModule + ".rst";
    rst.type = SignalType::Input;
    rst.width = 1;
    rst.isSigned = false;
    root->addSignal(rst);

    SignalInfo counter;
    counter.name = "counter";
    counter.fullPath = config.topModule + ".counter";
    counter.type = SignalType::Reg;
    counter.width = 8;
    counter.isSigned = false;
    root->addSignal(counter);

    SignalInfo dataOut;
    dataOut.name = "data_out";
    dataOut.fullPath = config.topModule + ".data_out";
    dataOut.type = SignalType::Output;
    dataOut.width = 32;
    dataOut.isSigned = false;
    root->addSignal(dataOut);

    // Add a child scope
    auto submod = std::make_unique<Scope>("submodule", root.get());
    SignalInfo subReg;
    subReg.name = "reg_a";
    subReg.fullPath = config.topModule + ".submodule.reg_a";
    subReg.type = SignalType::Reg;
    subReg.width = 16;
    subReg.isSigned = false;
    submod->addSignal(subReg);
    root->addChild(std::move(submod));

    state.setRootScope(std::move(root));

    // Initialize signal values
    state.setSignalValue(config.topModule + ".clk", SignalValue(1, 0));
    state.setSignalValue(config.topModule + ".rst", SignalValue(1, 1));
    state.setSignalValue(config.topModule + ".counter", SignalValue(8, 0));
    state.setSignalValue(config.topModule + ".data_out", SignalValue(32, 0));
    state.setSignalValue(config.topModule + ".submodule.reg_a",
                         SignalValue(16, 0));

    return true;
  }

  bool reset() override {
    state.setCycle(0);
    state.setTime(SimTime(0));
    state.setDeltaCycle(0);

    // Reset signal values
    state.setSignalValue(config.topModule + ".clk", SignalValue(1, 0));
    state.setSignalValue(config.topModule + ".rst", SignalValue(1, 1));
    state.setSignalValue(config.topModule + ".counter", SignalValue(8, 0));
    state.setSignalValue(config.topModule + ".data_out", SignalValue(32, 0));
    state.setSignalValue(config.topModule + ".submodule.reg_a",
                         SignalValue(16, 0));

    return true;
  }

  bool stepDelta() override {
    state.advanceDeltaCycle();

    // Simple mock behavior: toggle clock on each delta
    auto clk = state.getSignalValue(config.topModule + ".clk");
    if (clk.getBit(0) == LogicValue::Zero)
      state.setSignalValue(config.topModule + ".clk", SignalValue(1, 1));
    else
      state.setSignalValue(config.topModule + ".clk", SignalValue(1, 0));

    return true;
  }

  bool stepClock() override {
    state.advanceCycle();
    state.advanceTime(SimTime(10)); // 10ns per clock
    state.resetDeltaCycle();

    // Toggle clock
    state.setSignalValue(config.topModule + ".clk", SignalValue(1, 1));

    // Simple mock behavior: increment counter each cycle
    auto counter = state.getSignalValue(config.topModule + ".counter");
    uint64_t counterVal = counter.toAPInt().getZExtValue();
    state.setSignalValue(config.topModule + ".counter",
                         SignalValue(8, (counterVal + 1) & 0xFF));

    // Update data_out based on counter
    state.setSignalValue(config.topModule + ".data_out",
                         SignalValue(32, (counterVal + 1) * 0x1000));

    // Update submodule register
    auto subReg = state.getSignalValue(config.topModule + ".submodule.reg_a");
    uint64_t subRegVal = subReg.toAPInt().getZExtValue();
    state.setSignalValue(config.topModule + ".submodule.reg_a",
                         SignalValue(16, (subRegVal + 3) & 0xFFFF));

    // Toggle clock back
    state.setSignalValue(config.topModule + ".clk", SignalValue(1, 0));

    // Deassert reset after first cycle
    if (state.getCycle() == 1) {
      state.setSignalValue(config.topModule + ".rst", SignalValue(1, 0));
    }

    return true;
  }

  bool run(uint64_t cycles) override {
    for (uint64_t i = 0; i < cycles && !finished; ++i) {
      if (!stepClock())
        return false;
    }
    return true;
  }

  bool runUntil(const SimTime &time) override {
    while (state.getTime() < time && !finished) {
      if (!stepClock())
        return false;
    }
    return true;
  }

  SimState &getState() override { return state; }
  const SimState &getState() const override { return state; }

  bool setSignal(StringRef path, const SignalValue &value) override {
    if (!state.hasSignal(path)) {
      lastError = "Signal not found: " + path.str();
      return false;
    }
    forcedSignals[path.str()] = value;
    state.setSignalValue(path, value);
    return true;
  }

  bool releaseSignal(StringRef path) override {
    auto it = forcedSignals.find(path.str());
    if (it != forcedSignals.end()) {
      forcedSignals.erase(it);
      return true;
    }
    return false;
  }

  bool isFinished() const override { return finished; }

  StringRef getLastError() const override { return lastError; }

private:
  DebugConfig config;
  SimState state;
  std::string lastError;
  bool finished = false;
  std::map<std::string, SignalValue> forcedSignals;
};

} // namespace

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Hide default LLVM options
  cl::HideUnrelatedOptions({&mainCategory, &circt::getResourceGuardCategory()});

  // Set the bug report message
  setBugReportMsg(circtBugReportMsg);

  // Print version info
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse command line
  cl::ParseCommandLineOptions(
      argc, argv,
      "CIRCT Debug - Interactive Hardware Debugger\n\n"
      "This tool provides a GDB-like interface for debugging hardware designs.\n"
      "It allows stepping through simulations, setting breakpoints on signals\n"
      "or source locations, and inspecting signal values.\n\n"
      "Example usage:\n"
      "  circt-debug design.mlir --top myModule\n"
      "  circt-debug design.mlir --vcd trace.vcd\n"
      "  circt-debug design.mlir -x commands.txt --batch\n");
  circt::installResourceGuard();

  // Create debug configuration
  DebugConfig config;
  config.designFile = inputFilename;
  config.topModule = topModule;
  config.outputDir = outputDir;
  config.vcdFile = vcdFile;
  config.maxCycles = maxCycles;
  config.verbose = verbose;

  // Create simulation backend (mock for now)
  auto backend = std::make_unique<MockSimulationBackend>();

  // Create debug session
  DebugSession session(std::move(backend), config);

  // Run in DAP mode if requested
  if (dapMode) {
    dap::DAPServer server(session);
    return server.run();
  }

  // Create interactive CLI
  InteractiveCLI cli(session);

  // If command file specified, read and execute commands
  if (!commandFile.empty()) {
    std::ifstream cmdStream(commandFile);
    if (!cmdStream) {
      llvm::errs() << "Error: Cannot open command file: " << commandFile << "\n";
      return 1;
    }
    cli.setInput(cmdStream);

    if (batch) {
      // In batch mode, don't prompt
      return cli.run();
    }
  }

  // Run interactive session
  return cli.run();
}
