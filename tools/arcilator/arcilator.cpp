//===- arcilator.cpp - An experimental circuit simulator ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'arcilator' compiler, which converts HW designs into
// a corresponding LLVM-based software model.
//
//===----------------------------------------------------------------------===//

#include "BehavioralLowering.h"
#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "circt/Tools/arcilator/pipelines.h"
#include "circt/Runtime/MooreRuntime.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef ARCILATOR_ENABLE_JIT
#define ARC_RUNTIME_JITBIND_FNDECL
#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/JITBind.h"
#endif

#include <optional>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>

using namespace mlir;
using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory mainCategory("arcilator Options");

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"),
                                                llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observePorts("observe-ports", llvm::cl::desc("Make all ports observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeWires("observe-wires", llvm::cl::desc("Make all wires observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> observeNamedValues(
    "observe-named-values",
    llvm::cl::desc("Make values with `sv.namehint` observable"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeRegisters("observe-registers",
                     llvm::cl::desc("Make all registers observable"),
                     llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeMemories("observe-memories",
                    llvm::cl::desc("Make all memory contents observable"),
                    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> stateFile("state-file",
                                            llvm::cl::desc("State file"),
                                            llvm::cl::value_desc("filename"),
                                            llvm::cl::init(""),
                                            llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldInline("inline", llvm::cl::desc("Inline arcs"),
                                        llvm::cl::init(true),
                                        llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDedup("dedup",
                                       llvm::cl::desc("Deduplicate arcs"),
                                       llvm::cl::init(true),
                                       llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectEnables(
    "detect-enables",
    llvm::cl::desc("Infer enable conditions for states to avoid computation"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectResets(
    "detect-resets",
    llvm::cl::desc("Infer reset conditions for states to avoid computation"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    shouldMakeLUTs("lookup-tables",
                   llvm::cl::desc("Optimize arcs into lookup tables"),
                   llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    printDebugInfo("print-debug-info",
                   llvm::cl::desc("Print debug information"),
                   llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verbosePassExecutions(
    "verbose-pass-executions",
    llvm::cl::desc("Log executions of toplevel module passes"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<unsigned> splitFuncsThreshold(
    "split-funcs-threshold",
    llvm::cl::desc(
        "Split large MLIR functions that occur above the given size threshold"),
    llvm::cl::ValueOptional, llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> asyncResetsAsSync(
    "async-resets-as-sync",
    llvm::cl::desc("Treat asynchronous firreg resets as synchronous"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

// Options to control early-out from pipeline.
enum Until {
  UntilPreprocessing,
  UntilArcConversion,
  UntilArcOpt,
  UntilStateLowering,
  UntilStateAlloc,
  UntilLLVMLowering,
  UntilEnd
};
static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilPreprocessing, "preproc", "Input preprocessing"),
    clEnumValN(UntilArcConversion, "arc-conv", "Conversion of modules to arcs"),
    clEnumValN(UntilArcOpt, "arc-opt", "Arc optimizations"),
    clEnumValN(UntilStateLowering, "state-lowering", "Stateful arc lowering"),
    clEnumValN(UntilStateAlloc, "state-alloc", "State allocation"),
    clEnumValN(UntilLLVMLowering, "llvm-lowering", "Lowering to LLVM"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));
static llvm::cl::opt<Until> runUntilBefore(
    "until-before", llvm::cl::desc("Stop pipeline before a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));
static llvm::cl::opt<Until> runUntilAfter(
    "until-after", llvm::cl::desc("Stop pipeline after a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));

// Options to control the output format.
enum OutputFormat { OutputMLIR, OutputLLVM, OutputRunJIT, OutputDisabled };
static llvm::cl::opt<OutputFormat> outputFormat(
    llvm::cl::desc("Specify output format"),
    llvm::cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit MLIR dialects"),
                     clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
                     clEnumValN(OutputRunJIT, "run",
                                "Run the simulation and emit its output"),
                     clEnumValN(OutputDisabled, "disable-output",
                                "Do not output anything")),
    llvm::cl::init(OutputLLVM), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    jitEntryPoint("jit-entry",
                  llvm::cl::desc("Name of the function containing the "
                                 "simulation to run when output is set to run"),
                  llvm::cl::init("entry"), llvm::cl::cat(mainCategory));

static llvm::cl::list<std::string> sharedLibs{
    "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
    llvm::cl::MiscFlags::CommaSeparated, llvm::cl::cat(mainCategory)};

static llvm::cl::list<std::string>
    jitArgs("args",
            llvm::cl::desc("Arguments to pass to the JIT entry function"),
            llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
            llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    behavioral("behavioral",
               llvm::cl::desc("Use behavioral pipeline for LLHD process-based "
                              "designs (UVM/testbench support)"),
               llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    noRuntime("no-runtime",
              llvm::cl::desc("Don't emit calls to the runtime library"),
              llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> noJitRuntime(
    "no-jit-runtime",
    llvm::cl::desc("Don't bind the statically linked JIT Runtime Library"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> extraRuntimeArgs(
    "extra-runtime-args",
    llvm::cl::desc(
        "Extra arguments passed to the runtime library for JIT runs."),
    llvm::cl::init(""), llvm::cl::cat(mainCategory));

static llvm::cl::opt<int> jitOptLevel(
    "jit-opt-level",
    llvm::cl::desc("JIT optimization level (0-3). "
                   "Default: 0 in --behavioral mode, 3 otherwise."),
    llvm::cl::init(-1), llvm::cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

static bool shouldLogProgress() {
  static bool enabled = []() {
    const char *env = std::getenv("ARCILATOR_PROGRESS_LOG");
    return env && *env && std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

static bool shouldStripGlobalCtors() {
  static bool enabled = []() {
    const char *env = std::getenv("ARCILATOR_STRIP_GLOBAL_CTORS");
    return env && *env && std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

static void logProgress(const llvm::Twine &msg) {
  if (!shouldLogProgress())
    return;
  using Clock = std::chrono::steady_clock;
  static Clock::time_point start = Clock::now();
  auto now = Clock::now();
  auto elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
          .count();
  llvm::errs() << "[arcilator-progress +" << elapsedMs << "ms] " << msg
               << "\n";
}

static unsigned stripLLVMGlobalCtors(ModuleOp module) {
  SmallVector<LLVM::GlobalCtorsOp> ctors;
  module.walk([&](LLVM::GlobalCtorsOp op) { ctors.push_back(op); });
  for (auto op : ctors)
    op.erase();
  return ctors.size();
}

static unsigned countDefinedLLVMFunctions(ModuleOp module) {
  unsigned count = 0;
  module.walk([&](LLVM::LLVMFuncOp fn) {
    if (!fn.isExternal())
      ++count;
  });
  return count;
}

static LogicalResult pruneModuleToEntry(ModuleOp module, StringRef entrySymbol) {
  auto *ctx = module.getContext();
  module.walk([&](Operation *op) {
    auto sym = dyn_cast<SymbolOpInterface>(op);
    if (!sym)
      return;
    if (sym.getName() == entrySymbol)
      return;
    if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op); fn && fn.isExternal())
      return;
    op->setAttr("sym_visibility", StringAttr::get(ctx, "private"));
  });

  PassManager pm(ctx);
  pm.enableVerifier(false);
  pm.addPass(createSymbolDCEPass());
  return pm.run(module);
}

// Force the linker to keep the Moore runtime object file in this executable.
// This exposes __moore_* and uvm_* symbols to the JIT symbol resolver.
extern "C" {
extern int32_t __moore_string_len(MooreString *str);
extern int32_t uvm_hdl_check_path(MooreString *path);
}
// NOLINTBEGIN(cert-dcl50-cpp)
[[maybe_unused]] static volatile void *mooreRuntimeSymbolAnchors[] = {
    reinterpret_cast<void *>(&__moore_string_len),
    reinterpret_cast<void *>(&uvm_hdl_check_path),
};
// NOLINTEND(cert-dcl50-cpp)

// Minimal scheduler shims used by behavioral lowering in arcilator.
// These provide pointer-backed signal storage for llhd.sig/prb/drv so the
// JIT can execute without circt-sim's ProcessScheduler.
namespace {
static std::mutex behavioralSignalMutex;
static llvm::DenseMap<void *, size_t> behavioralSignalBytes;
} // namespace

extern "C" int64_t __arc_sched_current_time() { return 0; }

extern "C" void *__arc_sched_create_signal(void *initPtr, int64_t sizeBytes) {
  size_t allocBytes = sizeBytes > 0 ? static_cast<size_t>(sizeBytes) : 1;
  void *storage = std::malloc(allocBytes);
  if (!storage)
    return nullptr;
  if (initPtr && allocBytes)
    std::memcpy(storage, initPtr, allocBytes);
  else if (allocBytes)
    std::memset(storage, 0, allocBytes);

  std::lock_guard<std::mutex> guard(behavioralSignalMutex);
  behavioralSignalBytes[storage] = allocBytes;
  return storage;
}

extern "C" void *__arc_sched_read_signal(void *signalHandle) {
  return signalHandle;
}

extern "C" void __arc_sched_drive_signal(void *signalHandle, void *valuePtr,
                                         int64_t /*delayEncoded*/,
                                         int8_t enable) {
  if (!enable || !signalHandle || !valuePtr)
    return;
  std::lock_guard<std::mutex> guard(behavioralSignalMutex);
  auto it = behavioralSignalBytes.find(signalHandle);
  if (it == behavioralSignalBytes.end())
    return;
  std::memcpy(signalHandle, valuePtr, it->second);
}

// Minimal Moore behavioral shims that are still emitted as external calls by
// imported UVM codepaths.
extern "C" void __moore_delay(int64_t /*delayValue*/) {}

extern "C" void __moore_class_srandom(void * /*classPtr*/, int32_t /*seed*/) {}

#ifdef ARCILATOR_ENABLE_JIT

// Manually bind the IR API of the ArcRuntime to the JIT execution engine

template <typename PtrTy>
static void bindExecutionEngineSymbol(llvm::orc::SymbolMap &symbolMap,
                                      llvm::orc::MangleAndInterner &interner,
                                      StringRef symName, PtrTy symTarget) {
  symbolMap[interner(symName)] = {llvm::orc::ExecutorAddr::fromPtr(symTarget),
                                  llvm::JITSymbolFlags::Exported};
}

static void bindArcRuntimeSymbols(ExecutionEngine &executionEngine) {
  auto &runtimeCallbacks = runtime::getArcRuntimeAPICallbacks();
  executionEngine.registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    bindExecutionEngineSymbol(symbolMap, interner,
                              runtimeCallbacks.symNameAllocInstance,
                              runtimeCallbacks.fnAllocInstance);
    bindExecutionEngineSymbol(symbolMap, interner,
                              runtimeCallbacks.symNameDeleteInstance,
                              runtimeCallbacks.fnDeleteInstance);
    bindExecutionEngineSymbol(symbolMap, interner,
                              runtimeCallbacks.symNameOnEval,
                              runtimeCallbacks.fnOnEval);
    return symbolMap;
  });
}

static void bindBehavioralRuntimeSymbols(ExecutionEngine &executionEngine,
                                         ModuleOp module) {
  executionEngine.registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    auto tryBind = [&](StringRef name) {
      if (name.empty())
        return;
      auto sym = interner(name);
      if (symbolMap.count(sym))
        return;
      void *target =
          llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name.str().c_str());
      if (!target)
        return;
      bindExecutionEngineSymbol(symbolMap, interner, name, target);
    };

    bindExecutionEngineSymbol(symbolMap, interner, "__arc_sched_current_time",
                              &__arc_sched_current_time);
    bindExecutionEngineSymbol(symbolMap, interner, "__arc_sched_create_signal",
                              &__arc_sched_create_signal);
    bindExecutionEngineSymbol(symbolMap, interner, "__arc_sched_read_signal",
                              &__arc_sched_read_signal);
    bindExecutionEngineSymbol(symbolMap, interner, "__arc_sched_drive_signal",
                              &__arc_sched_drive_signal);
    bindExecutionEngineSymbol(symbolMap, interner, "__moore_delay",
                              &__moore_delay);
    bindExecutionEngineSymbol(symbolMap, interner, "__moore_class_srandom",
                              &__moore_class_srandom);

    // Resolve behavioral runtime externs from the host process.
    // This includes Moore/UVM helper APIs emitted as external LLVM funcs.
    for (auto fn : module.getOps<LLVM::LLVMFuncOp>()) {
      if (!fn.isExternal())
        continue;
      StringRef name = fn.getName();
      if (name.starts_with("__moore_") || name.starts_with("uvm_"))
        tryBind(name);
    }
    return symbolMap;
  });
}
#endif

/// Populate a pass manager with the arc simulator pipeline for the given
/// command line options. This pipeline lowers modules to the Arc dialect.
static void populateHwModuleToArcPipeline(PassManager &pm) {
  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator"));

  // Pre-process the input such that it no longer contains any SV dialect ops
  // and external modules that are relevant to the arc transformation are
  // represented as intrinsic ops.
  if (untilReached(UntilPreprocessing))
    return;

  ArcPreprocessingOptions preprocessingOpt;
  preprocessingOpt.observePorts = observePorts;
  preprocessingOpt.observeWires = observeWires;
  preprocessingOpt.observeNamedValues = observeNamedValues;
  preprocessingOpt.observeMemories = observeMemories;
  preprocessingOpt.asyncResetsAsSync = asyncResetsAsSync;
  populateArcPreprocessingPipeline(pm, preprocessingOpt);

  // Restructure the input from a `hw.module` hierarchy to a collection of arcs.
  if (untilReached(UntilArcConversion))
    return;

  ArcConversionOptions conversionOpt;
  conversionOpt.observeRegisters = observeRegisters;
  conversionOpt.shouldDedup = shouldDedup;
  populateArcConversionPipeline(pm, conversionOpt);

  // Perform arc-level optimizations that are not specific to software
  // simulation.
  if (untilReached(UntilArcOpt))
    return;

  ArcOptimizationOptions optimizationOpt;
  optimizationOpt.shouldDetectEnables = shouldDetectEnables;
  optimizationOpt.shouldDetectResets = shouldDetectResets;
  optimizationOpt.shouldMakeLUTs = shouldMakeLUTs;
  populateArcOptimizationPipeline(pm, optimizationOpt);

  // Lower stateful arcs into explicit state reads and writes.
  if (untilReached(UntilStateLowering))
    return;

  ArcStateLoweringOptions loweringOpt;
  loweringOpt.shouldInline = shouldInline;
  populateArcStateLoweringPipeline(pm, loweringOpt);

  // Allocate states.
  if (untilReached(UntilStateAlloc))
    return;

  ArcStateAllocationOptions allocationOpt;
  allocationOpt.splitFuncsThreshold = splitFuncsThreshold;
  populateArcStateAllocationPipeline(pm, allocationOpt);
}

/// Populate a pass manager with the behavioral simulation pipeline.
/// This pipeline still bypasses Arc conversion, but first runs the LLHD/Moore
/// cleanup sequence used by import flows so residual process/event-control ops
/// in helper functions are normalized before final LLVM lowering.
static void populateBehavioralToLLVMPipeline(PassManager &pm) {
  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator-behavioral"));

  // First, canonicalize to clean up the IR.
  pm.addPass(createCanonicalizerPass());

  // Re-run the LLHD/Moore cleanup sequence to normalize any residual
  // moore.wait_event / llhd.process structures that survived import.
  pm.addNestedPass<hw::HWModuleOp>(llhd::createWrapProceduralOpsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(llhd::createInlineCallsPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createConvertMooreToCorePass());
  pm.addPass(createSymbolDCEPass());

  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(llhd::createMem2RegPass());
  modulePM.addPass(llhd::createHoistSignalsPass());
  modulePM.addPass(llhd::createDeseqPass());
  modulePM.addPass(llhd::createLowerProcessesPass());
  modulePM.addPass(createCSEPass());
  modulePM.addPass(createBottomUpSimpleCanonicalizerPass());

  modulePM.addPass(llhd::createUnrollLoopsPass());
  modulePM.addPass(createCSEPass());
  modulePM.addPass(createBottomUpSimpleCanonicalizerPass());

  modulePM.addPass(llhd::createRemoveControlFlowPass());
  modulePM.addPass(createCSEPass());
  modulePM.addPass(createBottomUpSimpleCanonicalizerPass());
  modulePM.addPass(createMapArithToCombPass(true));

  modulePM.addPass(llhd::createCombineDrivesPass());
  modulePM.addPass(llhd::createSig2Reg());
  modulePM.addPass(createCSEPass());
  modulePM.addPass(createBottomUpSimpleCanonicalizerPass());

  // Lower behavioral LLHD/Moore/Sim ops to LLVM IR with runtime calls.
  pm.addPass(createLowerBehavioralToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
}

/// Normalize LLVM global initializers so LLVM IR translation does not crash on
/// pointer/aggregate globals initialized with integer zero attributes.
static void sanitizeLLVMGlobalInitializers(ModuleOp module) {
  module.walk([&](LLVM::GlobalOp globalOp) {
    auto intInit = llvm::dyn_cast_or_null<IntegerAttr>(globalOp.getValueAttr());
    if (!intInit || !intInit.getValue().isZero())
      return;
    if (llvm::isa<IntegerType>(globalOp.getGlobalType()))
      return;
    globalOp.setValueAttr(LLVM::ZeroAttr::get(module.getContext()));
  });
}

static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    logProgress("parse-start");
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    logProgress("parse-done");
  }
  if (!module)
    return failure();

  if (behavioral) {
    // Behavioral pipeline: lower LLHD/Moore/Sim directly to LLVM.
    PassManager pmBehavioral(&context);
    pmBehavioral.enableVerifier(verifyPasses);
    pmBehavioral.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pmBehavioral)))
      return failure();
    populateBehavioralToLLVMPipeline(pmBehavioral);
    logProgress("behavioral-pipeline-start");
    if (failed(pmBehavioral.run(module.get())))
      return failure();
    logProgress("behavioral-pipeline-done");
  } else {
    // Standard Arc pipeline: lower HwModule to Arc model.
    PassManager pmArc(&context);
    pmArc.enableVerifier(verifyPasses);
    pmArc.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pmArc)))
      return failure();
    populateHwModuleToArcPipeline(pmArc);
    logProgress("arc-pipeline-start");
    if (failed(pmArc.run(module.get())))
      return failure();
    logProgress("arc-pipeline-done");

    // Output state info as JSON if requested.
    if (!stateFile.empty() && !untilReached(UntilStateLowering)) {
      std::error_code ec;
      llvm::ToolOutputFile outputFile(stateFile, ec,
                                      llvm::sys::fs::OpenFlags::OF_None);
      if (ec) {
        llvm::errs() << "unable to open state file: " << ec.message() << '\n';
        return failure();
      }
      if (failed(collectAndExportModelInfo(module.get(), outputFile.os()))) {
        llvm::errs() << "failed to collect model info\n";
        return failure();
      }

      outputFile.keep();
    }

    // Lower Arc model to LLVM IR.
    PassManager pmLlvm(&context);
    pmLlvm.enableVerifier(verifyPasses);
    pmLlvm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pmLlvm)))
      return failure();
    if (verbosePassExecutions)
      pmLlvm.addInstrumentation(
          std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
              "arcilator"));

    if (!untilReached(UntilLLVMLowering)) {
      populateArcToLLVMPipeline(pmLlvm, !noRuntime, extraRuntimeArgs);
    }

    if (printDebugInfo && outputFormat == OutputLLVM)
      pmLlvm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
    logProgress("llvm-pipeline-start");
    if (failed(pmLlvm.run(module.get())))
      return failure();
    logProgress("llvm-pipeline-done");
  }

  sanitizeLLVMGlobalInitializers(module.get());
  logProgress("sanitize-globals-done");
  if (behavioral && shouldStripGlobalCtors()) {
    unsigned erased = stripLLVMGlobalCtors(module.get());
    if (erased > 0)
      logProgress(llvm::Twine("strip-global-ctors-done count=") +
                  llvm::Twine(erased));
  }

#ifdef ARCILATOR_ENABLE_JIT
  // Handle JIT execution.
  if (outputFormat == OutputRunJIT) {
    auto tsJit = ts.nest("JIT");
    if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd) {
      llvm::errs() << "full pipeline must be run for JIT execution\n";
      return failure();
    }

    std::string selectedEntryPoint = jitEntryPoint;
    Operation *toCall = module->lookupSymbol(selectedEntryPoint);
    if (!toCall && selectedEntryPoint == "entry") {
      auto pickFallback = [&](auto pred) -> Operation * {
        for (auto fn : module->getOps<LLVM::LLVMFuncOp>()) {
          if (fn.isExternal() || fn.getNumArguments() != 0)
            continue;
          auto visibility = fn.getSymVisibility();
          if (visibility && *visibility == "private")
            continue;
          StringRef name = fn.getName();
          std::string lower = llvm::StringRef(name).lower();
          if (pred(name, lower))
            return fn.getOperation();
        }
        return nullptr;
      };

      for (StringRef exact :
           {"hvl_top", "hdl_top", "HvlTop", "HdlTop", "hvlTop", "hdlTop"}) {
        if ((toCall = module->lookupSymbol(exact))) {
          selectedEntryPoint = exact.str();
          break;
        }
      }
      if (!toCall)
        toCall = pickFallback([](StringRef, StringRef lower) {
          return lower.contains("hvl") && lower.contains("top");
        });
      if (!toCall)
        toCall = pickFallback([](StringRef, StringRef lower) {
          return lower.contains("hdl") && lower.contains("top");
        });
      if (!toCall)
        toCall = pickFallback(
            [](StringRef, StringRef lower) { return lower.contains("top"); });
      if (toCall)
        selectedEntryPoint = cast<SymbolOpInterface>(toCall).getName().str();
    }
    if (!toCall) {
      llvm::errs() << "entry point not found: '" << jitEntryPoint << "'\n";
      return failure();
    }

    auto toCallFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(toCall);
    if (!toCallFunc) {
      llvm::errs() << "entry point '" << selectedEntryPoint
                   << "' was found but on an operation of type '"
                   << toCall->getName()
                   << "' while an LLVM function was expected\n";
      return failure();
    }

    unsigned numArgs = toCallFunc.getNumArguments();
    if (numArgs) {
      if (jitArgs.size() % numArgs != 0) {
          llvm::errs() << "entry point '" << selectedEntryPoint << "' has "
                       << numArgs
                     << " arguments, but provided " << jitArgs.size()
                     << " arguments (not a multiple)\n";
        return failure();
      }
      if (jitArgs.empty()) {
        llvm::errs() << "entry point '" << selectedEntryPoint
                     << "' must have no arguments\n";
        return failure();
      }
    } else if (!jitArgs.empty()) {
      llvm::errs() << "entry point '" << selectedEntryPoint
                   << "' has no arguments, but provided " << jitArgs.size()
                   << "arguments\n";
      return failure();
    }

    if (behavioral) {
      unsigned beforeFuncs = countDefinedLLVMFunctions(module.get());
      logProgress(llvm::Twine("jit-prune-start entry=") +
                  llvm::Twine(selectedEntryPoint) + llvm::Twine(" funcs=") +
                  llvm::Twine(beforeFuncs));
      if (failed(pruneModuleToEntry(module.get(), selectedEntryPoint))) {
        llvm::errs() << "failed to prune module symbols for entry point '"
                     << selectedEntryPoint << "'\n";
        return failure();
      }
      unsigned afterFuncs = countDefinedLLVMFunctions(module.get());
      logProgress(llvm::Twine("jit-prune-done funcs=") + llvm::Twine(afterFuncs));
    }

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());

    int effectiveJitOptLevel = jitOptLevel;
    if (effectiveJitOptLevel < 0)
      effectiveJitOptLevel = behavioral ? 0 : 3;
    if (effectiveJitOptLevel > 3) {
      llvm::errs() << "invalid --jit-opt-level: " << effectiveJitOptLevel
                   << " (expected 0..3)\n";
      return failure();
    }

    llvm::CodeGenOptLevel codeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    switch (effectiveJitOptLevel) {
    case 0:
      codeGenOptLevel = llvm::CodeGenOptLevel::None;
      break;
    case 1:
      codeGenOptLevel = llvm::CodeGenOptLevel::Less;
      break;
    case 2:
      codeGenOptLevel = llvm::CodeGenOptLevel::Default;
      break;
    case 3:
      codeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
      break;
    default:
      llvm_unreachable("effectiveJitOptLevel must be in [0, 3]");
    }

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = codeGenOptLevel;
    std::function<llvm::Error(llvm::Module *)> transformer =
        mlir::makeOptimizingTransformer(
            /*optLevel=*/effectiveJitOptLevel, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    engineOptions.transformer = transformer;
    engineOptions.sharedLibPaths = sharedLibraries;

    auto tsCompile = tsJit.nest("Compile");
    logProgress("jit-create-start");
    auto executionEngine =
        mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (!executionEngine) {
      llvm::handleAllErrors(
          executionEngine.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to create execution engine: "
                         << info.message() << "\n";
          });
      return failure();
    }
    logProgress("jit-create-done");

    if (!noJitRuntime)
      bindArcRuntimeSymbols(**executionEngine);
    if (behavioral)
      bindBehavioralRuntimeSymbols(**executionEngine, module.get());
    logProgress("jit-bind-symbols-done");

    logProgress("jit-lookup-start");
    auto expectedFunc = (*executionEngine)->lookupPacked(selectedEntryPoint);
    if (!expectedFunc) {
      llvm::handleAllErrors(
          expectedFunc.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to run simulation: " << info.message()
                         << "\n";
          });
      return failure();
    }
    logProgress("jit-lookup-done");
    tsCompile.stop();

    auto tsExecute = tsJit.nest("Execute");
    void (*simulationFunc)(void **) = *expectedFunc;

    for (unsigned i = 0, e = jitArgs.size(); i < e; i += numArgs) {
      std::vector<std::vector<uint64_t>> argsStorage;
      SmallVector<void *> args;
      argsStorage.reserve(numArgs);
      args.reserve(numArgs);

      // Repeated args are concatenated, so break apart in groups of multiples
      // of args.
      for (auto [val, arg] :
           llvm::zip(llvm::make_range(jitArgs.begin() + i,
                                      jitArgs.begin() + i + numArgs),
                     toCallFunc.getArguments())) {
        auto type = arg.getType();
        if (!type.isIntOrIndex()) {
          llvm::errs() << "argument " << arg.getArgNumber()
                       << " of entry point '" << selectedEntryPoint
                       << "' is not an integer or index type\n";
          return failure();
        }

        // TODO: This should probably be checking if DLTI is set on module.
        unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
        APInt apVal(width, 0);
        if (StringRef(val).getAsInteger(0, apVal)) {
          llvm::errs() << "invalid integer argument: '" << val << "'\n";
          return failure();
        }
        if (apVal.getBitWidth() > width) {
          llvm::errs() << "integer argument '" << val << "' (required width "
                       << apVal.getBitWidth() << ") is too large for type '"
                       << type << "'\n";
          return failure();
        }

        std::vector<uint64_t> argData;
        unsigned numWords = apVal.getNumWords();
        argData.resize(numWords);
        const uint64_t *rawData = apVal.getRawData();
        for (unsigned j = 0; j < numWords; ++j)
          argData[j] = rawData[j];

        argsStorage.push_back(std::move(argData));
        args.push_back(argsStorage.back().data());
      }

      (*simulationFunc)(args.data());
    }
    // Handle the case without arguments as before.
    if (jitArgs.empty())
      (*simulationFunc)(nullptr);
    logProgress("jit-execute-done");

    return success();
  }
#endif // ARCILATOR_ENABLE_JIT

  // Handle MLIR output.
  if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd ||
      outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    // Only set the debug info flag to true in order to not overwrite MLIR
    // printer CLI flags when the custom debug info option is not set.
    if (printDebugInfo)
      printingFlags.enableDebugInfo(printDebugInfo);
    auto outputTimer = ts.nest("Print MLIR output");
    module->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  // Handle LLVM output.
  if (outputFormat == OutputLLVM) {
    auto outputTimer = ts.nest("Print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    return success();
  }

  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<llvm::MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeArcilator(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    arc::ArcDialect,
    comb::CombDialect,
    emit::EmitDialect,
    hw::HWDialect,
    llhd::LLHDDialect,
    moore::MooreDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::DLTIDialect,
    mlir::func::FuncDialect,
    mlir::index::IndexDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect,
    om::OMDialect,
    seq::SeqDialect,
    sim::SimDialect,
    sv::SVDialect,
    verif::VerifDialect
  >();
  // clang-format on

  arc::initAllExternalInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

/// Main driver for the command. This sets up LLVM and MLIR, and parses command
/// line options before passing off to 'executeArcilator'. This is set up so we
/// can `exit(0)` at the end of the program to avoid teardown of the MLIRContext
/// and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  llvm::cl::HideUnrelatedOptions(
      {&mainCategory, &circt::getResourceGuardCategory()});

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();

    // Dialect passes:
    arc::registerPasses();
    registerConvertToArcsPass();
    registerLowerArcToLLVMPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  llvm::cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-based circuit simulator\n");

  circt::installResourceGuard();

  if (outputFormat == OutputRunJIT) {
#ifdef ARCILATOR_ENABLE_JIT
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
#else
    llvm::errs() << "This arcilator binary was not built with JIT support.\n";
    llvm::errs() << "To enable JIT features, build arcilator with MLIR's "
                    "execution engine.\n";
    llvm::errs() << "This can be achieved by building arcilator with the "
                    "host's LLVM target enabled.\n";
    exit(1);
#endif // ARCILATOR_ENABLE_JIT
  }

  MLIRContext context;
  auto result = executeArcilator(context);

  // Use "exit" instead of returning to signal completion. This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
