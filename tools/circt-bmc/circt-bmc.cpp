//===- circt-bmc.cpp - The circt-bmc bounded model checker ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-bmc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/ImportVerilog.h"
#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/SVAToLTL.h"
#include "circt/Conversion/VerifToSMT.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/SMTModel.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <atomic>
#include <map>

#ifdef CIRCT_BMC_ENABLE_JIT
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

namespace {
struct ResourceGuardPassInstrumentation final : public PassInstrumentation {
  void runBeforePass(Pass *pass, Operation *op) override {
    StringRef arg = pass->getArgument();
    StringRef label = arg.empty() ? pass->getName() : arg;
    StringRef opName = op ? op->getName().getStringRef() : StringRef("<null>");
    std::string combined = (label + "[" + opName + "]").str();
    circt::setResourceGuardPhase(combined);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-bmc Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module (or verif.formal op) to verify "
                        "properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<int> clockBound(
    "b", cl::Required,
    cl::desc("Specify a number of clock cycles to model check up to."),
    cl::value_desc("clock cycle count"), cl::cat(mainCategory));

static cl::opt<int> ignoreAssertionsUntil(
    "ignore-asserts-until", cl::Optional,
    cl::desc("Specify a number of initial clock cycles for which assertions "
             "should be ignored (e.g. so that a circuit can stabilize)."),
    cl::value_desc("number of cycles to ignore assertions for"),
    cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> printSolverOutput(
    "print-solver-output",
    cl::desc("Print the output (counterexample or proof) produced by the "
             "solver on each invocation and the assertion set that they "
             "prove/disprove."),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> printCounterexample(
    "print-counterexample",
    cl::desc("Print counterexample inputs for SAT/UNKNOWN results when a model "
             "is available"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> assumeKnownInputs(
    "assume-known-inputs",
    cl::desc("Assume input values are known (not X) for BMC and LEC "
             "operations."),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> failOnViolation(
    "fail-on-violation",
    cl::desc("Return failure when the bounded model check finds a violation"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> kInduction(
    "k-induction",
    cl::desc("Run k-induction (SMT-LIB only): base check at -b, then "
             "induction step at -b+1 (deprecated alias for --induction)"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> induction(
    "induction",
    cl::desc("Run k-induction (SMT-LIB only): base check at -b, then "
             "induction step at -b+1"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> liveness(
    "liveness",
    cl::desc("Run bounded liveness checking using only bmc.final properties"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> livenessLasso(
    "liveness-lasso",
    cl::desc("Require a lasso-style loop constraint in liveness mode "
             "(SMT-LIB only)"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string>
    z3PathOpt("z3-path",
              cl::desc("Path to z3 binary for --run-smtlib (optional)"),
              cl::value_desc("path"), cl::init(""), cl::cat(mainCategory));

static std::atomic<int> bmcJitResult{-1};

extern "C" void circt_bmc_report_result(int result) {
  bmcJitResult.store(result ? 1 : 0, std::memory_order_relaxed);
}

static cl::opt<bool> risingClocksOnly(
    "rising-clocks-only",
    cl::desc("Only consider the circuit and property on rising clock edges"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> allowMultiClock(
    "allow-multi-clock",
    cl::desc("Allow multiple explicit clock inputs by interleaving toggles"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> pruneUnreachableSymbols(
    "prune-unreachable-symbols",
    cl::desc("Prune symbols not reachable from the entry module"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> pruneBMCRegisters(
    "prune-bmc-registers",
    cl::desc("Prune BMC state, inputs, and combinational logic that do not "
             "affect properties"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> flattenModules(
    "flatten-modules",
    cl::desc("Flatten all module instances before processing (which will "
             "increase code size but allows multiple assertions across module "
             "boundaries to be supported)"),
    cl::init(true), cl::cat(mainCategory));

#ifdef CIRCT_BMC_ENABLE_JIT

enum OutputFormat {
  OutputMLIR,
  OutputLLVM,
  OutputSMTLIB,
  OutputRunJIT,
  OutputRunSMTLIB
};
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit SMT-LIB file"),
               clEnumValN(OutputRunSMTLIB, "run-smtlib",
                          "Run SMT-LIB via z3"),
               clEnumValN(OutputRunJIT, "run",
                          "Perform BMC and output result")),
    cl::init(OutputRunJIT), cl::cat(mainCategory));

static cl::list<std::string> sharedLibs{
    "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
    cl::MiscFlags::CommaSeparated, llvm::cl::cat(mainCategory)};

#else

enum OutputFormat { OutputMLIR, OutputLLVM, OutputSMTLIB, OutputRunSMTLIB };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit SMT-LIB file"),
               clEnumValN(OutputRunSMTLIB, "run-smtlib",
                          "Run SMT-LIB via z3")),
    cl::init(OutputLLVM), cl::cat(mainCategory));

#endif

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

static std::optional<std::string> findResultToken(StringRef text) {
  StringRef remaining = text;
  std::optional<std::string> result;
  while (!remaining.empty()) {
    remaining = remaining.ltrim(" \t\r\n");
    if (remaining.empty())
      break;
    StringRef token =
        remaining.take_until([](char c) { return c == ' ' || c == '\t' ||
                                                 c == '\r' || c == '\n'; });
    if (token == "sat" || token == "unsat" || token == "unknown")
      result = token.str();
    remaining = remaining.drop_front(token.size());
  }
  return result;
}

static bool hasSMTSolver(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::smt::SolverOp) { found = true; });
  return found;
}

static std::optional<bool> decodeBoolModelValue(StringRef value) {
  value = value.trim();
  if (value.equals_insensitive("true"))
    return true;
  if (value.equals_insensitive("false"))
    return false;
  if (value == "#b1")
    return true;
  if (value == "#b0")
    return false;
  if (value == "1")
    return true;
  if (value == "0")
    return false;
  if (value.contains("'")) {
    if (value.ends_with("1"))
      return true;
    if (value.ends_with("0"))
      return false;
  }
  return std::nullopt;
}

using StepMap = std::map<unsigned, bool>;
using WaveTable = std::map<std::string, StepMap>;

static llvm::StringSet<> collectEventWaveNames(ModuleOp module) {
  llvm::StringSet<> names;
  auto collectFromDetails = [&](ArrayAttr detailSets) {
    if (!detailSets)
      return;
    for (Attribute detailSetAttr : detailSets) {
      auto detailSet = dyn_cast<ArrayAttr>(detailSetAttr);
      if (!detailSet)
        continue;
      for (Attribute detailAttr : detailSet) {
        auto detail = dyn_cast<DictionaryAttr>(detailAttr);
        if (!detail)
          continue;
        for (StringRef field :
             {"signal_name", "sequence_name", "iff_name", "witness_name"}) {
          if (auto nameAttr = dyn_cast_or_null<StringAttr>(detail.get(field)))
            names.insert(nameAttr.getValue());
        }
      }
    }
  };

  module.walk([&](mlir::smt::SolverOp solver) {
    collectFromDetails(
        solver->getAttrOfType<ArrayAttr>("bmc_event_source_details"));
  });
  collectFromDetails(
      module->getAttrOfType<ArrayAttr>("circt.bmc_event_source_details"));
  return names;
}

static WaveTable buildWaveTable(const llvm::StringMap<std::string> &modelValues,
                                unsigned &maxStep,
                                const llvm::StringSet<> *anchorNames = nullptr) {
  WaveTable table;
  maxStep = 0;
  llvm::StringSet<> boolValueNames;
  for (const auto &kv : modelValues) {
    auto decoded = decodeBoolModelValue(kv.getValue());
    if (!decoded)
      continue;
    boolValueNames.insert(kv.getKey());
  }
  for (const auto &kv : modelValues) {
    auto decoded = decodeBoolModelValue(kv.getValue());
    if (!decoded)
      continue;
    StringRef name = kv.getKey();
    std::string base = name.str();
    unsigned step = 0;
    bool shouldAnchorAsBase = anchorNames && anchorNames->contains(name);
    if (!shouldAnchorAsBase) {
      size_t underscore = name.rfind('_');
      if (underscore != StringRef::npos && underscore + 1 < name.size()) {
        StringRef suffix = name.drop_front(underscore + 1);
        unsigned suffixStep = 0;
        if (!suffix.getAsInteger(10, suffixStep)) {
          StringRef candidateBase = name.take_front(underscore);
          // Treat `_N` as a step suffix only when an unsuffixed base
          // declaration also exists. Anchor names from event metadata are never
          // split, which avoids ambiguity for symbols like `sig_1`.
          if (boolValueNames.contains(candidateBase)) {
            base = candidateBase.str();
            // We use step 0 for the unsuffixed declaration; `_0` is next step.
            step = suffixStep + 1;
          }
        }
      }
    }
    table[base][step] = *decoded;
    maxStep = std::max(maxStep, step);
  }
  return table;
}

static void printMixedEventSources(
    ModuleOp module,
    const llvm::StringMap<std::string> *modelValues = nullptr) {
  bool printedHeader = false;
  bool sawEventSources = false;
  bool hasWitnessActivity = false;
  unsigned maxStep = 0;
  WaveTable waves;
  llvm::StringSet<> anchorNames;
  if (modelValues) {
    anchorNames = collectEventWaveNames(module);
    waves = buildWaveTable(*modelValues, maxStep, &anchorNames);
  }
  bool printedActivityHeader = false;
  bool printedByStepHeader = false;
  auto processEventAttrs = [&](ArrayAttr eventSources, ArrayAttr detailSets) {
    if (!eventSources || eventSources.empty())
      return;
    sawEventSources = true;
    if (!printedHeader) {
      llvm::errs() << "mixed event sources:\n";
      printedHeader = true;
    }
    for (unsigned i = 0; i < eventSources.size(); ++i) {
      auto sourceSet = dyn_cast<ArrayAttr>(eventSources[i]);
      if (!sourceSet)
        continue;
      llvm::errs() << "  [" << i << "] ";
      bool first = true;
      auto detailSet =
          detailSets && i < detailSets.size()
              ? dyn_cast<ArrayAttr>(detailSets[i])
              : ArrayAttr{};
      for (unsigned j = 0; j < sourceSet.size(); ++j) {
        Attribute sourceAttr = sourceSet[j];
        auto source = dyn_cast<StringAttr>(sourceAttr);
        if (!source)
          continue;
        StringRef label = source.getValue();
        if (detailSet && j < detailSet.size())
          if (auto detail = dyn_cast<DictionaryAttr>(detailSet[j]))
            if (auto detailLabel =
                    dyn_cast_or_null<StringAttr>(detail.get("label")))
              label = detailLabel.getValue();
        if (!first)
          llvm::errs() << ", ";
        llvm::errs() << label;
        first = false;
      }
      llvm::errs() << "\n";

      if (!modelValues || !detailSet)
        continue;
      std::map<unsigned, SmallVector<std::string, 4>> activeArmsByStep;
      for (unsigned j = 0; j < detailSet.size(); ++j) {
        auto detail = dyn_cast<DictionaryAttr>(detailSet[j]);
        if (!detail)
          continue;
        auto kindAttr = dyn_cast_or_null<StringAttr>(detail.get("kind"));
        if (!kindAttr)
          continue;
        StringRef kind = kindAttr.getValue();
        bool useWitness = false;
        const StepMap *armWave = nullptr;
        StringRef signalEdge;
        if (auto witnessNameAttr =
                dyn_cast_or_null<StringAttr>(detail.get("witness_name"))) {
          auto witnessIt = waves.find(witnessNameAttr.getValue().str());
          if (witnessIt != waves.end()) {
            armWave = &witnessIt->second;
            useWitness = true;
            hasWitnessActivity = true;
          }
        }
        if (!armWave) {
          if (kind == "signal") {
            auto signalNameAttr =
                dyn_cast_or_null<StringAttr>(detail.get("signal_name"));
            auto edgeAttr = dyn_cast_or_null<StringAttr>(detail.get("edge"));
            if (!signalNameAttr || !edgeAttr)
              continue;
            auto signalIt = waves.find(signalNameAttr.getValue().str());
            if (signalIt == waves.end())
              continue;
            armWave = &signalIt->second;
            signalEdge = edgeAttr.getValue();
          } else if (kind == "sequence") {
            auto sequenceNameAttr =
                dyn_cast_or_null<StringAttr>(detail.get("sequence_name"));
            if (!sequenceNameAttr)
              continue;
            auto sequenceIt = waves.find(sequenceNameAttr.getValue().str());
            if (sequenceIt == waves.end())
              continue;
            armWave = &sequenceIt->second;
          } else {
            continue;
          }
        }
        const StepMap *iffWave = nullptr;
        if (!useWitness) {
          if (auto iffNameAttr =
                  dyn_cast_or_null<StringAttr>(detail.get("iff_name"))) {
            auto iffIt = waves.find(iffNameAttr.getValue().str());
            if (iffIt != waves.end())
              iffWave = &iffIt->second;
          }
        }
        StringRef label = kind;
        if (auto detailLabel =
                dyn_cast_or_null<StringAttr>(detail.get("label")))
          label = detailLabel.getValue();
        SmallVector<unsigned, 8> activeSteps;
        unsigned startStep = useWitness || kind == "sequence" ? 0 : 1;
        for (unsigned step = startStep; step <= maxStep; ++step) {
          bool armFired = false;
          if (useWitness) {
            auto currIt = armWave->find(step);
            armFired = currIt != armWave->end() && currIt->second;
          } else if (kind == "signal") {
            auto prevIt = armWave->find(step - 1);
            auto currIt = armWave->find(step);
            if (prevIt == armWave->end() || currIt == armWave->end())
              continue;
            bool prev = prevIt->second;
            bool curr = currIt->second;
            if (signalEdge == "posedge")
              armFired = !prev && curr;
            else if (signalEdge == "negedge")
              armFired = prev && !curr;
            else if (signalEdge == "both")
              armFired = prev != curr;
          } else if (kind == "sequence") {
            auto currIt = armWave->find(step);
            armFired = currIt != armWave->end() && currIt->second;
          }
          if (!armFired)
            continue;
          if (iffWave) {
            auto iffIt = iffWave->find(step);
            if (iffIt == iffWave->end() || !iffIt->second)
              continue;
          }
          activeSteps.push_back(step);
          activeArmsByStep[step].push_back(label.str());
        }
        if (!printedActivityHeader) {
          llvm::errs() << (hasWitnessActivity ? "\nevent-arm activity:\n"
                                              : "\nestimated event-arm "
                                                "activity:\n");
          printedActivityHeader = true;
        }
        llvm::errs() << "  [" << i << "][" << j << "] " << label << " ->";
        if (activeSteps.empty()) {
          llvm::errs() << " none\n";
        } else {
          bool firstStep = true;
          for (unsigned step : activeSteps) {
            if (!firstStep)
              llvm::errs() << ",";
            llvm::errs() << " step " << step;
            firstStep = false;
          }
          llvm::errs() << "\n";
        }
      }
      if (!activeArmsByStep.empty()) {
        if (!printedByStepHeader) {
          llvm::errs() << (hasWitnessActivity ? "\nfired arms by step:\n"
                                              : "\nestimated fired arms by "
                                                "step:\n");
          printedByStepHeader = true;
        }
        for (const auto &stepAndArms : activeArmsByStep) {
          llvm::errs() << "  [" << i << "] step " << stepAndArms.first
                       << " -> ";
          bool firstArm = true;
          for (const auto &arm : stepAndArms.second) {
            if (!firstArm)
              llvm::errs() << ", ";
            llvm::errs() << arm;
            firstArm = false;
          }
          llvm::errs() << "\n";
        }
      }
    }
  };

  module.walk([&](mlir::smt::SolverOp solver) {
    auto eventSources = solver->getAttrOfType<ArrayAttr>("bmc_event_sources");
    if (!eventSources)
      eventSources =
          solver->getAttrOfType<ArrayAttr>("bmc_mixed_event_sources");
    auto detailSets =
        solver->getAttrOfType<ArrayAttr>("bmc_event_source_details");
    processEventAttrs(eventSources, detailSets);
  });

  if (sawEventSources)
    return;
  auto eventSources =
      module->getAttrOfType<ArrayAttr>("circt.bmc_event_sources");
  if (!eventSources)
    eventSources = module->getAttrOfType<ArrayAttr>("circt.bmc_mixed_event_sources");
  auto detailSets =
      module->getAttrOfType<ArrayAttr>("circt.bmc_event_source_details");
  processEventAttrs(eventSources, detailSets);
}

static bool parseDefineFun(StringRef text, size_t &pos, StringRef &name,
                           StringRef &value) {
  auto skipWs = [&](size_t &p) {
    while (p < text.size() &&
           (text[p] == ' ' || text[p] == '\t' || text[p] == '\r' ||
            text[p] == '\n'))
      ++p;
  };
  auto parseToken = [&](size_t &p, StringRef &out) -> bool {
    skipWs(p);
    if (p >= text.size())
      return false;
    size_t start = p;
    if (text[p] == '|') {
      ++p;
      size_t end = text.find('|', p);
      if (end == StringRef::npos)
        return false;
      p = end + 1;
      out = text.slice(start, p);
      return true;
    }
    while (p < text.size()) {
      char c = text[p];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '(' ||
          c == ')')
        break;
      ++p;
    }
    if (p == start)
      return false;
    out = text.slice(start, p);
    return true;
  };
  auto parseBalanced = [&](size_t &p, StringRef &out) -> bool {
    skipWs(p);
    if (p >= text.size())
      return false;
    if (text[p] != '(')
      return parseToken(p, out);
    size_t start = p;
    int depth = 0;
    while (p < text.size()) {
      char c = text[p++];
      if (c == '(')
        ++depth;
      else if (c == ')') {
        --depth;
        if (depth == 0) {
          out = text.slice(start, p);
          return true;
        }
      }
    }
    return false;
  };

  size_t p = pos;
  skipWs(p);
  if (!text.substr(p).starts_with("(define-fun"))
    return false;
  p += StringRef("(define-fun").size();
  if (!parseToken(p, name))
    return false;

  // Skip args and result type.
  StringRef ignored;
  if (!parseBalanced(p, ignored))
    return false;
  if (!parseBalanced(p, ignored))
    return false;

  if (!parseBalanced(p, value))
    return false;

  // Consume trailing ')'.
  skipWs(p);
  if (p >= text.size() || text[p] != ')')
    return false;
  ++p;

  pos = p;
  return true;
}

static llvm::StringMap<std::string> parseZ3Model(StringRef text) {
  llvm::StringMap<std::string> values;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t start = text.find("(define-fun", pos);
    if (start == StringRef::npos)
      break;
    size_t parsePos = start;
    StringRef name;
    StringRef value;
    if (parseDefineFun(text, parsePos, name, value)) {
      if (name.size() >= 2 && name.front() == '|' && name.back() == '|')
        name = name.drop_front().drop_back();
      values[name] = circt::normalizeSMTModelValue(value);
      pos = parsePos;
    } else {
      pos = start + 1;
    }
  }
  return values;
}

static LogicalResult insertGetModelBeforeReset(SmallString<128> smtPath) {
  auto buffer = llvm::MemoryBuffer::getFile(smtPath);
  if (!buffer) {
    llvm::errs() << "failed to read SMT file for model request\n";
    return failure();
  }
  std::string contents = buffer.get()->getBuffer().str();
  size_t resetPos = contents.rfind("(reset)");
  std::string getModel = "(get-model)\n";
  if (resetPos != std::string::npos)
    contents.insert(resetPos, getModel);
  else
    contents.append(getModel);

  std::error_code ec;
  llvm::raw_fd_ostream os(smtPath, ec);
  if (ec) {
    llvm::errs() << "failed to write SMT file: " << ec.message() << "\n";
    return failure();
  }
  os << contents;
  return success();
}

enum class BMCResult { Unsat, Sat, Unknown };

static const char *toResultString(BMCResult result) {
  switch (result) {
  case BMCResult::Unsat:
    return "UNSAT";
  case BMCResult::Sat:
    return "SAT";
  case BMCResult::Unknown:
    return "UNKNOWN";
  }
  return "UNKNOWN";
}

static LogicalResult runPassPipeline(MLIRContext &context, ModuleOp module,
                                     TimingScope &ts,
                                     ConvertVerifToSMTOptions convertOptions,
                                     unsigned boundOverride,
                                     bool emitResultMessages) {
  auto configurePassManager = [&](PassManager &pm) -> LogicalResult {
    pm.enableVerifier(verifyPasses);
    pm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    pm.addInstrumentation(std::make_unique<ResourceGuardPassInstrumentation>());
    if (verbosePassExecutions)
      pm.addInstrumentation(
          std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
              "circt-bmc"));
    return success();
  };

  PassManager pm(&context);
  if (failed(configurePassManager(pm)))
    return failure();

  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(sim::createStripSim());
  pm.addPass(verif::createLowerTestsPass());
  if (pruneUnreachableSymbols) {
    // Prune unreachable symbols early so unsupported ops in dead modules do
    // not block subsequent lowering.
    StripUnreachableSymbolsOptions pruneOptions;
    pruneOptions.entrySymbol = moduleName;
    pm.addPass(createStripUnreachableSymbols(pruneOptions));
  }

  bool hasLLHD = false;
  module.walk([&](Operation *op) {
    if (auto *dialect = op->getDialect())
      if (dialect->getNamespace() == "llhd")
        hasLLHD = true;
  });

  if (hasLLHD) {
    LlhdToCorePipelineOptions llhdOptions;
    pm.addNestedPass<hw::HWModuleOp>(llhd::createWrapProceduralOpsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(llhd::createInlineCallsPass());
    pm.addPass(createConvertMooreToCorePass());
    pm.addPass(mlir::createSymbolDCEPass());

    auto &llhdPrePM = pm.nest<hw::HWModuleOp>();
    if (llhdOptions.sroa)
      llhdPrePM.addPass(mlir::createSROA());
    llhdPrePM.addPass(llhd::createMem2RegPass());
    llhdPrePM.addPass(llhd::createHoistSignalsPass());
    llhdPrePM.addPass(llhd::createDeseqPass());

    // Hoist assertions before LLHD process lowering removes them.
    pm.addPass(createStripLLHDProcesses());

    auto &llhdPostPM = pm.nest<hw::HWModuleOp>();
    llhdPostPM.addPass(llhd::createLowerProcessesPass());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(createBottomUpSimpleCanonicalizerPass());
    llhdPostPM.addPass(llhd::createUnrollLoopsPass());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(createBottomUpSimpleCanonicalizerPass());
    llhdPostPM.addPass(llhd::createRemoveControlFlowPass());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(createBottomUpSimpleCanonicalizerPass());
    llhdPostPM.addPass(createMapArithToCombPass(true));
    llhdPostPM.addPass(llhd::createCombineDrivesPass());
    llhdPostPM.addPass(llhd::createSig2Reg());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(createBottomUpSimpleCanonicalizerPass());
    if (llhdOptions.detectMemories) {
      llhdPostPM.addPass(seq::createRegOfVecToMem());
      llhdPostPM.addPass(mlir::createCSEPass());
      llhdPostPM.addPass(createBottomUpSimpleCanonicalizerPass());
    }
  }
  pm.nest<hw::HWModuleOp>().addPass(createLowerSVAToLTLPass());
  pm.nest<hw::HWModuleOp>().addPass(createLowerClockedAssertLikePass());
  pm.nest<hw::HWModuleOp>().addPass(createLowerLTLToCorePass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(createBottomUpSimpleCanonicalizerPass());
  if (flattenModules) {
    pm.addPass(hw::createFlattenModules());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createBottomUpSimpleCanonicalizerPass());
  }
  // Normalize aggregate bitcasts before externalizing registers so any clock
  // keys computed during externalization match the post-normalization form
  // seen by LowerToBMC. This avoids multi-clock key mismatches caused by
  // alternate 4-state field extraction forms (e.g. hw.struct_extract vs
  // hw.struct_explode/comb.concat/comb.extract).
  pm.nest<hw::HWModuleOp>().addPass(hw::createHWAggregateToComb());
  pm.addPass(hw::createHWConvertBitcasts());
  ExternalizeRegistersOptions externalizeOptions;
  externalizeOptions.allowMultiClock = allowMultiClock;
  pm.addPass(createExternalizeRegisters(externalizeOptions));
  if (pruneBMCRegisters)
    pm.addPass(createPruneBMCRegisters());
  LowerToBMCOptions lowerToBMCOptions;
  lowerToBMCOptions.bound = boundOverride;
  lowerToBMCOptions.ignoreAssertionsUntil = ignoreAssertionsUntil;
  lowerToBMCOptions.topModule = moduleName;
  lowerToBMCOptions.risingClocksOnly = risingClocksOnly;
  lowerToBMCOptions.allowMultiClock = allowMultiClock;
  lowerToBMCOptions.emitResultMessages = emitResultMessages;
  pm.addPass(createLowerToBMC(lowerToBMCOptions));
  pm.addPass(createConvertHWToSMT());
  pm.addPass(createConvertCombToSMT());
  pm.addPass(createConvertVerifToSMT(convertOptions));
  pm.addPass(createBottomUpSimpleCanonicalizerPass());
  pm.addPass(createSMTDeadCodeEliminationPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (pruneUnreachableSymbols) {
    // A second pruning round catches dead helper symbols introduced by
    // lowering/conversion passes.
    StripUnreachableSymbolsOptions pruneOptions;
    pruneOptions.entrySymbol = moduleName;
    pm.addPass(createStripUnreachableSymbols(pruneOptions));
  }

  circt::setResourceGuardPhase("pass pipeline (pre-lowering)");
  if (failed(pm.run(module)))
    return failure();

  if (outputFormat == OutputMLIR || outputFormat == OutputSMTLIB ||
      outputFormat == OutputRunSMTLIB)
    return success();

  if (printCounterexample || printSolverOutput) {
    bool copiedEventAttrs = false;
    module.walk([&](mlir::smt::SolverOp solver) {
      if (copiedEventAttrs)
        return;
      auto eventSources = solver->getAttrOfType<ArrayAttr>("bmc_event_sources");
      if (!eventSources)
        eventSources =
            solver->getAttrOfType<ArrayAttr>("bmc_mixed_event_sources");
      if (!eventSources)
        return;
      copiedEventAttrs = true;
      module->setAttr("circt.bmc_event_sources", eventSources);
      if (auto details =
              solver->getAttrOfType<ArrayAttr>("bmc_event_source_details"))
        module->setAttr("circt.bmc_event_source_details", details);
    });
  }

  PassManager lowerPM(&context);
  if (failed(configurePassManager(lowerPM)))
    return failure();
  LowerSMTToZ3LLVMOptions options;
  options.debug = printSolverOutput;
  options.printModelInputs = printSolverOutput || printCounterexample;
  lowerPM.addPass(createLowerSMTToZ3LLVM(options));
  lowerPM.addPass(createCSEPass());
  lowerPM.addPass(createBottomUpSimpleCanonicalizerPass());
  lowerPM.addPass(LLVM::createDIScopeForLLVMFuncOpPass());

  circt::setResourceGuardPhase("pass pipeline (lower smt)");
  if (failed(lowerPM.run(module)))
    return failure();
  return success();
}

static FailureOr<BMCResult>
runSMTLIBSolver(ModuleOp module, bool wantSolverOutput, bool printResultLines) {
  if (!hasSMTSolver(module)) {
    if (printResultLines)
      llvm::outs()
          << "BMC_RESULT=UNSAT\nBound reached with no violations!\n";
    return BMCResult::Unsat;
  }

  std::optional<std::string> z3Program;
  if (!z3PathOpt.empty()) {
    if (!llvm::sys::fs::exists(z3PathOpt)) {
      llvm::errs() << "z3 not found at '" << z3PathOpt << "'\n";
      return failure();
    }
    z3Program = z3PathOpt;
  } else {
    auto z3Path = llvm::sys::findProgramByName("z3");
    if (!z3Path) {
      llvm::errs() << "z3 not found in PATH; cannot run SMT-LIB\n";
      return failure();
    }
    z3Program = *z3Path;
  }

  SmallString<128> smtPath;
  int smtFd = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("circt-bmc", "smt2",
                                                   smtFd, smtPath)) {
    llvm::errs() << "failed to create temporary SMT file: " << ec.message()
                 << "\n";
    return failure();
  }
  llvm::FileRemover smtRemover(smtPath);
  {
    llvm::raw_fd_ostream smtStream(smtFd, true);
    if (failed(smt::exportSMTLIB(module, smtStream)))
      return failure();
  }

  SmallVector<std::string> declaredNames;
  if (auto buffer = llvm::MemoryBuffer::getFile(smtPath)) {
    StringRef data = buffer.get()->getBuffer();
    for (StringRef line : llvm::split(data, '\n')) {
      line = line.trim();
      if (!line.starts_with("(declare-const "))
        continue;
      line = line.drop_front(StringRef("(declare-const ").size());
      StringRef name = line.take_until([](char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ')';
      });
      if (!name.empty())
        declaredNames.push_back(name.str());
    }
  }

  struct Z3Run {
    int exitCode = 0;
    std::string errMsg;
    std::string combinedOutput;
    std::optional<std::string> token;
  };

  auto runZ3 = [&](StringRef filePath, bool requestModel,
                   Z3Run &run) -> LogicalResult {
    SmallString<128> outPath;
    int outFd = -1;
    if (auto ec = llvm::sys::fs::createTemporaryFile("circt-bmc", "out", outFd,
                                                     outPath)) {
      llvm::errs() << "failed to create temporary output file: "
                   << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover outRemover(outPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(outFd);

    SmallString<128> errPath;
    int errFd = -1;
    if (auto ec = llvm::sys::fs::createTemporaryFile("circt-bmc", "err", errFd,
                                                     errPath)) {
      llvm::errs() << "failed to create temporary error file: " << ec.message()
                   << "\n";
      return failure();
    }
    llvm::FileRemover errRemover(errPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(errFd);

    SmallVector<StringRef, 4> args;
    args.push_back(*z3Program);
    if (requestModel)
      args.push_back("-model");
    args.push_back(filePath);
    std::array<std::optional<StringRef>, 3> redirects = {
        std::nullopt, outPath.str(), errPath.str()};
    run.exitCode = llvm::sys::ExecuteAndWait(
        *z3Program, args, std::nullopt, redirects, 0, 0, &run.errMsg);

    auto outBuffer = llvm::MemoryBuffer::getFile(outPath);
    if (!outBuffer) {
      llvm::errs() << "failed to read z3 output\n";
      return failure();
    }
    auto errBuffer = llvm::MemoryBuffer::getFile(errPath);
    run.combinedOutput = outBuffer.get()->getBuffer().str();
    if (errBuffer && !errBuffer.get()->getBuffer().empty()) {
      run.combinedOutput.append("\n");
      run.combinedOutput.append(errBuffer.get()->getBuffer().str());
    }
    run.token = findResultToken(run.combinedOutput);
    return success();
  };

  Z3Run initialRun;
  if (failed(runZ3(smtPath, wantSolverOutput, initialRun)))
    return failure();
  if (!initialRun.errMsg.empty()) {
    llvm::errs() << "z3 invocation failed: " << initialRun.errMsg << "\n";
    return failure();
  }
  const Z3Run *selectedRun = &initialRun;
  std::optional<SmallString<128>> smtModelPath;
  std::optional<llvm::FileRemover> smtModelRemover;
  Z3Run modelRun;
  if (wantSolverOutput &&
      (!initialRun.token || *initialRun.token == "sat" ||
       *initialRun.token == "unknown")) {
    SmallString<128> smtPathWithModel;
    int smtModelFd = -1;
    if (auto ec = llvm::sys::fs::createTemporaryFile("circt-bmc", "smt2",
                                                     smtModelFd,
                                                     smtPathWithModel)) {
      llvm::errs() << "failed to create temporary SMT file: " << ec.message()
                   << "\n";
      return failure();
    }
    llvm::FileRemover modelRemover(smtPathWithModel);
    smtModelRemover.emplace(std::move(modelRemover));
    llvm::sys::Process::SafelyCloseFileDescriptor(smtModelFd);
    if (auto buffer = llvm::MemoryBuffer::getFile(smtPath)) {
      std::error_code ec;
      llvm::raw_fd_ostream os(smtPathWithModel, ec);
      if (ec) {
        llvm::errs() << "failed to write SMT file: " << ec.message() << "\n";
        return failure();
      }
      os << buffer.get()->getBuffer();
    }
    if (failed(insertGetModelBeforeReset(smtPathWithModel)))
      return failure();
    if (failed(runZ3(smtPathWithModel, /*requestModel=*/true, modelRun)))
      return failure();
    if (!modelRun.errMsg.empty()) {
      llvm::errs() << "z3 invocation failed: " << modelRun.errMsg << "\n";
      return failure();
    }
    if (modelRun.token)
      selectedRun = &modelRun;
  }

  if (wantSolverOutput)
    llvm::errs() << "z3 output:\n" << selectedRun->combinedOutput << "\n";

  if (!selectedRun->token) {
    llvm::errs() << "unexpected z3 output: " << selectedRun->combinedOutput
                 << "\n";
    return failure();
  }
  StringRef token = *selectedRun->token;

  if (printCounterexample && (token == "sat" || token == "unknown")) {
    auto modelValues = parseZ3Model(selectedRun->combinedOutput);
    if (!modelValues.empty())
      printMixedEventSources(module, &modelValues);
    else
      printMixedEventSources(module);
    if (!declaredNames.empty() && !modelValues.empty()) {
        circt_smt_print_model_header();
        for (const auto &name : declaredNames) {
          auto it = modelValues.find(name);
          if (it == modelValues.end())
            continue;
          llvm::errs() << "  " << name << " = " << it->second << "\n";
        }
    }
  }

  if (token == "unsat") {
    if (printResultLines)
      llvm::outs()
          << "BMC_RESULT=UNSAT\nBound reached with no violations!\n";
    return BMCResult::Unsat;
  }
  if (token == "sat") {
    if (printResultLines)
      llvm::outs() << "BMC_RESULT=SAT\nAssertion can be violated!\n";
    return BMCResult::Sat;
  }
  if (token == "unknown") {
    if (printResultLines)
      llvm::outs() << "BMC_RESULT=UNKNOWN\nSolver returned unknown.\n";
    return BMCResult::Unknown;
  }
  llvm::errs() << "unexpected z3 result: " << token << "\n";
  return failure();
}

#ifdef CIRCT_BMC_ENABLE_JIT
static FailureOr<BMCResult> runJITSolver(ModuleOp module, TimingScope &ts) {
  auto handleErr = [](llvm::Error error) -> LogicalResult {
    llvm::handleAllErrors(std::move(error),
                          [](const llvm::ErrorInfoBase &info) {
                            llvm::errs() << "Error: ";
                            info.log(llvm::errs());
                            llvm::errs() << '\n';
                          });
    return failure();
  };

  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::function<llvm::Error(llvm::Module *)> transformer =
      mlir::makeOptimizingTransformer(
          /*optLevel*/ 3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  {
    auto timer = ts.nest("Setting up the JIT");
    auto entryPoint =
        dyn_cast_or_null<LLVM::LLVMFuncOp>(module.lookupSymbol(moduleName));
    if (!entryPoint || entryPoint.empty()) {
      llvm::errs() << "no valid entry point found, expected 'llvm.func' named '"
                   << moduleName << "'\n";
      return failure();
    }

    if (entryPoint.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << moduleName
                   << "' must have no arguments";
      return failure();
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = transformer;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    engineOptions.sharedLibPaths = sharedLibraries;
    engineOptions.enableObjectDump = true;

    auto expectedEngine =
        mlir::ExecutionEngine::create(module, engineOptions);
    if (!expectedEngine)
      return handleErr(expectedEngine.takeError());

    engine = std::move(*expectedEngine);

    // Register the circt_bmc_report_result callback symbol so JIT-compiled code
    // can call back to report BMC results.
    engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
      llvm::orc::SymbolMap symbolMap;
      symbolMap[interner("circt_bmc_report_result")] = {
          llvm::orc::ExecutorAddr::fromPtr(&circt_bmc_report_result),
          llvm::JITSymbolFlags::Exported};
      symbolMap[interner("circt_smt_print_model_header")] = {
          llvm::orc::ExecutorAddr::fromPtr(&circt_smt_print_model_header),
          llvm::JITSymbolFlags::Exported};
      symbolMap[interner("circt_smt_print_model_value")] = {
          llvm::orc::ExecutorAddr::fromPtr(&circt_smt_print_model_value),
          llvm::JITSymbolFlags::Exported};
      return symbolMap;
    });
  }

  auto timer = ts.nest("JIT Execution");
  bmcJitResult.store(-1, std::memory_order_relaxed);
  circt::resetCapturedSMTModelValues();
  if (auto err = engine->invokePacked(moduleName))
    return handleErr(std::move(err));
  int result = bmcJitResult.load(std::memory_order_relaxed);
  if (result < 0) {
    llvm::errs() << "BMC JIT did not report a result\n";
    return failure();
  }

  BMCResult bmcResult = result ? BMCResult::Unsat : BMCResult::Sat;
  if (printCounterexample && bmcResult == BMCResult::Sat) {
    auto modelValues = circt::getCapturedSMTModelValues();
    if (!modelValues.empty())
      printMixedEventSources(module, &modelValues);
    else
      printMixedEventSources(module);
  }

  return bmcResult;
}
#endif

static FailureOr<BMCResult> runBMCOnce(MLIRContext &context, ModuleOp module,
                                       TimingScope &ts,
                                       ConvertVerifToSMTOptions convertOptions,
                                       unsigned boundOverride,
                                       bool wantSolverOutput,
                                       bool printResultLines,
                                       bool emitResultMessages) {
  if (failed(runPassPipeline(context, module, ts, convertOptions,
                             boundOverride, emitResultMessages)))
    return failure();

#ifdef CIRCT_BMC_ENABLE_JIT
  if (outputFormat == OutputRunJIT)
    return runJITSolver(module, ts);
#endif

  circt::setResourceGuardPhase("run z3");
  auto timer = ts.nest("Run SMT-LIB via z3");
  return runSMTLIBSolver(module, wantSolverOutput, printResultLines);
}

static LogicalResult executeBMCWithInduction(MLIRContext &context) {
  if (liveness || livenessLasso) {
    llvm::errs()
        << "--liveness/--liveness-lasso is incompatible with "
           "--induction/--k-induction\n";
    return failure();
  }
#ifdef CIRCT_BMC_ENABLE_JIT
  if (outputFormat != OutputRunSMTLIB && outputFormat != OutputRunJIT) {
    llvm::errs()
        << "--induction/--k-induction requires --run or --run-smtlib\n";
    return failure();
  }
#else
  if (outputFormat != OutputRunSMTLIB) {
    llvm::errs() << "--induction/--k-induction requires --run-smtlib\n";
    return failure();
  }
#endif
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();
  bool wantSolverOutput = printSolverOutput || printCounterexample;

  circt::setResourceGuardPhase("parse");
  OwningOpRef<ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }
  if (!module)
    return failure();

  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  ConvertVerifToSMTOptions baseOptions;
  baseOptions.risingClocksOnly = risingClocksOnly;
  baseOptions.assumeKnownInputs = assumeKnownInputs;
  baseOptions.forSMTLIBExport = true;
  baseOptions.bmcMode = "induction-base";

  OwningOpRef<ModuleOp> baseModule(
      llvm::cast<ModuleOp>(module->clone()));
  bool emitResultMessages = true;
#ifdef CIRCT_BMC_ENABLE_JIT
  if (outputFormat == OutputRunJIT)
    emitResultMessages = false;
#endif
  auto baseResultOr =
      runBMCOnce(context, *baseModule, ts, baseOptions, clockBound,
                 wantSolverOutput, /*printResultLines=*/false,
                 emitResultMessages);
  if (failed(baseResultOr))
    return failure();
  BMCResult baseResult = *baseResultOr;
  llvm::outs() << "BMC_BASE=" << toResultString(baseResult) << "\n";

  if (baseResult != BMCResult::Unsat) {
    llvm::outs() << "BMC_RESULT=" << toResultString(baseResult) << "\n";
    if (baseResult == BMCResult::Sat)
      llvm::outs() << "Assertion can be violated!\n";
    else
      llvm::outs() << "Solver returned unknown.\n";
    outputFile.value()->keep();
    if (failOnViolation)
      return failure();
    return success();
  }

  ConvertVerifToSMTOptions stepOptions = baseOptions;
  stepOptions.bmcMode = "induction-step";
  OwningOpRef<ModuleOp> stepModule(
      llvm::cast<ModuleOp>(module->clone()));
  auto stepResultOr =
      runBMCOnce(context, *stepModule, ts, stepOptions, clockBound + 1,
                 wantSolverOutput, /*printResultLines=*/false,
                 emitResultMessages);
  if (failed(stepResultOr))
    return failure();
  BMCResult stepResult = *stepResultOr;
  llvm::outs() << "BMC_STEP=" << toResultString(stepResult) << "\n";

  if (stepResult == BMCResult::Unsat) {
    llvm::outs() << "BMC_RESULT=UNSAT\nInduction holds.\n";
    outputFile.value()->keep();
    return success();
  }

  llvm::outs() << "BMC_RESULT=UNKNOWN\nInduction step failed.\n";
  outputFile.value()->keep();
  if (failOnViolation)
    return failure();
  return success();
}

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeBMC(MLIRContext &context) {
  if (kInduction || induction)
    return executeBMCWithInduction(context);
  if (livenessLasso && !liveness) {
    llvm::errs() << "--liveness-lasso requires --liveness\n";
    return failure();
  }
  if (livenessLasso && outputFormat != OutputSMTLIB &&
      outputFormat != OutputRunSMTLIB) {
    llvm::errs()
        << "--liveness-lasso requires --emit-smtlib or --run-smtlib\n";
    return failure();
  }
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();
  bool wantSolverOutput = printSolverOutput || printCounterexample;

  circt::setResourceGuardPhase("parse");
  OwningOpRef<ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    // Parse the provided input files.
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }
  if (!module)
    return failure();

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  ConvertVerifToSMTOptions convertVerifToSMTOptions;
  convertVerifToSMTOptions.risingClocksOnly = risingClocksOnly;
  convertVerifToSMTOptions.assumeKnownInputs = assumeKnownInputs;
  convertVerifToSMTOptions.forSMTLIBExport =
      (outputFormat == OutputSMTLIB || outputFormat == OutputRunSMTLIB);
  convertVerifToSMTOptions.bmcMode =
      livenessLasso ? "liveness-lasso" : (liveness ? "liveness" : "bounded");
  if (failed(runPassPipeline(context, *module, ts, convertVerifToSMTOptions,
                             clockBound, /*emitResultMessages=*/true)))
    return failure();

  if (outputFormat == OutputMLIR) {
    circt::setResourceGuardPhase("print mlir");
    auto timer = ts.nest("Print MLIR output");
    OpPrintingFlags printingFlags;
    module->print(outputFile.value()->os(), printingFlags);
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputSMTLIB) {
    circt::setResourceGuardPhase("export smtlib");
    auto timer = ts.nest("Print SMT-LIB output");
    if (!hasSMTSolver(*module)) {
      // If no solver is present, there is nothing meaningful to export. Emit a
      // stub query that returns UNSAT, which corresponds to "no violations".
      auto &os = outputFile.value()->os();
      os << "(assert false)\n";
      os << "(check-sat)\n";
      os << "(reset)\n";
      outputFile.value()->keep();
      return success();
    }
    if (failed(smt::exportSMTLIB(module.get(), outputFile.value()->os())))
      return failure();
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputRunSMTLIB) {
    auto resultOr =
        runSMTLIBSolver(*module, wantSolverOutput, /*printResultLines=*/true);
    if (failed(resultOr))
      return failure();
    outputFile.value()->keep();
    BMCResult result = *resultOr;
    if (result == BMCResult::Unsat)
      return success();
    return failOnViolation ? failure() : success();
  }

  if (outputFormat == OutputLLVM) {
    auto timer = ts.nest("Translate to and print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    outputFile.value()->keep();
    return success();
  }

#ifdef CIRCT_BMC_ENABLE_JIT
  auto resultOr = runJITSolver(*module, ts);
  if (failed(resultOr))
    return failure();
  if (failOnViolation && *resultOr == BMCResult::Sat)
    return failure();
  return success();
#else
  return failure();
#endif
}

/// The entry point for the `circt-bmc` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeBMC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions({&mainCategory, &circt::getResourceGuardCategory()});

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-bmc - bounded model checker\n\n"
      "\tThis tool checks all possible executions of a hardware module up to a "
      "given time bound to check whether any asserted properties can be "
      "violated.\n");
  circt::installResourceGuard();

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::llhd::LLHDDialect,
    circt::ltl::LTLDialect,
    circt::moore::MooreDialect,
    circt::om::OMDialect,
    circt::sim::SimDialect,
    circt::seq::SeqDialect,
    mlir::smt::SMTDialect,
    circt::verif::VerifDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::BuiltinDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect
  >();
  // clang-format on
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeBMC(context)));
}
