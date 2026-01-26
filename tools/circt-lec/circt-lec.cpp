//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-lec' tool, which interfaces with a logical
/// engine to allow its user to check whether two input circuit descriptions
/// are equivalent, and when not provides a counterexample as for why.
///
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/DatapathToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/ImportVerilog.h"
#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/SVAToLTL.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/SVA/SVADialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <array>
#include <cctype>

#ifdef CIRCT_LEC_ENABLE_JIT
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string> firstModuleName(
    "c1", cl::Required,
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> secondModuleName(
    "c2", cl::Required,
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> printSolverOutput(
    "print-solver-output",
    cl::desc("Print the output (counterexample or proof) produced by the "
             "solver on each invocation and the assertion set that they "
             "prove/disprove."),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> flattenHWModules(
    "flatten-hw",
    cl::desc("Inline private hw.modules before equivalence checking"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<std::string>
    z3PathOpt("z3-path",
              cl::desc("Path to the z3 binary for --run-smtlib"),
              cl::value_desc("filename"), cl::init(""),
              cl::cat(mainCategory));

#ifdef CIRCT_LEC_ENABLE_JIT

enum OutputFormat {
  OutputMLIR,
  OutputLLVM,
  OutputSMTLIB,
  OutputRunSMTLIB,
  OutputRunJIT
};
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit LLVM MLIR dialect"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit object file"),
               clEnumValN(OutputRunSMTLIB, "run-smtlib",
                          "Run equivalence checking via SMT-LIB + z3"),
               clEnumValN(OutputRunJIT, "run",
                          "Perform LEC and output result")),
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
               clEnumValN(OutputSMTLIB, "emit-smtlib", "Emit object file"),
               clEnumValN(OutputRunSMTLIB, "run-smtlib",
                          "Run equivalence checking via SMT-LIB + z3")),
    cl::init(OutputLLVM), cl::cat(mainCategory));

#endif

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid
// conflict.
static FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src,
                                          StringAttr name) {

  SymbolTable destTable(dest), srcTable(src);
  StringAttr newName = {};
  for (auto &op : src.getOps()) {
    if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
      auto oldSymbol = symbol.getNameAttr();
      auto result = srcTable.renameToUnique(&op, {&destTable});
      if (failed(result))
        return src->emitError() << "failed to rename symbol " << oldSymbol;

      if (oldSymbol == name) {
        assert(!newName && "symbol must be unique");
        newName = *result;
      }
    }
  }

  if (!newName)
    return src->emitError()
           << "module " << name << " was not found in the second module";

  dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                         src.getBody()->getOperations());
  return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
static FailureOr<OwningOpRef<ModuleOp>>
parseAndMergeModules(MLIRContext &context, TimingScope &ts) {
  auto parserTimer = ts.nest("Parse and merge MLIR input(s)");

  if (inputFilenames.size() > 2) {
    llvm::errs() << "more than 2 files are provided!\n";
    return failure();
  }

  auto module = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
  if (!module)
    return failure();

  if (inputFilenames.size() == 2) {
    auto moduleOpt = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!moduleOpt)
      return failure();
    auto result = mergeModules(module.get(), moduleOpt.get(),
                               StringAttr::get(&context, secondModuleName));
    if (failed(result))
      return failure();

    secondModuleName.setValue(result->getValue().str());
  }

  return module;
}

static void skipWhitespace(StringRef text, size_t &pos) {
  while (pos < text.size() &&
         isspace(static_cast<unsigned char>(text[pos])))
    ++pos;
}

static bool parseAtom(StringRef text, size_t &pos, StringRef &out) {
  skipWhitespace(text, pos);
  if (pos >= text.size())
    return false;
  if (text[pos] == '|') {
    size_t end = text.find('|', pos + 1);
    if (end == StringRef::npos)
      return false;
    out = text.slice(pos, end + 1);
    pos = end + 1;
    return true;
  }
  size_t start = pos;
  while (pos < text.size()) {
    char c = text[pos];
    if (isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')')
      break;
    ++pos;
  }
  if (pos == start)
    return false;
  out = text.slice(start, pos);
  return true;
}

static bool parseExpr(StringRef text, size_t &pos, StringRef &out) {
  skipWhitespace(text, pos);
  if (pos >= text.size())
    return false;
  if (text[pos] != '(')
    return parseAtom(text, pos, out);

  size_t start = pos;
  unsigned depth = 0;
  while (pos < text.size()) {
    char c = text[pos];
    if (c == '(')
      ++depth;
    else if (c == ')') {
      if (depth == 0)
        break;
      --depth;
      if (depth == 0) {
        ++pos;
        out = text.slice(start, pos);
        return true;
      }
    }
    ++pos;
  }
  return false;
}

static bool parseDefineFun(StringRef text, size_t &pos, StringRef &name,
                           StringRef &value) {
  size_t localPos = pos;
  skipWhitespace(text, localPos);
  if (localPos >= text.size() || text[localPos] != '(')
    return false;
  ++localPos;
  StringRef keyword;
  if (!parseAtom(text, localPos, keyword) || keyword != "define-fun")
    return false;
  if (!parseAtom(text, localPos, name))
    return false;
  StringRef params;
  if (!parseExpr(text, localPos, params))
    return false;
  StringRef sort;
  if (!parseExpr(text, localPos, sort))
    return false;
  if (!parseExpr(text, localPos, value))
    return false;
  skipWhitespace(text, localPos);
  if (localPos < text.size() && text[localPos] == ')')
    ++localPos;
  pos = localPos;
  return true;
}

static std::string formatBitVectorValue(const llvm::APInt &value,
                                        unsigned width) {
  if (width == 0)
    return "0";
  if (width % 4 == 0) {
    llvm::SmallString<64> hexStr;
    value.toString(hexStr, 16, /*Signed=*/false);
    std::string hex(hexStr.begin(), hexStr.end());
    size_t digits = width / 4;
    if (hex.size() < digits)
      hex.insert(0, digits - hex.size(), '0');
    return (Twine(width) + "'h" + hex).str();
  }
  llvm::SmallString<64> decStr;
  value.toString(decStr, 10, /*Signed=*/false);
  return (Twine(width) + "'d" + StringRef(decStr)).str();
}

static std::optional<std::string> tryFormatBitVector(StringRef value) {
  StringRef work = value.trim();
  if (work.consume_front("#b")) {
    if (work.empty() || work.find_first_not_of("01") != StringRef::npos)
      return std::nullopt;
    unsigned width = work.size();
    llvm::APInt ap(width, work, 2);
    return formatBitVectorValue(ap, width);
  }
  if (work.consume_front("#x")) {
    if (work.empty() ||
        work.find_first_not_of("0123456789abcdefABCDEF") != StringRef::npos)
      return std::nullopt;
    unsigned width = work.size() * 4;
    llvm::APInt ap(width, work, 16);
    return formatBitVectorValue(ap, width);
  }
  if (!work.consume_front("(_ bv"))
    return std::nullopt;
  size_t numEnd = work.find_first_not_of("0123456789");
  if (numEnd == StringRef::npos)
    return std::nullopt;
  StringRef numStr = work.take_front(numEnd);
  work = work.drop_front(numEnd).ltrim();
  size_t widthEnd = work.find_first_not_of("0123456789");
  if (widthEnd == StringRef::npos)
    return std::nullopt;
  StringRef widthStr = work.take_front(widthEnd);
  work = work.drop_front(widthEnd).ltrim();
  if (!work.consume_front(")"))
    return std::nullopt;
  if (!work.trim().empty())
    return std::nullopt;
  unsigned width = 0;
  if (widthStr.getAsInteger(10, width))
    return std::nullopt;
  if (width == 0)
    return "0";
  llvm::APInt ap(width, numStr, 10);
  return formatBitVectorValue(ap, width);
}

static std::string normalizeModelValue(StringRef value) {
  value = value.trim();
  if (value == "true" || value == "false")
    return value.str();
  if (auto formatted = tryFormatBitVector(value))
    return *formatted;
  return value.str();
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
      values[name] = normalizeModelValue(value);
      pos = parsePos;
    } else {
      pos = start + 1;
    }
  }
  return values;
}

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeLEC(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  auto parsedModule = parseAndMergeModules(context, ts);
  if (failed(parsedModule))
    return failure();

  OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

  SmallVector<std::string> lecInputNames;
  if (outputFormat == OutputRunSMTLIB && printSolverOutput) {
    if (auto moduleA =
            module->lookupSymbol<hw::HWModuleOp>(firstModuleName)) {
      for (auto name : moduleA.getInputNames())
        if (auto strAttr = dyn_cast<StringAttr>(name))
          if (!strAttr.getValue().empty())
            lecInputNames.push_back(strAttr.getValue().str());
    }
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  bool hasLLHD = false;
  module->walk([&](Operation *op) {
    if (auto *dialect = op->getDialect())
      if (dialect->getNamespace() == "llhd")
        hasLLHD = true;
  });

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "circt-lec"));

  if (flattenHWModules)
    pm.addPass(hw::createFlattenModules());
  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(sim::createStripSim());
  if (hasLLHD) {
    pm.addPass(createLowerLLHDRefPorts());
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
    llhdPostPM.addPass(mlir::createCanonicalizerPass());
    llhdPostPM.addPass(llhd::createUnrollLoopsPass());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(mlir::createCanonicalizerPass());
    llhdPostPM.addPass(llhd::createRemoveControlFlowPass());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(mlir::createCanonicalizerPass());
    llhdPostPM.addPass(createMapArithToCombPass(true));
    llhdPostPM.addPass(llhd::createCombineDrivesPass());
    llhdPostPM.addPass(llhd::createSig2Reg());
    llhdPostPM.addPass(mlir::createCSEPass());
    llhdPostPM.addPass(mlir::createCanonicalizerPass());
    if (llhdOptions.detectMemories) {
      llhdPostPM.addPass(seq::createRegOfVecToMem());
      llhdPostPM.addPass(mlir::createCSEPass());
      llhdPostPM.addPass(mlir::createCanonicalizerPass());
    }
  }
  pm.addPass(createStripLLHDInterfaceSignals());
  pm.nest<hw::HWModuleOp>().addPass(createLowerSVAToLTLPass());
  pm.nest<hw::HWModuleOp>().addPass(createLowerClockedAssertLikePass());
  pm.nest<hw::HWModuleOp>().addPass(createLowerLTLToCorePass());
  ExternalizeRegistersOptions externalizeOptions;
  pm.addPass(createExternalizeRegisters(externalizeOptions));
  pm.nest<hw::HWModuleOp>().addPass(hw::createHWAggregateToComb());
  pm.addPass(hw::createHWConvertBitcasts());
  {
    ConstructLECOptions opts;
    opts.firstModule = firstModuleName;
    opts.secondModule = secondModuleName;
    if (outputFormat == OutputSMTLIB || outputFormat == OutputRunSMTLIB)
      opts.insertMode = lec::InsertAdditionalModeEnum::None;
    pm.addPass(createConstructLEC(opts));
  }
  pm.addPass(createConvertHWToSMT());
  pm.addPass(createConvertDatapathToSMT());
  pm.addPass(createConvertCombToSMT());
  pm.addPass(createConvertVerifToSMT());
  pm.addPass(createSimpleCanonicalizerPass());

  if (outputFormat != OutputMLIR && outputFormat != OutputSMTLIB &&
      outputFormat != OutputRunSMTLIB) {
    LowerSMTToZ3LLVMOptions options;
    options.debug = printSolverOutput;
    pm.addPass(createLowerSMTToZ3LLVM(options));
    pm.addPass(createCSEPass());
    pm.addPass(createSimpleCanonicalizerPass());
    pm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (failed(pm.run(module.get())))
    return failure();

  if (outputFormat == OutputMLIR) {
    auto timer = ts.nest("Print MLIR output");
    OpPrintingFlags printingFlags;
    module->print(outputFile.value()->os(), printingFlags);
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputSMTLIB) {
    auto timer = ts.nest("Print SMT-LIB output");
    if (failed(smt::exportSMTLIB(module.get(), outputFile.value()->os())))
      return failure();
    outputFile.value()->keep();
    return success();
  }

  if (outputFormat == OutputRunSMTLIB) {
    auto timer = ts.nest("Run SMT-LIB via z3");
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
    if (auto ec = llvm::sys::fs::createTemporaryFile("circt-lec", "smt2",
                                                     smtFd, smtPath)) {
      llvm::errs() << "failed to create temporary SMT file: "
                   << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover smtRemover(smtPath);
    {
      llvm::raw_fd_ostream smtStream(smtFd, true);
      if (failed(smt::exportSMTLIB(module.get(), smtStream)))
        return failure();
    }

    SmallString<128> outPath;
    int outFd = -1;
    if (auto ec =
            llvm::sys::fs::createTemporaryFile("circt-lec", "out", outFd,
                                               outPath)) {
      llvm::errs() << "failed to create temporary output file: "
                   << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover outRemover(outPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(outFd);

    SmallString<128> errPath;
    int errFd = -1;
    if (auto ec =
            llvm::sys::fs::createTemporaryFile("circt-lec", "err", errFd,
                                               errPath)) {
      llvm::errs() << "failed to create temporary error file: "
                   << ec.message() << "\n";
      return failure();
    }
    llvm::FileRemover errRemover(errPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(errFd);

    SmallVector<StringRef, 4> args;
    args.push_back(*z3Program);
    if (printSolverOutput)
      args.push_back("-model");
    args.push_back(smtPath);
    std::string errMsg;
    std::array<std::optional<StringRef>, 3> redirects = {
        std::nullopt, outPath.str(), errPath.str()};
    int result = llvm::sys::ExecuteAndWait(*z3Program, args, std::nullopt,
                                           redirects, 0, 0, &errMsg);
    if (result != 0 || !errMsg.empty()) {
      llvm::errs() << "z3 invocation failed";
      if (!errMsg.empty())
        llvm::errs() << ": " << errMsg;
      llvm::errs() << "\n";
      return failure();
    }

    auto outBuffer = llvm::MemoryBuffer::getFile(outPath);
    if (!outBuffer) {
      llvm::errs() << "failed to read z3 output\n";
      return failure();
    }
    auto errBuffer = llvm::MemoryBuffer::getFile(errPath);
    std::string combinedOutput = outBuffer.get()->getBuffer().str();
    if (errBuffer && !errBuffer.get()->getBuffer().empty()) {
      combinedOutput.append("\n");
      combinedOutput.append(errBuffer.get()->getBuffer().str());
    }
    if (printSolverOutput) {
      llvm::errs() << "z3 output:\n" << combinedOutput << "\n";
    }

    auto findResultToken = [](StringRef text) -> std::optional<StringRef> {
      StringRef remaining = text;
      std::optional<StringRef> result;
      while (!remaining.empty()) {
        remaining = remaining.ltrim(" \t\r\n");
        if (remaining.empty())
          break;
        StringRef token =
            remaining.take_until([](char c) { return c == ' ' || c == '\t' ||
                                                     c == '\r' || c == '\n'; });
        if (token == "sat" || token == "unsat" || token == "unknown")
          result = token;
        remaining = remaining.drop_front(token.size());
      }
      return result;
    };

    auto token = findResultToken(combinedOutput);
    auto maybePrintCounterexample = [&](StringRef result) {
      if (!printSolverOutput)
        return;
      if (result != "sat" && result != "unknown")
        return;
      if (lecInputNames.empty())
        return;

      auto modelValues = parseZ3Model(combinedOutput);
      if (modelValues.empty())
        return;

      auto findValue = [&](StringRef inputName) -> std::optional<StringRef> {
        if (auto it = modelValues.find(inputName); it != modelValues.end())
          return it->second;
        StringRef candidate;
        for (const auto &entry : modelValues) {
          if (!entry.getKey().starts_with(inputName))
            continue;
          if (entry.getKey().size() == inputName.size())
            continue;
          if (entry.getKey()[inputName.size()] != '_')
            continue;
          if (!candidate.empty())
            return std::nullopt;
          candidate = entry.getKey();
        }
        if (!candidate.empty())
          return modelValues.find(candidate)->second;
        return std::nullopt;
      };

      bool printed = false;
      for (const auto &name : lecInputNames) {
        auto value = findValue(name);
        if (!value)
          continue;
        if (!printed) {
          llvm::errs() << "counterexample inputs:\n";
          printed = true;
        }
        llvm::errs() << "  " << name << " = " << *value << "\n";
      }
    };
    if (token && *token == "unsat") {
      outputFile.value()->os() << "c1 == c2\n";
    } else if (token && (*token == "sat" || *token == "unknown")) {
      outputFile.value()->os() << "c1 != c2\n";
      maybePrintCounterexample(*token);
    } else {
      llvm::errs() << "unexpected z3 output: " << combinedOutput << "\n";
      return failure();
    }
    outputFile.value()->keep();
    return success();
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

#ifdef CIRCT_LEC_ENABLE_JIT

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
    auto entryPoint = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        module->lookupSymbol(firstModuleName));
    if (!entryPoint || entryPoint.empty()) {
      llvm::errs() << "no valid entry point found, expected 'llvm.func' named '"
                   << firstModuleName << "'\n";
      return failure();
    }

    if (entryPoint.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << firstModuleName
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
        mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (!expectedEngine)
      return handleErr(expectedEngine.takeError());

    engine = std::move(*expectedEngine);
  }

  auto timer = ts.nest("JIT Execution");
  if (auto err = engine->invokePacked(firstModuleName))
    return handleErr(std::move(err));

  return success();
#else
  return failure();
#endif
}

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

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
      "circt-lec - logical equivalence checker\n\n"
      "\tThis tool compares two input circuit descriptions to determine whether"
      " they are logically equivalent.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::datapath::DatapathDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::ltl::LTLDialect,
    circt::llhd::LLHDDialect,
    circt::om::OMDialect,
    circt::sim::SimDialect,
    circt::sva::SVADialect,
    circt::seq::SeqDialect,
    mlir::smt::SMTDialect,
    circt::verif::VerifDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::BuiltinDialect,
    mlir::func::FuncDialect,
    mlir::scf::SCFDialect,
    mlir::LLVM::LLVMDialect
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
  exit(failed(executeLEC(context)));
}
