//===- ImportVerilog.cpp - Slang Verilog frontend integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements bridging from the slang Verilog frontend to CIRCT dialects.
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/SourceMgr.h"

#include "slang/ast/Compilation.h"
#include "slang/ast/SystemSubroutine.h"
#include "slang/diagnostics/DiagnosticClient.h"
#include "slang/diagnostics/ExpressionsDiags.h"
#include "slang/driver/Driver.h"
#include "slang/parsing/Preprocessor.h"
#include "slang/syntax/SyntaxPrinter.h"
#include "slang/util/Bag.h"
#include "slang/util/VersionInfo.h"

using namespace mlir;
using namespace circt;
using namespace ImportVerilog;

using llvm::SourceMgr;

std::string circt::getSlangVersion() {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "slang version ";
  os << slang::VersionInfo::getMajor() << ".";
  os << slang::VersionInfo::getMinor() << ".";
  os << slang::VersionInfo::getPatch() << "+";
  os << slang::VersionInfo::getHash();
  return buffer;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// Convert a slang `SourceLocation` to an MLIR `Location`.
static Location convertLocation(MLIRContext *context,
                                const slang::SourceManager &sourceManager,
                                slang::SourceLocation loc) {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    auto fileName = sourceManager.getFileName(loc);
    auto line = sourceManager.getLineNumber(loc);
    auto column = sourceManager.getColumnNumber(loc);
    return FileLineColLoc::get(context, fileName, line, column);
  }
  return UnknownLoc::get(context);
}

Location Context::convertLocation(slang::SourceLocation loc) {
  return ::convertLocation(getContext(), sourceManager, loc);
}

Location Context::convertLocation(slang::SourceRange range) {
  return convertLocation(range.start());
}

namespace {
/// A converter that can be plugged into a slang `DiagnosticEngine` as a client
/// that will map slang diagnostics to their MLIR counterpart and emit them.
class MlirDiagnosticClient : public slang::DiagnosticClient {
public:
  MlirDiagnosticClient(MLIRContext *context) : context(context) {}

  void report(const slang::ReportedDiagnostic &diag) override {
    // Generate the primary MLIR diagnostic.
    auto &diagEngine = context->getDiagEngine();
    auto mlirDiag = diagEngine.emit(convertLocation(diag.location),
                                    getSeverity(diag.severity));
    mlirDiag << diag.formattedMessage;

    // Append the name of the option that can be used to control this
    // diagnostic.
    auto optionName = engine->getOptionName(diag.originalDiagnostic.code);
    if (!optionName.empty())
      mlirDiag << " [-W" << optionName << "]";

    // Write out macro expansions, if we have any, in reverse order.
    for (auto loc : std::views::reverse(diag.expansionLocs)) {
      auto &note = mlirDiag.attachNote(
          convertLocation(sourceManager->getFullyOriginalLoc(loc)));
      auto macroName = sourceManager->getMacroName(loc);
      if (macroName.empty())
        note << "expanded from here";
      else
        note << "expanded from macro '" << macroName << "'";
    }

    // Write out the include stack.
    slang::SmallVector<slang::SourceLocation> includeStack;
    getIncludeStack(diag.location.buffer(), includeStack);
    for (auto &loc : std::views::reverse(includeStack))
      mlirDiag.attachNote(convertLocation(loc)) << "included from here";
  }

  /// Convert a slang `SourceLocation` to an MLIR `Location`.
  Location convertLocation(slang::SourceLocation loc) const {
    return ::convertLocation(context, *sourceManager, loc);
  }

  static DiagnosticSeverity getSeverity(slang::DiagnosticSeverity severity) {
    switch (severity) {
    case slang::DiagnosticSeverity::Fatal:
    case slang::DiagnosticSeverity::Error:
      return DiagnosticSeverity::Error;
    case slang::DiagnosticSeverity::Warning:
      return DiagnosticSeverity::Warning;
    case slang::DiagnosticSeverity::Ignored:
    case slang::DiagnosticSeverity::Note:
      return DiagnosticSeverity::Remark;
    }
    llvm_unreachable("all slang diagnostic severities should be handled");
    return DiagnosticSeverity::Error;
  }

private:
  MLIRContext *context;
};
} // namespace

// Allow for `slang::BufferID` to be used as hash map keys.
namespace llvm {
template <>
struct DenseMapInfo<slang::BufferID> {
  static slang::BufferID getEmptyKey() { return slang::BufferID(); }
  static slang::BufferID getTombstoneKey() {
    return slang::BufferID(UINT32_MAX - 1, ""sv);
    // UINT32_MAX is already used by `BufferID::getPlaceholder`.
  }
  static unsigned getHashValue(slang::BufferID id) {
    return llvm::hash_value(id.getId());
  }
  static bool isEqual(slang::BufferID a, slang::BufferID b) { return a == b; }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {
const static ImportVerilogOptions defaultOptions;

struct ImportDriver {
  ImportDriver(MLIRContext *mlirContext, TimingScope &ts,
               const ImportVerilogOptions *options)
      : mlirContext(mlirContext), ts(ts),
        options(options ? *options : defaultOptions) {}

  LogicalResult prepareDriver(SourceMgr &sourceMgr);
  LogicalResult importVerilog(ModuleOp module);
  LogicalResult preprocessVerilog(llvm::raw_ostream &os);

  MLIRContext *mlirContext;
  TimingScope &ts;
  const ImportVerilogOptions &options;

  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  slang::driver::Driver driver;
};
} // namespace

/// Populate the Slang driver with source files from the given `sourceMgr`, and
/// configure driver options based on the `ImportVerilogOptions` passed to the
/// `ImportDriver` constructor.
LogicalResult ImportDriver::prepareDriver(SourceMgr &sourceMgr) {
  // Use slang's driver which conveniently packages a lot of the things we
  // need for compilation.
  auto diagClient = std::make_shared<MlirDiagnosticClient>(mlirContext);
  driver.diagEngine.addClient(diagClient);

  for (const auto &value : options.commandFiles)
    if (!driver.processCommandFiles(value, /*makeRelative=*/true,
                                    /*separateUnit=*/true))
      return failure();

  // Populate the source manager with the source files.
  // NOTE: This is a bit ugly since we're essentially copying the Verilog
  // source text in memory. At a later stage we'll want to extend slang's
  // SourceManager such that it can contain non-owned buffers. This will do
  // for now.
  DenseSet<StringRef> seenBuffers;
  for (unsigned i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    const llvm::MemoryBuffer *mlirBuffer = sourceMgr.getMemoryBuffer(i + 1);
    auto name = mlirBuffer->getBufferIdentifier();
    if (!name.empty() && !seenBuffers.insert(name).second)
      continue; // Slang doesn't like listing the same buffer twice
    auto slangBuffer =
        driver.sourceManager.assignText(name, mlirBuffer->getBuffer());
    driver.sourceLoader.addBuffer(slangBuffer);
  }

  for (const auto &libDir : options.libDirs)
    driver.sourceLoader.addSearchDirectories(libDir);

  for (const auto &libExt : options.libExts)
    driver.sourceLoader.addSearchExtension(libExt);

  for (const auto &includeDir : options.includeDirs)
    if (driver.sourceManager.addUserDirectories(includeDir))
      return failure();

  for (const auto &includeSystemDir : options.includeSystemDirs)
    if (driver.sourceManager.addSystemDirectories(includeSystemDir))
      return failure();

  // These options are exposed on circt-verilog and should behave like slang's
  // standard command line parser, including support for comma-separated values.
  auto forEachCommaValue =
      [](const std::vector<std::string> &values, auto &&processOne) {
        for (const auto &value : values) {
          StringRef remaining = value;
          while (!remaining.empty()) {
            auto [part, rest] = remaining.split(',');
            auto token = part.trim();
            if (!token.empty())
              processOne(token);
            remaining = rest;
          }
        }
      };

  forEachCommaValue(options.suppressWarningsPaths, [&](StringRef pathPattern) {
    if (auto ec = driver.diagEngine.addIgnorePaths(pathPattern))
      emitWarning(UnknownLoc::get(mlirContext))
          << "--suppress-warnings path '" << pathPattern
          << "': " << ec.message();
  });

  forEachCommaValue(options.suppressMacroWarningsPaths,
                    [&](StringRef pathPattern) {
                      if (auto ec =
                              driver.diagEngine.addIgnoreMacroPaths(pathPattern))
                        emitWarning(UnknownLoc::get(mlirContext))
                            << "--suppress-macro-warnings path '" << pathPattern
                            << "': " << ec.message();
                    });

  forEachCommaValue(options.libraryFiles, [&](StringRef libraryPattern) {
    StringRef libraryName;
    StringRef filePattern = libraryPattern;
    auto eqPos = libraryPattern.find('=');
    if (eqPos != StringRef::npos) {
      libraryName = libraryPattern.take_front(eqPos).trim();
      filePattern = libraryPattern.drop_front(eqPos + 1).trim();
    }
    driver.sourceLoader.addLibraryFiles(libraryName, filePattern);
  });

  forEachCommaValue(options.libraryMapFiles, [&](StringRef mapPattern) {
    driver.sourceLoader.addLibraryMaps(mapPattern, {}, slang::Bag{});
  });

  // Populate the driver options.
  driver.options.excludeExts.insert(options.excludeExts.begin(),
                                    options.excludeExts.end());
  driver.options.ignoreDirectives = options.ignoreDirectives;

  driver.options.maxIncludeDepth = options.maxIncludeDepth;
  driver.options.defines = options.defines;
  driver.options.undefines = options.undefines;
  driver.options.librariesInheritMacros = options.librariesInheritMacros;

  driver.options.languageVersion = options.languageVersion;
  driver.options.maxParseDepth = options.maxParseDepth;
  driver.options.maxLexerErrors = options.maxLexerErrors;
  driver.options.numThreads = options.numThreads;
  driver.options.maxInstanceDepth = options.maxInstanceDepth;
  driver.options.maxGenerateSteps = options.maxGenerateSteps;
  driver.options.maxConstexprDepth = options.maxConstexprDepth;
  driver.options.maxConstexprSteps = options.maxConstexprSteps;
  driver.options.maxConstexprBacktrace = options.maxConstexprBacktrace;
  driver.options.maxInstanceArray = options.maxInstanceArray;

  driver.options.timeScale = options.timeScale;
  // Enable AllowUseBeforeDeclare by default — forward references are
  // ubiquitous in real SV code and accepted by VCS/Xcelium.  The explicit
  // CLI flag can still override this.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::AllowUseBeforeDeclare,
      options.allowUseBeforeDeclare.value_or(true));
  // Enable AllowUnnamedGenerate by default — references to implicit genblk
  // names are accepted by all major tools.
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::AllowUnnamedGenerate, true);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::IgnoreUnknownModules,
      options.ignoreUnknownModules);
  // Handle compat option - set VCS compatibility flags directly
  // We don't use slang's native compat mode because CIRCT bypasses
  // addStandardArgs() which initializes the flags map
  if (options.compat.has_value()) {
    auto compatStr = *options.compat;
    if (compatStr == "vcs" || compatStr == "all") {
      // VCS compatibility flags (from slang's VCS_COMP_FLAGS)
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowHierarchicalConst, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::RelaxEnumConversions, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowUseBeforeDeclare, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::RelaxStringConversions, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowRecursiveImplicitCall, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowBareValParamAssignment, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowSelfDeterminedStreamConcat, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowMergingAnsiPorts, true);
      // Also enable AllowVirtualIfaceWithOverride for Xcelium compatibility
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowVirtualIfaceWithOverride, true);
    }
    if (compatStr == "all") {
      // Additional flags for "all" compat mode
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowTopLevelIfacePorts, true);
      driver.options.compilationFlags.emplace(
          slang::ast::CompilationFlags::AllowUnnamedGenerate, true);
    }
  }
  // Explicit flag for virtual interface with override
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::AllowVirtualIfaceWithOverride,
      options.allowVirtualIfaceWithOverride);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::LintMode,
      options.mode == ImportVerilogOptions::Mode::OnlyLint);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::DisableInstanceCaching, false);
  driver.options.topModules = options.topModules;
  driver.options.paramOverrides = options.paramOverrides;
  forEachCommaValue(options.libraryOrder, [&](StringRef libraryName) {
    driver.options.libraryOrder.emplace_back(libraryName.str());
  });
  driver.options.defaultLibName = options.defaultLibName;

  driver.options.errorLimit = options.errorLimit;
  driver.options.warningOptions = options.warningOptions;

  driver.options.singleUnit = options.singleUnit;

  if (!driver.processOptions())
    return failure();

  // Set DynamicNotProcedural severity AFTER processOptions() so it doesn't
  // get overwritten by the default warning option processing.
  if (options.allowNonProceduralDynamic.value_or(false))
    driver.diagEngine.setSeverity(slang::diag::DynamicNotProcedural,
                                  slang::DiagnosticSeverity::Warning);

  // Downgrade out-of-bounds index/range accesses from error to warning by
  // default. Most tools (VCS, Xcelium, yosys) accept these and handle them
  // at runtime, while slang treats them as hard errors.
  driver.diagEngine.setSeverity(slang::diag::IndexOOB,
                                slang::DiagnosticSeverity::Warning);
  driver.diagEngine.setSeverity(slang::diag::RangeOOB,
                                slang::DiagnosticSeverity::Warning);

  return success();
}

/// Parse and elaborate the prepared source files, and populate the given MLIR
/// `module` with corresponding operations.
LogicalResult ImportDriver::importVerilog(ModuleOp module) {
  // Parse the input.
  auto parseTimer = ts.nest("Verilog parser");
  bool parseSuccess = driver.parseAllSources();
  parseTimer.stop();

  // Elaborate the input.
  auto compileTimer = ts.nest("Verilog elaboration");
  auto compilation = driver.createCompilation();

  // Register vendor-specific system functions as stubs.
  // $get_initial_random_seed is a Verilator extension that returns the
  // initial random seed. We stub it to return int (value 0 at runtime).
  compilation->addSystemSubroutine(
      std::make_shared<slang::ast::NonConstantFunction>(
          "$get_initial_random_seed", compilation->getIntType()));

  // $initstate is used in formal verification to indicate the initial state.
  // It returns 1 during the initial/reset state and 0 otherwise.
  // We stub it to return bit (value 0) since in simulation we are never in
  // the formal initial state.
  compilation->addSystemSubroutine(
      std::make_shared<slang::ast::NonConstantFunction>(
          "$initstate", compilation->getBitType()));

  for (auto &diag : compilation->getAllDiagnostics())
    driver.diagEngine.issue(diag);
  if (!parseSuccess || driver.diagEngine.getNumErrors() > 0)
    return failure();

  compileTimer.stop();

  // If we were only supposed to lint the input, return here. This leaves the
  // module empty, but any Slang linting messages got reported as diagnostics.
  if (options.mode == ImportVerilogOptions::Mode::OnlyLint)
    return success();

  // Traverse the parsed Verilog AST and map it to the equivalent CIRCT ops.
  mlirContext
      ->loadDialect<moore::MooreDialect, hw::HWDialect, cf::ControlFlowDialect,
                    func::FuncDialect, verif::VerifDialect, ltl::LTLDialect,
                    debug::DebugDialect, mlir::LLVM::LLVMDialect>();
  auto conversionTimer = ts.nest("Verilog to dialect mapping");
  Context context(options, *compilation, module, driver.sourceManager);
  if (failed(context.convertCompilation()))
    return failure();
  conversionTimer.stop();

  // Run the verifier on the constructed module to ensure it is clean.
  auto verifierTimer = ts.nest("Post-parse verification");
  if (failed(verify(module))) {
    // The verifier should have emitted diagnostics, but add a summary message
    // in case it didn't for some reason.
    mlir::emitError(UnknownLoc::get(mlirContext))
        << "generated MLIR module failed to verify; "
           "this is likely a bug in circt-verilog";
    return failure();
  }
  return success();
}

/// Preprocess the prepared source files and print them to the given output
/// stream.
LogicalResult ImportDriver::preprocessVerilog(llvm::raw_ostream &os) {
  auto parseTimer = ts.nest("Verilog preprocessing");

  // Run the preprocessor to completion across all sources previously added with
  // `pushSource`, report diagnostics, and print the output.
  auto preprocessAndPrint = [&](slang::parsing::Preprocessor &preprocessor) {
    slang::syntax::SyntaxPrinter output;
    output.setIncludeComments(false);
    while (true) {
      slang::parsing::Token token = preprocessor.next();
      output.print(token);
      if (token.kind == slang::parsing::TokenKind::EndOfFile)
        break;
    }

    for (auto &diag : preprocessor.getDiagnostics()) {
      if (diag.isError()) {
        driver.diagEngine.issue(diag);
        return failure();
      }
    }
    os << output.str();
    return success();
  };

  // Depending on whether the single-unit option is set, either add all source
  // files to a single preprocessor such that they share define macros and
  // directives, or create a separate preprocessor for each, such that each
  // source file is in its own compilation unit.
  auto optionBag = driver.createOptionBag();
  if (driver.options.singleUnit == true) {
    slang::BumpAllocator alloc;
    slang::Diagnostics diagnostics;
    slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                              diagnostics, optionBag);
    // Sources have to be pushed in reverse, as they form a stack in the
    // preprocessor. Last pushed source is processed first.
    auto sources = driver.sourceLoader.loadSources();
    for (auto &buffer : std::views::reverse(sources))
      preprocessor.pushSource(buffer);
    if (failed(preprocessAndPrint(preprocessor)))
      return failure();
  } else {
    for (auto &buffer : driver.sourceLoader.loadSources()) {
      slang::BumpAllocator alloc;
      slang::Diagnostics diagnostics;
      slang::parsing::Preprocessor preprocessor(driver.sourceManager, alloc,
                                                diagnostics, optionBag);
      preprocessor.pushSource(buffer);
      if (failed(preprocessAndPrint(preprocessor)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Parse the specified Verilog inputs into the specified MLIR context.
LogicalResult circt::importVerilog(SourceMgr &sourceMgr,
                                   MLIRContext *mlirContext, TimingScope &ts,
                                   ModuleOp module,
                                   const ImportVerilogOptions *options) {
  ImportDriver importDriver(mlirContext, ts, options);
  if (failed(importDriver.prepareDriver(sourceMgr)))
    return failure();
  return importDriver.importVerilog(module);
}

/// Run the files in a source manager through Slang's Verilog preprocessor and
/// emit the result to the given output stream.
LogicalResult circt::preprocessVerilog(SourceMgr &sourceMgr,
                                       MLIRContext *mlirContext,
                                       TimingScope &ts, llvm::raw_ostream &os,
                                       const ImportVerilogOptions *options) {
  ImportDriver importDriver(mlirContext, ts, options);
  if (failed(importDriver.prepareDriver(sourceMgr)))
    return failure();
  return importDriver.preprocessVerilog(os);
}

/// Entry point as an MLIR translation.
void circt::registerFromVerilogTranslation() {
  static TranslateToMLIRRegistration fromVerilog(
      "import-verilog", "import Verilog or SystemVerilog",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        TimingScope ts;
        OwningOpRef<ModuleOp> module(
            ModuleOp::create(UnknownLoc::get(context)));
        ImportVerilogOptions options;
        options.debugInfo = true;
        options.warningOptions.push_back("no-missing-top");
        if (failed(
                importVerilog(sourceMgr, context, ts, module.get(), &options)))
          module = {};
        return module;
      });
}

//===----------------------------------------------------------------------===//
// Pass Pipeline
//===----------------------------------------------------------------------===//

/// Optimize and simplify the Moore dialect IR.
void circt::populateVerilogToMoorePipeline(OpPassManager &pm) {
  {
    // Perform an initial cleanup and preprocessing across all
    // modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }

  pm.addPass(moore::createVTablesPass());

  // Remove unused symbols.
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(moore::createLowerConcatRefPass());

  {
    // Perform module-specific transformations.
    auto &modulePM = pm.nest<moore::SVModuleOp>();
    // TODO: Enable the following once it not longer interferes with @(...)
    // event control checks. The introduced dummy variables make the event
    // control observe a static local variable that never changes, instead of
    // observing a module-wide signal.
    // modulePM.addPass(moore::createSimplifyProceduresPass());
    modulePM.addPass(mlir::createSROA());
  }

  {
    // Perform a final cleanup across all modules/functions.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createMem2Reg());
    anyPM.addPass(mlir::createCSEPass());
    anyPM.addPass(mlir::createCanonicalizerPass());
  }
}

/// Convert Moore dialect IR into core dialect IR
void circt::populateMooreToCorePipeline(OpPassManager &pm) {
  // Perform the conversion.
  pm.addPass(createConvertMooreToCorePass());

  {
    // Conversion to the core dialects likely uncovers new canonicalization
    // opportunities.
    auto &anyPM = pm.nestAny();
    anyPM.addPass(mlir::createCSEPass());
    // Use the bottom-up canonicalizer with a rewrite cap to avoid pathological
    // behavior consuming unbounded memory/time in large imports.
    anyPM.addPass(circt::createBottomUpSimpleCanonicalizerPass());
  }
}

/// Convert LLHD dialect IR into core dialect IR
void circt::populateLlhdToCorePipeline(
    OpPassManager &pm, const LlhdToCorePipelineOptions &options) {
  // Inline function calls and lower SCF to CF.
  pm.addNestedPass<hw::HWModuleOp>(llhd::createWrapProceduralOpsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(llhd::createInlineCallsPass());

  // Run MooreToCore again to convert moore.wait_event/detect_event ops that
  // were in functions (e.g., interface tasks) and have now been inlined into
  // llhd.process ops. The first MooreToCore pass leaves these unconverted when
  // they're in func.func, using dynamic legality rules. After inlining, they
  // are now inside llhd.process and can be converted.
  pm.addPass(createConvertMooreToCorePass());

  pm.addPass(mlir::createSymbolDCEPass());

  // Simplify processes, replace signals with process results, and detect
  // registers.
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  // See https://github.com/llvm/circt/issues/8804.
  if (options.sroa) {
    modulePM.addPass(mlir::createSROA());
  }
  modulePM.addPass(llhd::createMem2RegPass());
  modulePM.addPass(llhd::createHoistSignalsPass());
  modulePM.addPass(llhd::createDeseqPass());
  modulePM.addPass(llhd::createLowerProcessesPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Unroll loops and remove control flow.
  modulePM.addPass(llhd::createUnrollLoopsPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());
  modulePM.addPass(llhd::createRemoveControlFlowPass());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Convert `arith.select` generated by some of the control flow canonicalizers
  // to `comb.mux`.
  modulePM.addPass(createMapArithToCombPass(true));

  // Simplify module-level signals.
  modulePM.addPass(llhd::createCombineDrivesPass());
  modulePM.addPass(llhd::createSig2Reg());
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());

  // Map `seq.firreg` with array type and `hw.array_inject` self-feedback to
  // `seq.firmem` ops.
  if (options.detectMemories) {
    modulePM.addPass(seq::createRegOfVecToMem());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(circt::createBottomUpSimpleCanonicalizerPass());
  }
}
