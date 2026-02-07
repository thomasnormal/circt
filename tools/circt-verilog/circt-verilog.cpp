//===- circt-verilog.cpp - Getting Verilog into CIRCT ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to parse Verilog and SystemVerilog input
// files. This builds on CIRCT's ImportVerilog library, which ultimately relies
// on slang to do the heavy lifting.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Diagnostics.h"
#include "circt/Support/Passes.h"
#include "circt/Support/ResourceGuard.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include <cstdlib>

using namespace mlir;
using namespace circt;
namespace cl = llvm::cl;
using llvm::WithColor;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

namespace {
enum class Format {
  SV,
  MLIR,
};

enum class LoweringMode {
  OnlyPreprocess,
  OnlyLint,
  OnlyParse,
  OutputIRMoore,
  OutputIRLLHD,
  OutputIRHW,
  Full
};

struct CLOptions {
  cl::OptionCategory cat{"Verilog Frontend Options"};

  cl::opt<Format> format{
      "format", cl::desc("Input file format (auto-detected by default)"),
      cl::values(
          clEnumValN(Format::SV, "sv", "Parse as SystemVerilog files"),
          clEnumValN(Format::MLIR, "mlir", "Parse as MLIR or MLIRBC file")),
      cl::cat(cat)};

  cl::list<std::string> inputFilenames{cl::Positional,
                                       cl::desc("<input files>"), cl::cat(cat)};

  cl::opt<std::string> outputFilename{
      "o", cl::desc("Output filename (`-` for stdout)"),
      cl::value_desc("filename"), cl::init("-"), cl::cat(cat)};

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match expected-* lines on the "
               "corresponding line"),
      cl::init(false), cl::Hidden, cl::cat(cat)};

  cl::opt<bool> verbosePassExecutions{
      "verbose-pass-executions",
      cl::desc("Print passes as they are being executed"), cl::init(false),
      cl::cat(cat)};

  cl::opt<LoweringMode> loweringMode{
      cl::desc("Specify how to process the input:"),
      cl::values(
          clEnumValN(
              LoweringMode::OnlyPreprocess, "E",
              "Only run the preprocessor (and print preprocessed files)"),
          clEnumValN(LoweringMode::OnlyLint, "lint-only",
                     "Only lint the input, without elaboration and mapping to "
                     "CIRCT IR"),
          clEnumValN(LoweringMode::OnlyParse, "parse-only",
                     "Only parse and elaborate the input, without mapping to "
                     "CIRCT IR"),
          clEnumValN(LoweringMode::OutputIRMoore, "ir-moore",
                     "Run the entire pass manager to just before MooreToCore "
                     "conversion, and emit the resulting Moore dialect IR"),
          clEnumValN(
              LoweringMode::OutputIRLLHD, "ir-llhd",
              "Run the entire pass manager to just before the LLHD pipeline "
              ", and emit the resulting LLHD+Core dialect IR"),
          clEnumValN(LoweringMode::OutputIRHW, "ir-hw",
                     "Run the MooreToCore and LLHD lowering pipelines, and "
                     "emit the resulting HW/Comb/Seq dialect IR")),
      cl::init(LoweringMode::Full), cl::cat(cat)};

  cl::opt<bool> debugInfo{"g", cl::desc("Generate debug information"),
                          cl::cat(cat)};

  cl::opt<bool> lowerAlwaysAtStarAsComb{
      "always-at-star-as-comb",
      cl::desc("Interpret `always @(*)` as `always_comb`"), cl::init(true),
      cl::cat(cat)};

  cl::opt<bool> detectMemories{
      "detect-memories",
      cl::desc("Detect memories and lower them to `seq.firmem`"),
      cl::init(true), cl::cat(cat)};

  cl::opt<bool> sroa{
      "sroa",
      cl::desc("Destructure arrays and structs into individual signals."),
      cl::init(false), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Include paths
  //===--------------------------------------------------------------------===//

  cl::list<std::string> includeDirs{
      "I", cl::desc("Additional include search paths"), cl::value_desc("dir"),
      cl::Prefix, cl::cat(cat)};
  cl::alias includeDirsLong{"include-dir", cl::desc("Alias for -I"),
                            cl::aliasopt(includeDirs), cl::NotHidden,
                            cl::cat(cat)};

  cl::opt<std::string> uvmPath{
      "uvm-path",
      cl::desc("Path to UVM library root (defaults to $UVM_HOME)"),
      cl::value_desc("dir"), cl::cat(cat)};
  cl::opt<bool> noUvmAutoInclude{
      "no-uvm-auto-include",
      cl::desc("Disable automatic injection of UVM package and macros"),
      cl::init(false), cl::cat(cat)};

  cl::list<std::string> includeSystemDirs{
      "isystem", cl::desc("Additional system include search paths"),
      cl::value_desc("dir"), cl::cat(cat)};

  cl::list<std::string> libDirs{
      "y",
      cl::desc(
          "Library search paths, which will be searched for missing modules"),
      cl::value_desc("dir"), cl::Prefix, cl::cat(cat)};
  cl::alias libDirsLong{"libdir", cl::desc("Alias for -y"),
                        cl::aliasopt(libDirs), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> libExts{
      "Y", cl::desc("Additional library file extensions to search"),
      cl::value_desc("ext"), cl::Prefix, cl::cat(cat)};
  cl::alias libExtsLong{"libext", cl::desc("Alias for -Y"),
                        cl::aliasopt(libExts), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> excludeExts{
      "exclude-ext",
      cl::desc("Exclude provided source files with these extensions"),
      cl::value_desc("ext"), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Preprocessor
  //===--------------------------------------------------------------------===//

  cl::list<std::string> defines{
      "D",
      cl::desc("Define <macro> to <value> (or 1 if <value> omitted) in all "
               "source files"),
      cl::value_desc("<macro>=<value>"), cl::Prefix, cl::cat(cat)};
  cl::alias definesLong{"define-macro", cl::desc("Alias for -D"),
                        cl::aliasopt(defines), cl::NotHidden, cl::cat(cat)};

  cl::list<std::string> undefines{
      "U", cl::desc("Undefine macro name at the start of all source files"),
      cl::value_desc("macro"), cl::Prefix, cl::cat(cat)};
  cl::alias undefinesLong{"undefine-macro", cl::desc("Alias for -U"),
                          cl::aliasopt(undefines), cl::NotHidden, cl::cat(cat)};

  cl::opt<uint32_t> maxIncludeDepth{
      "max-include-depth",
      cl::desc("Maximum depth of nested include files allowed"),
      cl::value_desc("depth"), cl::cat(cat)};

  cl::opt<bool> librariesInheritMacros{
      "libraries-inherit-macros",
      cl::desc("If true, library files will inherit macro definitions from the "
               "primary source files. --single-unit must also be passed when "
               "this option is used."),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> disableLocalIncludes{
      "disable-local-includes",
      cl::desc("Disable local include lookup relative to the including file"),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> enableLegacyProtect{
      "enable-legacy-protect",
      cl::desc("Enable legacy protected envelope directives"),
      cl::init(false), cl::cat(cat)};

  cl::list<std::string> translateOffOptions{
      "translate-off-format",
      cl::desc("Comment directive format marking disabled source text as "
               "<common>,<start>,<end>"),
      cl::value_desc("<common>,<start>,<end>"), cl::cat(cat)};

  cl::list<std::string> mapKeywordVersion{
      "map-keyword-version",
      cl::desc("Parse matching files with a specific keyword version as "
               "<keyword-version>+<file-pattern>[,...]"),
      cl::value_desc("<keyword-version>+<file-pattern>[,...]"), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Compilation
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> timeScale{
      "timescale",
      cl::desc("Default time scale to use for design elements that don't "
               "specify one explicitly"),
      cl::value_desc("<base>/<precision>"), cl::cat(cat)};

  cl::opt<std::string> minTypMax{
      "timing", cl::desc("Select min:typ:max value for compilation"),
      cl::value_desc("min|typ|max"), cl::cat(cat)};
  cl::alias minTypMaxShort{"T", cl::desc("Alias for --timing"),
                           cl::aliasopt(minTypMax), cl::NotHidden,
                           cl::cat(cat)};

  cl::opt<std::string> languageVersion{
      "language-version",
      cl::desc("Set the SystemVerilog language keyword version"),
      cl::value_desc("version"), cl::cat(cat)};

  cl::opt<uint32_t> maxParseDepth{
      "max-parse-depth",
      cl::desc("Maximum parser call stack depth before erroring"),
      cl::value_desc("depth"), cl::cat(cat)};

  cl::opt<uint32_t> maxLexerErrors{
      "max-lexer-errors",
      cl::desc("Maximum lexer errors before aborting parse"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<uint32_t> numThreads{
      "num-threads",
      cl::desc("Number of parser threads to use"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<uint32_t> maxInstanceDepth{
      "max-instance-depth",
      cl::desc("Maximum nested instance depth before erroring"),
      cl::value_desc("depth"), cl::cat(cat)};

  cl::opt<uint32_t> maxGenerateSteps{
      "max-generate-steps",
      cl::desc("Maximum steps to expand one generate construct"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<uint32_t> maxConstexprDepth{
      "max-constexpr-depth",
      cl::desc("Maximum constexpr call depth before erroring"),
      cl::value_desc("depth"), cl::cat(cat)};

  cl::opt<uint32_t> maxConstexprSteps{
      "max-constexpr-steps",
      cl::desc("Maximum constexpr evaluation steps before erroring"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<uint32_t> maxConstexprBacktrace{
      "max-constexpr-backtrace",
      cl::desc("Maximum constexpr callstack frames shown in diagnostics"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<uint32_t> maxInstanceArray{
      "max-instance-array",
      cl::desc("Maximum allowed elements in an instance array"),
      cl::value_desc("count"), cl::cat(cat)};

  cl::opt<bool> allowUseBeforeDeclare{
      "allow-use-before-declare",
      cl::desc(
          "Don't issue an error for use of names before their declarations."),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> ignoreUnknownModules{
      "ignore-unknown-modules",
      cl::desc("Don't issue an error for instantiations of unknown modules, "
               "interface, and programs."),
      cl::init(false), cl::cat(cat)};

  cl::opt<std::string> compat{
      "compat",
      cl::desc("Attempt to increase compatibility with the specified tool. "
               "Valid values are 'vcs' for Synopsys VCS or 'all' for all "
               "compatibility options."),
      cl::value_desc("tool"), cl::cat(cat)};

  cl::opt<bool> allowVirtualIfaceWithOverride{
      "allow-virtual-iface-with-override",
      cl::desc("Allow interface instances that are bind/defparam targets to "
               "be assigned to virtual interfaces (matches Xcelium behavior)."),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> ignoreTimingControls{
      "ignore-timing-controls",
      cl::desc("Ignore timing controls (event/delay waits) during lowering"),
      cl::init(false), cl::cat(cat)};

  cl::opt<bool> allowNonProceduralDynamic{
      "allow-nonprocedural-dynamic",
      cl::desc("Allow dynamic type members in non-procedural contexts "
               "(converts to always_comb blocks). Use --allow-nonprocedural-dynamic=false "
               "for strict SystemVerilog semantics."),
      cl::init(true), cl::cat(cat)};

  cl::list<std::string> topModules{
      "top",
      cl::desc("One or more top-level modules to instantiate (instead of "
               "figuring it out automatically)"),
      cl::value_desc("name"), cl::cat(cat)};

  cl::list<std::string> paramOverrides{
      "G",
      cl::desc("One or more parameter overrides to apply when instantiating "
               "top-level modules"),
      cl::value_desc("<name>=<value>"), cl::Prefix, cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // Diagnostics control
  //===--------------------------------------------------------------------===//

  cl::list<std::string> warningOptions{
      "W", cl::desc("Control the specified warning"), cl::value_desc("warning"),
      cl::Prefix, cl::cat(cat)};

  cl::opt<uint32_t> errorLimit{
      "error-limit",
      cl::desc("Limit on the number of errors that will be printed. Setting "
               "this to zero will disable the limit."),
      cl::value_desc("limit"), cl::cat(cat)};

  cl::list<std::string> suppressWarningsPaths{
      "suppress-warnings",
      cl::desc("One or more paths in which to suppress warnings"),
      cl::value_desc("filename"), cl::cat(cat)};

  cl::list<std::string> suppressMacroWarningsPaths{
      "suppress-macro-warnings",
      cl::desc("One or more paths in which to suppress warnings originating "
               "in macro expansions"),
      cl::value_desc("filename"), cl::cat(cat)};

  cl::opt<std::string> diagnosticFormat{
      "diagnostic-format",
      cl::desc("Output format for diagnostics (terminal, plain, json, sarif)"),
      cl::value_desc("format"), cl::init("terminal"), cl::cat(cat)};

  cl::opt<bool> noColor{
      "no-color",
      cl::desc("Disable colored output in diagnostics"),
      cl::init(false), cl::cat(cat)};

  cl::opt<std::string> diagnosticOutput{
      "diagnostic-output",
      cl::desc("Output file for diagnostics (default: stderr)"),
      cl::value_desc("filename"), cl::init(""), cl::cat(cat)};

  //===--------------------------------------------------------------------===//
  // File lists
  //===--------------------------------------------------------------------===//

  cl::opt<bool> singleUnit{
      "single-unit",
      cl::desc("Treat all input files as a single compilation unit"),
      cl::init(false), cl::cat(cat)};

  cl::list<std::string> libraryFiles{
      "l",
      cl::desc(
          "One or more library files, which are separate compilation units "
          "where modules are not automatically instantiated."),
      cl::value_desc("filename"), cl::Prefix, cl::cat(cat)};

  cl::list<std::string> libraryMapFiles{
      "libmap",
      cl::desc("One or more library map files to parse for library mappings"),
      cl::value_desc("filename"), cl::cat(cat)};

  cl::list<std::string> libraryOrder{
      "L",
      cl::desc("A list of library names controlling module lookup priority"),
      cl::value_desc("library"), cl::Prefix, cl::cat(cat)};

  cl::opt<std::string> defaultLibName{
      "defaultLibName", cl::desc("Set the default source library name"),
      cl::value_desc("name"), cl::cat(cat)};

  cl::list<std::string> commandFiles{
      "C",
      cl::desc(
          "One or more command files, which are independent compilation units "
          "where modules are automatically instantiated."),
      cl::value_desc("filename"), cl::Prefix, cl::cat(cat)};
};
} // namespace

static CLOptions opts;

static bool hasUvmPkgInput() {
  for (const auto &inputFilename : opts.inputFilenames) {
    if (llvm::sys::path::filename(inputFilename) == "uvm_pkg.sv")
      return true;
  }
  return false;
}

static bool hasUvmMacrosInput() {
  for (const auto &inputFilename : opts.inputFilenames) {
    if (llvm::sys::path::filename(inputFilename) == "uvm_macros.svh")
      return true;
  }
  return false;
}

static void addUvmSupportIfAvailable() {
  if (opts.noUvmAutoInclude)
    return;

  if (opts.format != Format::SV)
    return;

  // Only warn about missing UVM if the user explicitly pointed us at a UVM
  // installation (via `--uvm-path` or `UVM_HOME`). Otherwise, stay silent: most
  // CIRCT users are not compiling UVM testbenches, and warning spam breaks
  // non-UVM regression tests that FileCheck stdout/stderr.
  const bool uvmExplicitlyRequested =
      !opts.uvmPath.empty() || (std::getenv("UVM_HOME") != nullptr);

  if (opts.uvmPath.empty()) {
    if (const char *uvmHome = std::getenv("UVM_HOME"))
      opts.uvmPath = uvmHome;
  }

  // Auto-discover bundled UVM library relative to the binary location.
  // The bundled copy lives at lib/Runtime/uvm-core/src/ in the source tree,
  // which is <binary_dir>/../../lib/Runtime/uvm-core/ relative to bin/.
  if (opts.uvmPath.empty()) {
    auto mainExe = llvm::sys::fs::getMainExecutable(
        "circt-verilog", (void *)&addUvmSupportIfAvailable);
    if (!mainExe.empty()) {
      llvm::SmallString<256> binDir(mainExe);
      llvm::sys::path::remove_filename(binDir); // remove binary name
      // Try <bin>/../lib/Runtime/uvm-core/src/ (install layout)
      llvm::SmallString<256> candidate(binDir);
      llvm::sys::path::append(candidate, "..");
      llvm::sys::path::append(candidate, "lib");
      llvm::sys::path::append(candidate, "Runtime");
      llvm::sys::path::append(candidate, "uvm-core");
      llvm::sys::path::append(candidate, "src");
      llvm::sys::path::append(candidate, "uvm_pkg.sv");
      llvm::sys::fs::make_absolute(candidate);
      if (llvm::sys::fs::exists(candidate)) {
        candidate = binDir;
        llvm::sys::path::append(candidate, "..");
        llvm::sys::path::append(candidate, "lib");
        llvm::sys::path::append(candidate, "Runtime");
        llvm::sys::path::append(candidate, "uvm-core");
        llvm::sys::fs::make_absolute(candidate);
        opts.uvmPath = std::string(candidate);
      }
    }
    // Also search ~/uvm-core as a fallback.
    if (opts.uvmPath.empty()) {
      const char *home = std::getenv("HOME");
      if (home) {
        llvm::SmallString<256> homeCandidate(home);
        llvm::sys::path::append(homeCandidate, "uvm-core", "src",
                                "uvm_pkg.sv");
        if (llvm::sys::fs::exists(homeCandidate)) {
          homeCandidate = std::string(home) + "/uvm-core";
          opts.uvmPath = std::string(homeCandidate);
        }
      }
    }
  }

  std::string uvmPkgPath;
  std::string uvmMacrosPath;
  std::string uvmIncludeDir;
  if (!opts.uvmPath.empty()) {
    llvm::SmallString<256> candidate(opts.uvmPath);
    llvm::sys::path::append(candidate, "uvm_pkg.sv");
    if (llvm::sys::fs::exists(candidate)) {
      uvmPkgPath = candidate.str().str();
      llvm::SmallString<256> macrosCandidate(opts.uvmPath);
      llvm::sys::path::append(macrosCandidate, "uvm_macros.svh");
      if (llvm::sys::fs::exists(macrosCandidate))
        uvmMacrosPath = macrosCandidate.str().str();
      uvmIncludeDir = opts.uvmPath;
    } else {
      candidate = opts.uvmPath;
      llvm::sys::path::append(candidate, "src", "uvm_pkg.sv");
      if (llvm::sys::fs::exists(candidate)) {
        uvmPkgPath = candidate.str().str();
        llvm::SmallString<256> includeDir(opts.uvmPath);
        llvm::sys::path::append(includeDir, "src");
        uvmIncludeDir = includeDir.str().str();
        llvm::SmallString<256> macrosCandidate(opts.uvmPath);
        llvm::sys::path::append(macrosCandidate, "src", "uvm_macros.svh");
        if (llvm::sys::fs::exists(macrosCandidate))
          uvmMacrosPath = macrosCandidate.str().str();
      }
    }
  }

  if (uvmPkgPath.empty()) {
    if (uvmExplicitlyRequested) {
      // No UVM library found. Print a warning to help users find uvm-core.
      llvm::errs()
          << "warning: UVM library not found. To use UVM, either:\n"
          << "  1. Set UVM_HOME environment variable to your uvm-core directory\n"
          << "  2. Use --uvm-path=<path> to specify the UVM library location\n"
          << "  3. Use --no-uvm-auto-include to disable UVM auto-inclusion\n"
          << "  Recommended: Use Accellera's uvm-core from "
             "https://github.com/accellera-official/uvm-core\n";
    }
    return;
  }

  if (!uvmIncludeDir.empty() &&
      llvm::find(opts.includeDirs, uvmIncludeDir) == opts.includeDirs.end())
    opts.includeDirs.push_back(uvmIncludeDir);

  if (!uvmMacrosPath.empty() && !hasUvmMacrosInput()) {
    opts.inputFilenames.insert(opts.inputFilenames.begin(), uvmMacrosPath);
    if (opts.singleUnit.getNumOccurrences() == 0)
      opts.singleUnit = true;
  }

  if (!hasUvmPkgInput()) {
    auto insertIt = opts.inputFilenames.begin();
    if (!uvmMacrosPath.empty() && hasUvmMacrosInput())
      ++insertIt;
    opts.inputFilenames.insert(insertIt, uvmPkgPath);
  }
}

/// Populate the given pass manager with transformations as configured by the
/// command line options.
static void populatePasses(PassManager &pm) {
  populateVerilogToMoorePipeline(pm);
  if (opts.loweringMode == LoweringMode::OutputIRMoore)
    return;
  populateMooreToCorePipeline(pm);
  if (opts.loweringMode == LoweringMode::OutputIRLLHD)
    return;
  // OutputIRHW and Full modes both require LLHD lowering to convert
  // llhd.process and other LLHD ops into HW/Comb/Seq dialect ops.
  // This is necessary for simulation via arcilator.
  LlhdToCorePipelineOptions options;
  options.detectMemories = opts.detectMemories;
  options.sroa = opts.sroa;
  populateLlhdToCorePipeline(pm, options);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

static LogicalResult executeWithSources(MLIRContext *context,
                                        llvm::SourceMgr &sourceMgr) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Map the command line options to `ImportVerilog`'s conversion options.
  ImportVerilogOptions options;
  if (opts.loweringMode == LoweringMode::OnlyLint)
    options.mode = ImportVerilogOptions::Mode::OnlyLint;
  else if (opts.loweringMode == LoweringMode::OnlyParse)
    options.mode = ImportVerilogOptions::Mode::OnlyParse;
  options.debugInfo = opts.debugInfo;
  options.lowerAlwaysAtStarAsComb = opts.lowerAlwaysAtStarAsComb;

  options.includeDirs = opts.includeDirs;
  options.includeSystemDirs = opts.includeSystemDirs;
  options.libDirs = opts.libDirs;
  options.libExts = opts.libExts;
  options.excludeExts = opts.excludeExts;

  options.defines = opts.defines;
  options.undefines = opts.undefines;
  if (opts.maxIncludeDepth.getNumOccurrences() > 0)
    options.maxIncludeDepth = opts.maxIncludeDepth;
  options.librariesInheritMacros = opts.librariesInheritMacros;
  options.disableLocalIncludes = opts.disableLocalIncludes;
  options.enableLegacyProtect = opts.enableLegacyProtect;
  options.translateOffOptions = opts.translateOffOptions;
  options.keywordVersionMappings = opts.mapKeywordVersion;

  if (opts.languageVersion.getNumOccurrences() > 0)
    options.languageVersion = opts.languageVersion;
  if (opts.maxParseDepth.getNumOccurrences() > 0)
    options.maxParseDepth = opts.maxParseDepth;
  if (opts.maxLexerErrors.getNumOccurrences() > 0)
    options.maxLexerErrors = opts.maxLexerErrors;
  if (opts.numThreads.getNumOccurrences() > 0)
    options.numThreads = opts.numThreads;
  if (opts.maxInstanceDepth.getNumOccurrences() > 0)
    options.maxInstanceDepth = opts.maxInstanceDepth;
  if (opts.maxGenerateSteps.getNumOccurrences() > 0)
    options.maxGenerateSteps = opts.maxGenerateSteps;
  if (opts.maxConstexprDepth.getNumOccurrences() > 0)
    options.maxConstexprDepth = opts.maxConstexprDepth;
  if (opts.maxConstexprSteps.getNumOccurrences() > 0)
    options.maxConstexprSteps = opts.maxConstexprSteps;
  if (opts.maxConstexprBacktrace.getNumOccurrences() > 0)
    options.maxConstexprBacktrace = opts.maxConstexprBacktrace;
  if (opts.maxInstanceArray.getNumOccurrences() > 0)
    options.maxInstanceArray = opts.maxInstanceArray;

  if (opts.timeScale.getNumOccurrences() > 0)
    options.timeScale = opts.timeScale;
  if (opts.minTypMax.getNumOccurrences() > 0)
    options.minTypMax = opts.minTypMax;
  if (opts.allowUseBeforeDeclare.getNumOccurrences() > 0)
    options.allowUseBeforeDeclare = opts.allowUseBeforeDeclare;
  options.ignoreUnknownModules = opts.ignoreUnknownModules;
  if (opts.compat.getNumOccurrences() > 0)
    options.compat = opts.compat;
  options.allowVirtualIfaceWithOverride = opts.allowVirtualIfaceWithOverride;
  options.ignoreTimingControls = opts.ignoreTimingControls;
  options.allowNonProceduralDynamic = opts.allowNonProceduralDynamic;
  if (opts.loweringMode != LoweringMode::OnlyLint)
    options.topModules = opts.topModules;
  options.paramOverrides = opts.paramOverrides;

  options.warningOptions = opts.warningOptions;
  if (opts.errorLimit.getNumOccurrences() > 0)
    options.errorLimit = opts.errorLimit;
  options.suppressWarningsPaths = opts.suppressWarningsPaths;
  options.suppressMacroWarningsPaths = opts.suppressMacroWarningsPaths;

  options.singleUnit = opts.singleUnit;
  options.libraryFiles = opts.libraryFiles;
  options.libraryMapFiles = opts.libraryMapFiles;
  options.libraryOrder = opts.libraryOrder;
  if (opts.defaultLibName.getNumOccurrences() > 0)
    options.defaultLibName = opts.defaultLibName;
  options.commandFiles = opts.commandFiles;

  // Open the output file.
  std::string errorMessage;
  auto outputFile = openOutputFile(opts.outputFilename, &errorMessage);
  if (!outputFile) {
    WithColor::error() << errorMessage << "\n";
    return failure();
  }

  // Parse the input as SystemVerilog or MLIR file.
  OwningOpRef<ModuleOp> module;
  switch (opts.format) {
  case Format::SV: {
    // If the user requested for the files to be only preprocessed, do so and
    // print the results to the configured output file.
    if (opts.loweringMode == LoweringMode::OnlyPreprocess) {
      auto result =
          preprocessVerilog(sourceMgr, context, ts, outputFile->os(), &options);
      if (succeeded(result))
        outputFile->keep();
      return result;
    }

    // Parse the Verilog input into an MLIR module.
    module = ModuleOp::create(UnknownLoc::get(context));
    if (failed(importVerilog(sourceMgr, context, ts, module.get(), &options)))
      return failure();

    // If the user requested for the files to be only linted, the module remains
    // empty and there is nothing left to do.
    if (opts.loweringMode == LoweringMode::OnlyLint)
      return success();
  } break;

  case Format::MLIR: {
    auto parserTimer = ts.nest("MLIR Parser");
    module = parseSourceFile<ModuleOp>(sourceMgr, context);
  } break;
  }
  if (!module)
    return failure();

  // If the user requested anything besides simply parsing the input, run the
  // appropriate transformation passes according to the command line options.
  if (opts.loweringMode != LoweringMode::OnlyParse) {
    PassManager pm(context);
    pm.enableVerifier(true);
    pm.enableTiming(ts);
    if (opts.verbosePassExecutions)
      pm.addInstrumentation(
          std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
              "circt-verilog"));
    if (failed(applyPassManagerCLOptions(pm)))
      return failure();
    populatePasses(pm);
    if (failed(pm.run(module.get())))
      return failure();
  }

  // Print the final MLIR.
  auto outputTimer = ts.nest("MLIR Printer");
  module->print(outputFile->os());
  outputFile->keep();
  return success();
}

static LogicalResult execute(MLIRContext *context) {
  // Default to reading from stdin if no files were provided except if
  // commandfiles were.
  if (opts.inputFilenames.empty() && opts.commandFiles.empty()) {
    opts.inputFilenames.push_back("-");
  }

  // Auto-detect the input format if it was not explicitly specified.
  if (opts.format.getNumOccurrences() == 0) {
    std::optional<Format> detectedFormat = std::nullopt;
    for (const auto &inputFilename : opts.inputFilenames) {
      std::optional<Format> format = std::nullopt;
      auto name = StringRef(inputFilename);
      if (name.ends_with(".v") || name.ends_with(".sv") ||
          name.ends_with(".vh") || name.ends_with(".svh"))
        format = Format::SV;
      else if (name.ends_with(".mlir") || name.ends_with(".mlirbc"))
        format = Format::MLIR;
      if (!format)
        continue;
      if (detectedFormat && format != detectedFormat) {
        detectedFormat = std::nullopt;
        break;
      }
      detectedFormat = format;
    }
  if (!detectedFormat) {
      if (!opts.commandFiles.empty()) {
        detectedFormat = Format::SV;
      } else {
        WithColor::error() << "cannot auto-detect input format; use --format\n";
        return failure();
      }
    }
    opts.format = *detectedFormat;
  }

  addUvmSupportIfAvailable();

  // Open the input files.
  llvm::SourceMgr sourceMgr;
  DenseSet<StringRef> seenInputFilenames;
  for (const auto &inputFilename : opts.inputFilenames) {
    // Skip empty filenames that might result from command line parsing issues.
    if (inputFilename.empty()) {
      WithColor::warning() << "ignoring empty input filename\n";
      continue;
    }

    // Don't add the same file multiple times.
    if (!seenInputFilenames.insert(inputFilename).second) {
      WithColor::warning() << "redundant input file `" << inputFilename
                           << "`\n";
      continue;
    }

    std::string errorMessage;
    auto buffer = openInputFile(inputFilename, &errorMessage);
    if (!buffer) {
      WithColor::error() << errorMessage << "\n";
      return failure();
    }
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  }

  // Call `executeWithSources` with either the regular diagnostic handler, or,
  // if `--verify-diagnostics` is set, with the verifying handler.
  if (opts.verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler handler(sourceMgr, context);
    context->printOpOnDiagnostic(false);
    (void)executeWithSources(context, sourceMgr);
    return handler.verify();
  }

  // Check if the user requested a specific diagnostic format.
  auto diagFormat = parseDiagnosticOutputFormat(opts.diagnosticFormat);
  if (!diagFormat) {
    WithColor::error() << "invalid diagnostic format: " << opts.diagnosticFormat
                       << "\n";
    return failure();
  }

  // Use rich diagnostic output if a specific format was requested.
  if (*diagFormat != DiagnosticOutputFormat::Terminal || opts.noColor ||
      !opts.diagnosticOutput.empty()) {
    // Open diagnostic output file if specified.
    std::unique_ptr<llvm::ToolOutputFile> diagFile;
    llvm::raw_ostream *diagOS = &llvm::errs();
    if (!opts.diagnosticOutput.empty()) {
      std::string errorMessage;
      diagFile = openOutputFile(opts.diagnosticOutput, &errorMessage);
      if (!diagFile) {
        WithColor::error() << errorMessage << "\n";
        return failure();
      }
      diagOS = &diagFile->os();
    }

    // Create the diagnostic printer.
    DiagnosticPrinter printer(*diagOS, *diagFormat, &sourceMgr);
    printer.setUseColors(!opts.noColor);

    // Create the rich diagnostic handler.
    RichDiagnosticHandler handler(context, printer);

    auto result = executeWithSources(context, sourceMgr);

    // Flush the diagnostic output.
    printer.flush();

    // Keep the diagnostic output file if we created one.
    if (diagFile)
      diagFile->keep();

    return result;
  }

  // Fall back to the standard MLIR diagnostic handler.
  SourceMgrDiagnosticHandler handler(sourceMgr, context);
  return executeWithSources(context, sourceMgr);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circtBugReportMsg);

  // Print the CIRCT and Slang versions when requested.
  cl::AddExtraVersionPrinter([](raw_ostream &os) {
    os << getCirctVersion() << '\n';
    os << getSlangVersion() << '\n';
  });

  // Register any pass manager command line options.
  llhd::registerPasses();
  moore::registerPasses();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv,
                              "Verilog and SystemVerilog frontend\n");

  // Set a default wall-clock timeout for circt-verilog. Compilation of even
  // very large SystemVerilog designs should complete well within 10 minutes;
  // anything longer almost certainly indicates an infinite loop or combinational
  // explosion. RSS limits are handled by installResourceGuard()'s smart
  // defaults (40% of system RAM, capped at 10 GB). Using overwrite=0 lets
  // explicit user settings (env vars or CLI flags) take precedence.
  ::setenv("CIRCT_MAX_WALL_MS", "600000", /*overwrite=*/0); // 10 min timeout
  circt::installResourceGuard();

  // Register the dialects.
  // clang-format off
  DialectRegistry registry;
  registry.insert<
    arith::ArithDialect,
    cf::ControlFlowDialect,
    comb::CombDialect,
    debug::DebugDialect,
    func::FuncDialect,
    hw::HWDialect,
    llhd::LLHDDialect,
    LLVM::LLVMDialect,
    moore::MooreDialect,
    scf::SCFDialect,
    seq::SeqDialect,
    sim::SimDialect,
    verif::VerifDialect
  >();
  // clang-format on

  // Perform the actual work and use "exit" to avoid slow context teardown.
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  MLIRContext context(registry);
  exit(failed(execute(&context)));
}
