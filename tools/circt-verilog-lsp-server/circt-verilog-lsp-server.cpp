//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a utility to run CIRCT Verilog LSP server.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

#include "circt/Support/ResourceGuard.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/Program.h"

#include <cstdlib>

using namespace llvm::lsp;

int main(int argc, char **argv) {
  //===--------------------------------------------------------------------===//
  // LSP options
  //===--------------------------------------------------------------------===//

  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Info),
  };

  llvm::cl::opt<llvm::lsp::JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(llvm::lsp::JSONStreamStyle::Standard,
                                  "standard", "usual LSP protocol"),
                       clEnumValN(llvm::lsp::JSONStreamStyle::Delimited,
                                  "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(llvm::lsp::JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };

  //===--------------------------------------------------------------------===//
  // Include paths
  //===--------------------------------------------------------------------===//

  llvm::cl::list<std::string> libDirs{
      "y",
      llvm::cl::desc(
          "Library search paths, which will be searched for missing modules"),
      llvm::cl::value_desc("dir"), llvm::cl::Prefix};
  llvm::cl::alias libDirsLong{"libdir", llvm::cl::desc("Alias for -y"),
                              llvm::cl::aliasopt(libDirs), llvm::cl::NotHidden};

  llvm::cl::list<std::string> sourceLocationIncludeDirs(
      "source-location-include-dir",
      llvm::cl::desc("Root directory of file source locations"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  //===--------------------------------------------------------------------===//
  // Command files
  //===--------------------------------------------------------------------===//

  llvm::cl::list<std::string> commandFiles{
      "C",
      llvm::cl::desc(
          "One or more command files, which are independent compilation units "
          "where modules are automatically instantiated."),
      llvm::cl::value_desc("filename"), llvm::cl::Prefix};
  llvm::cl::alias commandFilesLong{
      "command-file", llvm::cl::desc("Alias for -C"),
      llvm::cl::aliasopt(commandFiles), llvm::cl::NotHidden};

  //===--------------------------------------------------------------------===//
  // UVM support
  //===--------------------------------------------------------------------===//

  llvm::cl::opt<std::string> uvmPath{
      "uvm-path",
      llvm::cl::desc("Path to UVM library source directory (e.g., ~/uvm-core/src). "
                     "If not specified, UVM_HOME environment variable will be checked."),
      llvm::cl::value_desc("directory"),
      llvm::cl::init("")};

  //===------------------------------------------------------------------===//
  // Debounce tuning
  //===------------------------------------------------------------------===//
  llvm::cl::opt<bool> noDebounce{
      "no-debounce",
      llvm::cl::desc("Disable debouncing (rebuild synchronously on change)"),
      llvm::cl::init(false)};

  llvm::cl::opt<unsigned> debounceMinMs{
      "debounce-min-ms",
      llvm::cl::desc("Minimum idle time (ms) before rebuild"),
      llvm::cl::init(150)};

  llvm::cl::opt<unsigned> debounceMaxMs{
      "debounce-max-ms",
      llvm::cl::desc("Maximum wait (ms) while edits continue; 0 = no cap"),
      llvm::cl::init(500)};

  //===--------------------------------------------------------------------===//
  // Testing
  //===--------------------------------------------------------------------===//

  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "Verilog LSP Language Server");
  circt::installResourceGuard();

  if (litTest) {
    inputStyle = llvm::lsp::JSONStreamStyle::Delimited;
    logLevel = llvm::lsp::Logger::Level::Debug;
    prettyPrint = true;
    noDebounce = true;
    debounceMinMs = 0;
    debounceMaxMs = 0;
  }

  // Configure the logger.
  llvm::lsp::Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  (void)llvm::sys::ChangeStdinToBinary();
  llvm::lsp::JSONTransport transport(stdin, llvm::outs(), inputStyle,
                                     prettyPrint);

  // Resolve UVM path: prefer command line, then environment variable.
  std::string resolvedUvmPath = uvmPath;
  if (resolvedUvmPath.empty()) {
    if (const char *envUvm = std::getenv("UVM_HOME"))
      resolvedUvmPath = std::string(envUvm) + "/src";
  }

  // Configure the servers and start the main language server.
  circt::lsp::VerilogServerOptions options(libDirs, sourceLocationIncludeDirs,
                                           commandFiles, resolvedUvmPath);
  circt::lsp::LSPServerOptions lspOptions(noDebounce, debounceMinMs,
                                          debounceMaxMs);
  return failed(
      circt::lsp::CirctVerilogLspServerMain(lspOptions, options, transport));
}
