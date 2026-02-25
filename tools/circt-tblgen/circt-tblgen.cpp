//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

int main(int argc, char **argv) {
#if defined(__EMSCRIPTEN__)
  // Wasm/node integrations may invoke this entrypoint repeatedly via callMain
  // in the same loaded module instance.
  llvm::cl::ResetAllOptionOccurrences();

  auto hasArg = [&](llvm::StringRef needle) {
    for (int i = 1; i < argc; ++i)
      if (argv[i] && llvm::StringRef(argv[i]) == needle)
        return true;
    return false;
  };

  const bool wantVersion = hasArg("--version");
  const bool wantHelp = hasArg("--help") || hasArg("-help") || hasArg("-h");
  const bool wantHelpHidden = hasArg("--help-hidden");
  const bool wantHelpList = hasArg("--help-list");
  const bool wantHelpListHidden = hasArg("--help-list-hidden");

  // Handle help/version locally in wasm mode so one-shot option paths do not
  // rely on process-style exits from LLVM's option handling.
  if (wantVersion) {
    llvm::cl::PrintVersionMessage();
    llvm::outs().flush();
    llvm::errs().flush();
    return 0;
  }
  if (wantHelp || wantHelpHidden || wantHelpList || wantHelpListHidden) {
    llvm::outs()
        << "OVERVIEW: CIRCT TableGen Generator\n\n"
        << "USAGE: circt-tblgen [generator] [options] <input file>\n\n"
        << "Common generators:\n"
        << "  -print-records    Print all records to stdout\n\n"
        << "Common options:\n"
        << "  -I <directory>    Add directory to include search path\n"
        << "  -o <filename>     Output filename ('-' for stdout)\n"
        << "  --version         Display version information\n"
        << "  --help            Display this help text\n";
    llvm::outs().flush();
    llvm::errs().flush();
    return 0;
  }
#endif

  return MlirTblgenMain(argc, argv);
}
