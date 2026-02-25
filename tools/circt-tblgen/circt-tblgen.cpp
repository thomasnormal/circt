//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
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

#if defined(__EMSCRIPTEN__)
namespace {
enum DeprecatedAction { None, Warn, Error };

static DeprecatedAction actionOnDeprecatedValue;
static const mlir::GenInfo *generator;

static bool findUse(const Init *field, const Init *deprecatedInit,
                    llvm::DenseMap<const Init *, bool> &known) {
  if (field == deprecatedInit)
    return true;

  auto it = known.find(field);
  if (it != known.end())
    return it->second;

  auto memoize = [&](bool val) {
    known[field] = val;
    return val;
  };

  if (auto *defInit = dyn_cast<DefInit>(field)) {
    // Recurse only through anonymous defs; non-anonymous defs are checked in
    // the main scan so users get one direct deprecation site each.
    if (!defInit->getDef()->isAnonymous())
      return false;

    return memoize(
        llvm::any_of(defInit->getDef()->getValues(), [&](const RecordVal &val) {
          return findUse(val.getValue(), deprecatedInit, known);
        }));
  }

  if (auto *dagInit = dyn_cast<DagInit>(field)) {
    if (findUse(dagInit->getOperator(), deprecatedInit, known))
      return memoize(true);
    return memoize(llvm::any_of(dagInit->getArgs(), [&](const Init *arg) {
      return findUse(arg, deprecatedInit, known);
    }));
  }

  if (const auto *listInit = dyn_cast<ListInit>(field))
    return memoize(llvm::any_of(listInit->getElements(), [&](const Init *jt) {
      return findUse(jt, deprecatedInit, known);
    }));

  return false;
}

static bool findUse(Record &record, const Init *deprecatedInit,
                    llvm::DenseMap<const Init *, bool> &known) {
  return llvm::any_of(record.getValues(), [&](const RecordVal &val) {
    return findUse(val.getValue(), deprecatedInit, known);
  });
}

static void warnOfDeprecatedUses(const RecordKeeper &records) {
  bool deprecatedDefsFound = false;
  for (auto &it : records.getDefs()) {
    const RecordVal *deprecatedField = it.second->getValue("odsDeprecated");
    if (!deprecatedField || !deprecatedField->getValue())
      continue;

    llvm::DenseMap<const Init *, bool> hasUse;
    if (auto *deprecatedMessage =
            dyn_cast<StringInit>(deprecatedField->getValue())) {
      for (auto &jt : records.getDefs()) {
        if (jt.second->isAnonymous())
          continue;
        if (findUse(*jt.second, it.second->getDefInit(), hasUse)) {
          PrintWarning(jt.second->getLoc(),
                       "Using deprecated def `" + it.first + "`");
          PrintNote(deprecatedMessage->getAsUnquotedString());
          deprecatedDefsFound = true;
        }
      }
    }
  }
  if (deprecatedDefsFound && actionOnDeprecatedValue == DeprecatedAction::Error)
    PrintFatalNote("Error'ing out due to deprecated defs");
}

static bool wasmTblgenMain(raw_ostream &os, const RecordKeeper &records) {
  if (actionOnDeprecatedValue != DeprecatedAction::None)
    warnOfDeprecatedUses(records);
  if (!generator) {
    os << records;
    return false;
  }
  return generator->invoke(records, os);
}

static int wasmTblgenDriverMain(int argc, char **argv) {
  static llvm::cl::opt<DeprecatedAction, true> actionOnDeprecated(
      "on-deprecated", llvm::cl::desc("Action to perform on deprecated def"),
      llvm::cl::values(
          clEnumValN(DeprecatedAction::None, "none", "No action"),
          clEnumValN(DeprecatedAction::Warn, "warn", "Warn on use"),
          clEnumValN(DeprecatedAction::Error, "error", "Error on use")),
      cl::location(actionOnDeprecatedValue), llvm::cl::init(Warn));

  static llvm::cl::opt<const mlir::GenInfo *, true, mlir::GenNameParser>
      generatorOpt(
      "", llvm::cl::desc("Generator to run"), cl::location(::generator));

  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(
      argv[0], [](TableGenOutputFiles &outFiles, const RecordKeeper &records) {
        std::string output;
        raw_string_ostream outputOS(output);
        bool result = wasmTblgenMain(outputOS, records);
        outFiles = {output, {}};
        return result;
      });
}
} // namespace
#endif

int main(int argc, char **argv) {
#if defined(__EMSCRIPTEN__)
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

#if defined(__EMSCRIPTEN__)
  return wasmTblgenDriverMain(argc, argv);
#else
  return MlirTblgenMain(argc, argv);
#endif
}
