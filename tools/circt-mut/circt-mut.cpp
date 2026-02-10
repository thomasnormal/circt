//===- circt-mut.cpp - Mutation workflow frontend ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// circt-mut is a stable CLI frontend for mutation workflows. Subcommands are
// migrated from utility scripts into native code incrementally.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

static void printHelp(raw_ostream &os) {
  os << "circt-mut - CIRCT mutation workflow frontend\n\n";
  os << "Usage:\n";
  os << "  circt-mut <subcommand> [args...]\n\n";
  os << "Subcommands:\n";
  os << "  init      Initialize a mutation campaign project template\n";
  os << "  run       Run campaign from circt-mut.toml project config\n";
  os << "  report    Aggregate campaign metrics from cover/matrix outputs\n";
  os << "  cover     Run mutation coverage flow (run_mutation_cover.sh)\n";
  os << "  matrix    Run mutation lane matrix flow (run_mutation_matrix.sh)\n";
  os << "  generate  Generate mutation lists (native path; script fallback)\n\n";
  os << "Environment:\n";
  os << "  CIRCT_MUT_SCRIPTS_DIR  Override script directory location\n\n";
  os << "Examples:\n";
  os << "  circt-mut init --project-dir mut-campaign\n";
  os << "  circt-mut run --project-dir mut-campaign --mode all\n";
  os << "  circt-mut report --project-dir mut-campaign --mode all\n";
  os << "  circt-mut cover --help\n";
  os << "  circt-mut matrix --lanes-tsv lanes.tsv --out-dir out\n";
  os << "  circt-mut generate --design design.v --out mutations.txt\n";
}

static void printGenerateHelp(raw_ostream &os) {
  os << "usage: circt-mut generate [options]\n\n";
  os << "Required:\n";
  os << "  --design FILE             Input design (.il/.v/.sv)\n";
  os << "  --out FILE                Output mutation list file\n\n";
  os << "Optional:\n";
  os << "  --top NAME                Top module name (recommended for .v/.sv)\n";
  os << "  --count N                 Number of mutations to generate (default: 1000)\n";
  os << "  --seed N                  Random seed for mutate (default: 1)\n";
  os << "  --yosys PATH              Yosys executable (default: yosys)\n";
  os << "  --mode NAME               Mutate mode (repeatable)\n";
  os << "                            Families: arith,control,balanced,all,\n";
  os << "                                      stuck,invert,connect\n";
  os << "  --modes CSV               Comma-separated mutate modes\n";
  os << "  --mode-count NAME=COUNT   Explicit mutation count for a mode (repeatable)\n";
  os << "  --mode-counts CSV         Comma-separated NAME=COUNT mode allocations\n";
  os << "  --mode-weight NAME=WEIGHT Relative weight for a mode (repeatable)\n";
  os << "  --mode-weights CSV        Comma-separated NAME=WEIGHT mode weights\n";
  os << "  --profile NAME            Named mutation profile (repeatable)\n";
  os << "  --profiles CSV            Comma-separated named mutation profiles\n";
  os << "  --cfg KEY=VALUE           Mutate config entry (repeatable)\n";
  os << "  --cfgs CSV                Comma-separated KEY=VALUE config entries\n";
  os << "  --select EXPR             Additional mutate select expression (repeatable)\n";
  os << "  --selects CSV             Comma-separated mutate select expressions\n";
  os << "  --cache-dir DIR           Optional cache directory for generated mutation lists\n";
  os << "  -h, --help                Show help\n\n";
  os << "Output format:\n";
  os << "  Each line in --out is \"<id> <mutation-spec>\" (MCY-compatible).\n";
}

static void printInitHelp(raw_ostream &os) {
  os << "usage: circt-mut init [options]\n\n";
  os << "Options:\n";
  os << "  --project-dir DIR        Project root directory (default: .)\n";
  os << "  --design FILE            Cover/matrix design path (default: design.il)\n";
  os << "  --mutations-file FILE    Mutation list path (default: mutations.txt)\n";
  os << "  --tests-manifest FILE    Tests manifest path (default: tests.tsv)\n";
  os << "  --lanes-tsv FILE         Matrix lane manifest path (default: lanes.tsv)\n";
  os << "  --cover-work-dir DIR     Cover output root (default: out/cover)\n";
  os << "  --matrix-out-dir DIR     Matrix output root (default: out/matrix)\n";
  os << "  --force                  Overwrite generated files if present\n";
  os << "  -h, --help               Show help\n\n";
  os << "Generated files:\n";
  os << "  <project-dir>/circt-mut.toml\n";
  os << "  <project-dir>/<tests-manifest>\n";
  os << "  <project-dir>/<lanes-tsv>\n";
}

static void printRunHelp(raw_ostream &os) {
  os << "usage: circt-mut run [options]\n\n";
  os << "Options:\n";
  os << "  --project-dir DIR        Project root directory (default: .)\n";
  os << "  --config FILE            Config file path (default: <project-dir>/circt-mut.toml)\n";
  os << "  --mode MODE              cover|matrix|all (default: all)\n";
  os << "  -h, --help               Show help\n";
}

static void printReportHelp(raw_ostream &os) {
  os << "usage: circt-mut report [options]\n\n";
  os << "Options:\n";
  os << "  --project-dir DIR        Project root directory (default: .)\n";
  os << "  --config FILE            Config file path (default: <project-dir>/circt-mut.toml)\n";
  os << "  --mode MODE              cover|matrix|all (default: all)\n";
  os << "  --cover-work-dir DIR     Override cover work directory\n";
  os << "  --matrix-out-dir DIR     Override matrix output directory\n";
  os << "  --compare FILE           Compare against baseline report TSV\n";
  os << "  --compare-history-latest FILE\n";
  os << "                           Compare against latest snapshot in history TSV\n";
  os << "  --trend-history FILE     Compute trend summary from history TSV\n";
  os << "  --trend-window N         Use latest N history runs for trends (0=all)\n";
  os << "  --policy-profile NAME    Apply built-in report policy profile\n";
  os << "                           formal-regression-basic|formal-regression-trend\n";
  os << "  --append-history FILE    Append current report rows to history TSV\n";
  os << "  --fail-if-delta-gt RULE  Fail if numeric delta exceeds threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
  os << "  --fail-if-delta-lt RULE  Fail if numeric delta is below threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
  os << "  --fail-if-trend-delta-gt RULE\n";
  os << "                           Fail if trend delta exceeds threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
  os << "  --fail-if-trend-delta-lt RULE\n";
  os << "                           Fail if trend delta is below threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
  os << "  --out FILE               Write report TSV to FILE (also prints to stdout)\n";
  os << "  -h, --help               Show help\n";
}

struct DeltaGateRule {
  std::string key;
  double threshold = 0.0;
};

static std::optional<StringRef> mapSubcommandToScript(StringRef subcommand) {
  if (subcommand == "cover")
    return StringRef("run_mutation_cover.sh");
  if (subcommand == "matrix")
    return StringRef("run_mutation_matrix.sh");
  if (subcommand == "generate")
    return StringRef("generate_mutations_yosys.sh");
  return std::nullopt;
}

static std::optional<std::string> getEnvVar(StringRef key) {
  if (const char *v = std::getenv(key.str().c_str()))
    return std::string(v);
  return std::nullopt;
}

static bool existsAndExecutable(StringRef path) {
  return sys::fs::exists(path) && sys::fs::can_execute(path);
}

static std::optional<std::string> resolveScriptPath(const char *argv0,
                                                    StringRef scriptName) {
  SmallVector<std::string, 8> candidateDirs;

  if (auto envScriptsDir = getEnvVar("CIRCT_MUT_SCRIPTS_DIR");
      envScriptsDir && !envScriptsDir->empty())
    candidateDirs.push_back(*envScriptsDir);

  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (!mainExec.empty()) {
    SmallString<256> exeDir(mainExec);
    sys::path::remove_filename(exeDir);

    // Typical build tree: <repo>/build/bin/circt-mut
    SmallString<256> fromBuild(exeDir);
    sys::path::append(fromBuild, "..", "..", "utils");
    sys::path::remove_dots(fromBuild, true);
    candidateDirs.push_back(std::string(fromBuild.str()));

    // Typical install tree: <prefix>/bin/circt-mut and scripts in
    // <prefix>/share/circt/utils.
    SmallString<256> fromInstall(exeDir);
    sys::path::append(fromInstall, "..", "share", "circt", "utils");
    sys::path::remove_dots(fromInstall, true);
    candidateDirs.push_back(std::string(fromInstall.str()));
  }

  // Fallback for in-tree invocation from repository root.
  candidateDirs.push_back("utils");

  for (const auto &dir : candidateDirs) {
    SmallString<256> candidate(dir);
    sys::path::append(candidate, scriptName);
    if (existsAndExecutable(candidate))
      return std::string(candidate.str());
  }

  return std::nullopt;
}

static int dispatchToScript(StringRef scriptPath,
                            ArrayRef<StringRef> forwardedArgs) {
  SmallVector<StringRef, 16> args;
  args.push_back(scriptPath);
  args.append(forwardedArgs.begin(), forwardedArgs.end());

  std::string errMsg;
  int rc =
      sys::ExecuteAndWait(scriptPath, args, /*Env=*/std::nullopt,
                          /*Redirects=*/{},
                          /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (!errMsg.empty())
    errs() << "circt-mut: execution error: " << errMsg << "\n";
  return rc;
}

static std::optional<std::string> resolveToolPath(StringRef tool) {
  if (tool.contains('/')) {
    if (existsAndExecutable(tool))
      return std::string(tool);
    return std::nullopt;
  }
  if (ErrorOr<std::string> found = sys::findProgramByName(tool))
    return *found;
  return std::nullopt;
}

static std::optional<std::string> resolveToolPathFromEnvPath(StringRef tool) {
  if (tool.contains('/')) {
    if (existsAndExecutable(tool))
      return std::string(tool);
    return std::nullopt;
  }
  return sys::Process::FindInEnvPath("PATH", tool);
}

static std::optional<std::string>
resolveCirctToolPathForWorkflow(const char *argv0, StringRef requested,
                                StringRef toolName, StringRef scriptName) {
  if (requested != "auto")
    return resolveToolPath(requested);

  std::optional<std::string> scriptPath = resolveScriptPath(argv0, scriptName);
  std::string installCandidate;
  std::string repoCandidate;
  if (scriptPath) {
    SmallString<256> scriptDir(*scriptPath);
    sys::path::remove_filename(scriptDir);

    SmallString<256> repoBuild(scriptDir);
    sys::path::append(repoBuild, "..", "build", "bin", toolName);
    sys::path::remove_dots(repoBuild, true);
    repoCandidate = std::string(repoBuild.str());

    StringRef dir = scriptDir;
    StringRef dirBase = sys::path::filename(dir);
    SmallString<256> parent(dir);
    sys::path::remove_filename(parent);
    StringRef parentBase = sys::path::filename(parent);
    SmallString<256> grandParent(parent);
    sys::path::remove_filename(grandParent);
    StringRef grandParentBase = sys::path::filename(grandParent);
    if (dirBase == "utils" && parentBase == "circt" && grandParentBase == "share") {
      SmallString<256> installBin(scriptDir);
      sys::path::append(installBin, "..", "..", "..", "bin");
      sys::path::append(installBin, toolName);
      sys::path::remove_dots(installBin, true);
      installCandidate = std::string(installBin.str());
    }
  }

  if (!installCandidate.empty() && existsAndExecutable(installCandidate))
    return installCandidate;
  if (auto found = resolveToolPath(toolName))
    return found;
  if (!repoCandidate.empty() && existsAndExecutable(repoCandidate))
    return repoCandidate;

  // Fallback: look next to circt-mut.
  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (!mainExec.empty()) {
    SmallString<256> exeDir(mainExec);
    sys::path::remove_filename(exeDir);
    SmallString<256> sibling(exeDir);
    sys::path::append(sibling, toolName);
    if (existsAndExecutable(sibling))
      return std::string(sibling.str());
  }
  return std::nullopt;
}

static bool hasNonZeroDecimalValue(StringRef value) {
  if (value.empty())
    return false;
  return value.find_first_not_of('0') != StringRef::npos;
}

static bool parsePositiveUIntPreflight(StringRef value, uint64_t &out) {
  if (value.getAsInteger(10, out))
    return false;
  return out > 0;
}

enum class AllocationParseErrorKind {
  None,
  InvalidEntry,
  InvalidValue,
};

struct AllocationParseResult {
  bool enabled = false;
  uint64_t total = 0;
  AllocationParseErrorKind errorKind = AllocationParseErrorKind::None;
  std::string entry;
  std::string modeName;
  std::string value;
};

static AllocationParseResult parseModeAllocationCSV(StringRef csv) {
  AllocationParseResult result;
  if (csv.empty())
    return result;
  SmallVector<StringRef, 16> entries;
  csv.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef rawEntry : entries) {
    StringRef entry = rawEntry.trim();
    if (entry.empty())
      continue;
    auto split = entry.split('=');
    StringRef modeName = split.first.trim();
    StringRef valueRef = split.second.trim();
    if (modeName.empty() || valueRef == split.first) {
      result.errorKind = AllocationParseErrorKind::InvalidEntry;
      result.entry = entry.str();
      return result;
    }
    uint64_t parsed = 0;
    if (!parsePositiveUIntPreflight(valueRef, parsed)) {
      result.errorKind = AllocationParseErrorKind::InvalidValue;
      result.modeName = modeName.str();
      result.value = valueRef.str();
      return result;
    }
    result.total += parsed;
    result.enabled = true;
  }
  return result;
}

struct CoverRewriteResult {
  bool ok = false;
  std::string error;
  SmallVector<std::string, 32> rewrittenArgs;
};

static CoverRewriteResult rewriteCoverArgs(const char *argv0,
                                           ArrayRef<StringRef> args) {
  CoverRewriteResult result;
  bool hasMutationsFile = false;
  bool hasGenerateMutations = false;
  std::string generateMutations;
  std::string mutationsModeCounts;
  std::string mutationsModeWeights;
  std::string globalFilterTimeoutSeconds;
  std::string globalFilterLECTimeoutSeconds;
  std::string globalFilterBMCTimeoutSeconds;
  std::string globalFilterBMCBound;
  std::string globalFilterBMCIgnoreAssertsUntil;
  std::string bmcOrigCacheMaxEntries;
  std::string bmcOrigCacheMaxBytes;
  std::string bmcOrigCacheMaxAgeSeconds;
  std::string bmcOrigCacheEvictionPolicy;
  bool hasGlobalFilterCmd = false;
  bool hasGlobalFilterLEC = false;
  bool hasGlobalFilterBMC = false;
  bool hasGlobalFilterChain = false;
  std::string globalFilterChainMode;
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    auto valueFromArg = [&]() -> StringRef {
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos)
        return arg.substr(eqPos + 1);
      if (i + 1 < args.size())
        return args[i + 1];
      return StringRef();
    };

    auto resolveWithOptionalValue =
        [&](StringRef flag, StringRef toolName) -> bool {
      std::string requested = "auto";
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        requested = arg.substr(eqPos + 1).str();
        if (requested.empty())
          requested = "auto";
      } else if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
        requested = args[++i].str();
      }
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, requested, toolName, "run_mutation_cover.sh");
      if (!resolved) {
        result.error =
            (Twine("circt-mut cover: unable to resolve ") + flag +
             " executable: " + requested +
             " (searched repo build/bin, install-tree sibling bin, and PATH).")
                .str();
        return false;
      }
      result.rewrittenArgs.push_back(flag.str());
      result.rewrittenArgs.push_back(*resolved);
      return true;
    };
    auto resolveWithRequiredValue = [&](StringRef flag) -> bool {
      std::string requested;
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        requested = arg.substr(eqPos + 1).str();
      } else if (i + 1 < args.size()) {
        requested = args[++i].str();
      }
      if (requested.empty()) {
        result.error = (Twine("circt-mut cover: missing value for ") + flag).str();
        return false;
      }
      auto resolved = resolveToolPath(requested);
      if (!resolved) {
        result.error = (Twine("circt-mut cover: unable to resolve ") + flag +
                        " executable: " + requested)
                           .str();
        return false;
      }
      result.rewrittenArgs.push_back(flag.str());
      result.rewrittenArgs.push_back(*resolved);
      return true;
    };

    if (arg == "--formal-global-propagate-circt-lec" ||
        arg.starts_with("--formal-global-propagate-circt-lec=")) {
      hasGlobalFilterLEC = true;
      if (!resolveWithOptionalValue("--formal-global-propagate-circt-lec",
                                    "circt-lec"))
        return result;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-bmc" ||
        arg.starts_with("--formal-global-propagate-circt-bmc=")) {
      hasGlobalFilterBMC = true;
      if (!resolveWithOptionalValue("--formal-global-propagate-circt-bmc",
                                    "circt-bmc"))
        return result;
      continue;
    }
    if (arg == "--formal-global-propagate-z3" ||
        arg.starts_with("--formal-global-propagate-z3=")) {
      if (!resolveWithRequiredValue("--formal-global-propagate-z3"))
        return result;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-z3" ||
        arg.starts_with("--formal-global-propagate-bmc-z3=")) {
      if (!resolveWithRequiredValue("--formal-global-propagate-bmc-z3"))
        return result;
      continue;
    }
    if (arg == "--mutations-yosys" || arg.starts_with("--mutations-yosys=")) {
      if (!resolveWithRequiredValue("--mutations-yosys"))
        return result;
      continue;
    }
    if (arg == "--mutations-file" || arg.starts_with("--mutations-file="))
      hasMutationsFile = true;
    if (arg == "--generate-mutations" ||
        arg.starts_with("--generate-mutations=")) {
      hasGenerateMutations = true;
      generateMutations = valueFromArg().str();
    }
    if (arg == "--mutations-mode-counts" ||
        arg.starts_with("--mutations-mode-counts="))
      mutationsModeCounts = valueFromArg().str();
    if (arg == "--mutations-mode-weights" ||
        arg.starts_with("--mutations-mode-weights="))
      mutationsModeWeights = valueFromArg().str();
    if (arg == "--formal-global-propagate-cmd" ||
        arg.starts_with("--formal-global-propagate-cmd="))
      hasGlobalFilterCmd = true;
    if (arg == "--formal-global-propagate-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-timeout-seconds=")) {
      globalFilterTimeoutSeconds = valueFromArg().str();
    }
    if (arg == "--formal-global-propagate-lec-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-lec-timeout-seconds=")) {
      globalFilterLECTimeoutSeconds = valueFromArg().str();
    }
    if (arg == "--formal-global-propagate-bmc-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-bmc-timeout-seconds=")) {
      globalFilterBMCTimeoutSeconds = valueFromArg().str();
    }
    if (arg == "--formal-global-propagate-bmc-bound" ||
        arg.starts_with("--formal-global-propagate-bmc-bound=")) {
      globalFilterBMCBound = valueFromArg().str();
    }
    if (arg == "--formal-global-propagate-bmc-ignore-asserts-until" ||
        arg.starts_with("--formal-global-propagate-bmc-ignore-asserts-until=")) {
      globalFilterBMCIgnoreAssertsUntil = valueFromArg().str();
    }
    if (arg == "--bmc-orig-cache-max-entries" ||
        arg.starts_with("--bmc-orig-cache-max-entries=")) {
      bmcOrigCacheMaxEntries = valueFromArg().str();
    }
    if (arg == "--bmc-orig-cache-max-bytes" ||
        arg.starts_with("--bmc-orig-cache-max-bytes=")) {
      bmcOrigCacheMaxBytes = valueFromArg().str();
    }
    if (arg == "--bmc-orig-cache-max-age-seconds" ||
        arg.starts_with("--bmc-orig-cache-max-age-seconds=")) {
      bmcOrigCacheMaxAgeSeconds = valueFromArg().str();
    }
    if (arg == "--bmc-orig-cache-eviction-policy" ||
        arg.starts_with("--bmc-orig-cache-eviction-policy=")) {
      bmcOrigCacheEvictionPolicy = valueFromArg().str();
    }
    if (arg == "--formal-global-propagate-circt-chain" ||
        arg.starts_with("--formal-global-propagate-circt-chain=")) {
      constexpr StringRef chainPrefix =
          "--formal-global-propagate-circt-chain=";
      hasGlobalFilterChain = true;
      if (arg.starts_with(chainPrefix))
        globalFilterChainMode = arg.substr(chainPrefix.size()).str();
      else if (i + 1 < args.size())
        globalFilterChainMode = args[i + 1].str();
      else
        globalFilterChainMode.clear();
    }

    result.rewrittenArgs.push_back(arg.str());
  }

  if (hasMutationsFile && hasGenerateMutations) {
    result.error = "circt-mut cover: --mutations-file and --generate-mutations "
                   "are mutually exclusive.";
    return result;
  }
  if (hasGenerateMutations) {
    if (generateMutations.empty()) {
      result.error = "circt-mut cover: missing value for --generate-mutations.";
      return result;
    }
    uint64_t generateCount = 0;
    if (!parsePositiveUIntPreflight(generateMutations, generateCount)) {
      result.error = (Twine("circt-mut cover: invalid --generate-mutations "
                            "value: ") +
                      generateMutations + " (expected positive integer).")
                         .str();
      return result;
    }

    AllocationParseResult modeCounts =
        parseModeAllocationCSV(mutationsModeCounts);
    if (modeCounts.errorKind == AllocationParseErrorKind::InvalidEntry) {
      result.error =
          (Twine("circt-mut cover: invalid --mutations-mode-counts entry: ") +
           modeCounts.entry + " (expected NAME=COUNT).")
              .str();
      return result;
    }
    if (modeCounts.errorKind == AllocationParseErrorKind::InvalidValue) {
      result.error = (Twine("circt-mut cover: invalid --mutations-mode-count "
                            "value for ") +
                      modeCounts.modeName + ": " + modeCounts.value +
                      " (expected positive integer).")
                         .str();
      return result;
    }

    AllocationParseResult modeWeights =
        parseModeAllocationCSV(mutationsModeWeights);
    if (modeWeights.errorKind == AllocationParseErrorKind::InvalidEntry) {
      result.error =
          (Twine("circt-mut cover: invalid --mutations-mode-weights entry: ") +
           modeWeights.entry + " (expected NAME=WEIGHT).")
              .str();
      return result;
    }
    if (modeWeights.errorKind == AllocationParseErrorKind::InvalidValue) {
      result.error = (Twine("circt-mut cover: invalid --mutations-mode-weight "
                            "value for ") +
                      modeWeights.modeName + ": " + modeWeights.value +
                      " (expected positive integer).")
                         .str();
      return result;
    }

    if (modeCounts.enabled && modeWeights.enabled) {
      result.error = "circt-mut cover: use either --mutations-mode-counts or "
                     "--mutations-mode-weights, not both.";
      return result;
    }
    if (modeCounts.enabled && modeCounts.total != generateCount) {
      result.error =
          (Twine("circt-mut cover: --mutations-mode-counts total (") +
           Twine(modeCounts.total) + ") must match --generate-mutations (" +
           Twine(generateCount) + ").")
              .str();
      return result;
    }
  }

  if (hasGlobalFilterChain) {
    if (globalFilterChainMode != "lec-then-bmc" &&
        globalFilterChainMode != "bmc-then-lec" &&
        globalFilterChainMode != "consensus" &&
        globalFilterChainMode != "auto") {
      result.error =
          (Twine("Invalid --formal-global-propagate-circt-chain value: ") +
           globalFilterChainMode +
           " (expected lec-then-bmc|bmc-then-lec|consensus|auto).")
              .str();
      return result;
    }
    if (hasGlobalFilterCmd) {
      result.error = "--formal-global-propagate-circt-chain cannot be combined "
                     "with --formal-global-propagate-cmd.";
      return result;
    }
    if (!hasGlobalFilterLEC) {
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, "auto", "circt-lec", "run_mutation_cover.sh");
      if (!resolved) {
        result.error =
            "circt-mut cover: unable to resolve --formal-global-propagate-circt-lec executable: auto (searched repo build/bin, install-tree sibling bin, and PATH).";
        return result;
      }
      result.rewrittenArgs.push_back("--formal-global-propagate-circt-lec");
      result.rewrittenArgs.push_back(*resolved);
      hasGlobalFilterLEC = true;
    }
    if (!hasGlobalFilterBMC) {
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, "auto", "circt-bmc", "run_mutation_cover.sh");
      if (!resolved) {
        result.error =
            "circt-mut cover: unable to resolve --formal-global-propagate-circt-bmc executable: auto (searched repo build/bin, install-tree sibling bin, and PATH).";
        return result;
      }
      result.rewrittenArgs.push_back("--formal-global-propagate-circt-bmc");
      result.rewrittenArgs.push_back(*resolved);
      hasGlobalFilterBMC = true;
    }
  } else {
    int modeCount = 0;
    modeCount += hasGlobalFilterCmd ? 1 : 0;
    modeCount += hasGlobalFilterLEC ? 1 : 0;
    modeCount += hasGlobalFilterBMC ? 1 : 0;
    if (modeCount > 1) {
      result.error =
          "Use only one global filter mode: --formal-global-propagate-cmd, --formal-global-propagate-circt-lec, --formal-global-propagate-circt-bmc, or --formal-global-propagate-circt-chain.";
      return result;
    }
  }
  auto validateCoverRegex = [&](StringRef value, const Regex &pattern,
                                StringRef flag, StringRef expected) -> bool {
    if (value.empty())
      return true;
    if (pattern.match(value))
      return true;
    result.error =
        (Twine("Invalid ") + flag + " value: " + value + " (expected " +
         expected + ").")
            .str();
    return false;
  };
  if (!validateCoverRegex(globalFilterTimeoutSeconds, Regex("^[0-9]+$"),
                          "--formal-global-propagate-timeout-seconds",
                          "0-9 integer"))
    return result;
  if (!validateCoverRegex(globalFilterLECTimeoutSeconds, Regex("^[0-9]+$"),
                          "--formal-global-propagate-lec-timeout-seconds",
                          "0-9 integer"))
    return result;
  if (!validateCoverRegex(globalFilterBMCTimeoutSeconds, Regex("^[0-9]+$"),
                          "--formal-global-propagate-bmc-timeout-seconds",
                          "0-9 integer"))
    return result;
  if (!validateCoverRegex(globalFilterBMCBound, Regex("^[1-9][0-9]*$"),
                          "--formal-global-propagate-bmc-bound",
                          "positive integer"))
    return result;
  if (!validateCoverRegex(globalFilterBMCIgnoreAssertsUntil, Regex("^[0-9]+$"),
                          "--formal-global-propagate-bmc-ignore-asserts-until",
                          "0-9 integer"))
    return result;
  if (!validateCoverRegex(bmcOrigCacheMaxEntries, Regex("^[0-9]+$"),
                          "--bmc-orig-cache-max-entries", "0-9 integer"))
    return result;
  if (!validateCoverRegex(bmcOrigCacheMaxBytes, Regex("^[0-9]+$"),
                          "--bmc-orig-cache-max-bytes", "0-9 integer"))
    return result;
  if (!validateCoverRegex(bmcOrigCacheMaxAgeSeconds, Regex("^[0-9]+$"),
                          "--bmc-orig-cache-max-age-seconds", "0-9 integer"))
    return result;
  if (!validateCoverRegex(bmcOrigCacheEvictionPolicy,
                          Regex("^(lru|fifo|cost-lru)$"),
                          "--bmc-orig-cache-eviction-policy",
                          "lru|fifo|cost-lru"))
    return result;
  bool hasAnyGlobalFilterMode =
      hasGlobalFilterCmd || hasGlobalFilterLEC || hasGlobalFilterBMC ||
      hasGlobalFilterChain;
  bool needsTimeoutTool =
      hasAnyGlobalFilterMode &&
      (hasNonZeroDecimalValue(globalFilterTimeoutSeconds) ||
       hasNonZeroDecimalValue(globalFilterLECTimeoutSeconds) ||
       hasNonZeroDecimalValue(globalFilterBMCTimeoutSeconds));
  if (needsTimeoutTool && !resolveToolPathFromEnvPath("timeout")) {
    result.error = "circt-mut cover: unable to resolve timeout executable "
                   "required by --formal-global-propagate-*-timeout-seconds; "
                   "install coreutils timeout or set PATH.";
    return result;
  }

  result.ok = true;
  return result;
}

struct MatrixRewriteResult {
  bool ok = false;
  std::string error;
  SmallVector<std::string, 32> rewrittenArgs;
};

struct MatrixLanePreflightDefaults {
  std::string mutationsYosys;
  std::string mutationsModeCounts;
  std::string mutationsModeWeights;
  std::string globalFilterCmd;
  std::string globalFilterTimeoutSeconds;
  std::string globalFilterLECTimeoutSeconds;
  std::string globalFilterBMCTimeoutSeconds;
  std::string globalFilterCirctLEC;
  std::string globalFilterCirctBMC;
  std::string globalFilterCirctChain;
  std::string globalFilterZ3;
  std::string globalFilterBMCZ3;
  std::string globalFilterBMCBound;
  std::string globalFilterBMCIgnoreAssertsUntil;
  bool globalFilterAssumeKnownInputs = false;
  bool globalFilterAcceptXpropOnly = false;
  bool globalFilterBMCRunSMTLib = false;
  bool globalFilterBMCAssumeKnownInputs = false;
  std::string bmcOrigCacheMaxEntries;
  std::string bmcOrigCacheMaxBytes;
  std::string bmcOrigCacheMaxAgeSeconds;
  std::string bmcOrigCacheEvictionPolicy;
  bool skipBaseline = false;
  bool failOnUndetected = false;
  bool failOnErrors = false;
};

static bool preflightMatrixLaneTools(
    const char *argv0, StringRef lanesTSVPath,
    const MatrixLanePreflightDefaults &defaults, std::string &error,
    bool &needsTimeoutTool) {
  auto bufferOrErr = MemoryBuffer::getFile(lanesTSVPath);
  if (!bufferOrErr) {
    error = (Twine("circt-mut matrix: unable to read --lanes-tsv: ") +
             lanesTSVPath)
                .str();
    return false;
  }

  SmallVector<StringRef, 128> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    StringRef rawLine = lines[lineIdx];
    StringRef trimmed = rawLine.trim();
    if (trimmed.empty() || trimmed.starts_with("#"))
      continue;

    SmallVector<StringRef, 64> cols;
    rawLine.split(cols, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    auto getCol = [&](size_t idx) -> StringRef {
      if (idx >= cols.size())
        return StringRef();
      return cols[idx].trim();
    };
    auto normalized = [&](StringRef value) -> StringRef {
      return value.empty() || value == "-" ? StringRef() : value;
    };
    auto withDefault = [&](StringRef laneValue,
                           StringRef defaultValue) -> StringRef {
      StringRef lane = normalized(laneValue);
      if (!lane.empty())
        return lane;
      return normalized(defaultValue);
    };

    StringRef laneID = getCol(0);
    StringRef mutationsFile = getCol(2);
    StringRef generateCount = getCol(7);
    StringRef laneMutationsYosys = getCol(10);
    StringRef laneMutationsModeCounts = getCol(33);
    StringRef laneGlobalFilterCmd = getCol(14);
    StringRef laneGlobalFilterLEC = getCol(15);
    StringRef laneGlobalFilterBMC = getCol(16);
    StringRef laneGlobalFilterBMCBound = getCol(18);
    StringRef laneGlobalFilterBMCRunSMTLib = getCol(20);
    StringRef laneGlobalFilterBMCZ3 = getCol(21);
    StringRef laneGlobalFilterBMCAssumeKnownInputs = getCol(22);
    StringRef laneGlobalFilterBMCIgnoreAssertsUntil = getCol(23);
    StringRef laneGlobalFilterZ3 = getCol(27);
    StringRef laneGlobalFilterAssumeKnownInputs = getCol(28);
    StringRef laneGlobalFilterAcceptXpropOnly = getCol(29);
    StringRef laneGlobalFilterChain = getCol(34);
    StringRef laneBMCOrigCacheMaxEntries = getCol(35);
    StringRef laneBMCOrigCacheMaxBytes = getCol(36);
    StringRef laneBMCOrigCacheMaxAgeSeconds = getCol(37);
    StringRef laneBMCOrigCacheEvictionPolicy = getCol(38);
    StringRef laneSkipBaseline = getCol(39);
    StringRef laneFailOnUndetected = getCol(40);
    StringRef laneFailOnErrors = getCol(41);
    StringRef laneGlobalFilterTimeoutSeconds = getCol(42);
    StringRef laneGlobalFilterLECTimeoutSeconds = getCol(43);
    StringRef laneGlobalFilterBMCTimeoutSeconds = getCol(44);
    StringRef laneMutationsModeWeights = getCol(45);
    std::string laneLabel = laneID.empty() ? "<unknown>" : laneID.str();

    auto resolveLaneTool = [&](StringRef laneValue, StringRef defaultValue,
                               StringRef flag, StringRef toolName,
                               bool allowAuto) -> bool {
      StringRef requested = withDefault(laneValue, defaultValue);
      if (requested.empty())
        return true;
      std::optional<std::string> resolved;
      if (allowAuto)
        resolved = resolveCirctToolPathForWorkflow(argv0, requested, toolName,
                                                   "run_mutation_matrix.sh");
      else
        resolved = resolveToolPath(requested);
      if (resolved)
        return true;
      error =
          (Twine("circt-mut matrix: unable to resolve lane ") + flag +
           " executable in --lanes-tsv at line " +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + requested)
              .str();
      return false;
    };
    auto validateRegex = [&](StringRef laneValue, StringRef defaultValue,
                             const Regex &pattern, StringRef fieldName,
                             StringRef expected) -> bool {
      StringRef value = withDefault(laneValue, defaultValue);
      if (value.empty())
        return true;
      if (pattern.match(value))
        return true;
      error = (Twine("Invalid lane ") + fieldName + " value in --lanes-tsv at "
               "line " + Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + value + " (expected " +
               expected + ").")
                  .str();
      return false;
    };
    auto parseGateOverride = [&](StringRef laneValue,
                                 StringRef fieldName) -> bool {
      StringRef value = normalized(laneValue);
      if (value.empty())
        return true;
      if (value == "1" || value == "0" || value == "true" || value == "false" ||
          value == "yes" || value == "no")
        return true;
      error =
          (Twine("Invalid lane ") + fieldName + " override in --lanes-tsv at "
           "line " + Twine(static_cast<unsigned long long>(lineIdx + 1)) +
           " (lane " + laneLabel + "): " + value +
           " (expected 1|0|true|false|yes|no|-).")
              .str();
      return false;
    };
    auto validateBoolOverride = [&](StringRef laneValue,
                                    StringRef fieldName) -> bool {
      StringRef value = normalized(laneValue);
      if (value.empty())
        return true;
      if (value == "1" || value == "0" || value == "true" ||
          value == "false" || value == "yes" || value == "no")
        return true;
      error = (Twine("Invalid lane ") + fieldName + " override in --lanes-tsv "
               "at line " +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + value +
               " (expected 1|0|true|false|yes|no|-).")
                  .str();
      return false;
    };

    StringRef effectiveCmd =
        withDefault(laneGlobalFilterCmd, defaults.globalFilterCmd);
    StringRef effectiveLEC =
        withDefault(laneGlobalFilterLEC, defaults.globalFilterCirctLEC);
    StringRef effectiveBMC =
        withDefault(laneGlobalFilterBMC, defaults.globalFilterCirctBMC);
    StringRef effectiveChain =
        withDefault(laneGlobalFilterChain, defaults.globalFilterCirctChain);
    StringRef effectiveTimeoutSeconds = withDefault(
        laneGlobalFilterTimeoutSeconds, defaults.globalFilterTimeoutSeconds);
    StringRef effectiveLECTimeoutSeconds =
        withDefault(laneGlobalFilterLECTimeoutSeconds,
                    defaults.globalFilterLECTimeoutSeconds);
    StringRef effectiveBMCTimeoutSeconds =
        withDefault(laneGlobalFilterBMCTimeoutSeconds,
                    defaults.globalFilterBMCTimeoutSeconds);

    if (!effectiveChain.empty()) {
      if (effectiveChain != "lec-then-bmc" && effectiveChain != "bmc-then-lec" &&
          effectiveChain != "consensus" && effectiveChain != "auto") {
        error =
            (Twine("Invalid lane global_propagate_circt_chain value in "
                   "--lanes-tsv at line ") +
             Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
             laneLabel + "): " + effectiveChain +
             " (expected lec-then-bmc|bmc-then-lec|consensus|auto).")
                .str();
        return false;
      }
      if (!effectiveCmd.empty()) {
        error =
            (Twine("Lane global filter config conflict in --lanes-tsv at line ") +
             Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
             laneLabel +
             "): --formal-global-propagate-circt-chain cannot be combined "
             "with --formal-global-propagate-cmd.")
                .str();
        return false;
      }
      if (effectiveLEC.empty() || effectiveBMC.empty()) {
        error =
            (Twine("Lane global filter config error in --lanes-tsv at line ") +
             Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
             laneLabel +
             "): --formal-global-propagate-circt-chain requires both "
             "--formal-global-propagate-circt-lec and "
             "--formal-global-propagate-circt-bmc (lane or default).")
                .str();
        return false;
      }
    } else {
      int modeCount = 0;
      modeCount += effectiveCmd.empty() ? 0 : 1;
      modeCount += effectiveLEC.empty() ? 0 : 1;
      modeCount += effectiveBMC.empty() ? 0 : 1;
      if (modeCount > 1) {
        error =
            (Twine("Lane global filter config conflict in --lanes-tsv at line ") +
             Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
             laneLabel +
             "): use only one global filter mode "
             "(cmd|circt-lec|circt-bmc) unless chain mode is set.")
                .str();
        return false;
      }
    }

    if (!resolveLaneTool(laneGlobalFilterLEC, defaults.globalFilterCirctLEC,
                         "global_propagate_circt_lec", "circt-lec",
                         /*allowAuto=*/true))
      return false;
    if (!resolveLaneTool(laneGlobalFilterBMC, defaults.globalFilterCirctBMC,
                         "global_propagate_circt_bmc", "circt-bmc",
                         /*allowAuto=*/true))
      return false;

    bool useLECFilter = !effectiveLEC.empty();
    bool useBMCFilter = !effectiveBMC.empty();
    if (useLECFilter &&
        !resolveLaneTool(laneGlobalFilterZ3, defaults.globalFilterZ3,
                         "global_propagate_z3", "", /*allowAuto=*/false))
      return false;
    if (useBMCFilter &&
        !resolveLaneTool(laneGlobalFilterBMCZ3, defaults.globalFilterBMCZ3,
                         "global_propagate_bmc_z3", "", /*allowAuto=*/false))
      return false;
    if (!validateRegex(laneGlobalFilterTimeoutSeconds,
                       defaults.globalFilterTimeoutSeconds, Regex("^[0-9]+$"),
                       "global_propagate_timeout_seconds", "0-9 integer"))
      return false;
    if (!validateRegex(laneGlobalFilterLECTimeoutSeconds,
                       defaults.globalFilterLECTimeoutSeconds,
                       Regex("^[0-9]+$"), "global_propagate_lec_timeout_seconds",
                       "0-9 integer"))
      return false;
    if (!validateRegex(laneGlobalFilterBMCTimeoutSeconds,
                       defaults.globalFilterBMCTimeoutSeconds,
                       Regex("^[0-9]+$"), "global_propagate_bmc_timeout_seconds",
                       "0-9 integer"))
      return false;
    if (!validateRegex(laneGlobalFilterBMCBound, defaults.globalFilterBMCBound,
                       Regex("^[1-9][0-9]*$"), "global_propagate_bmc_bound",
                       "positive integer"))
      return false;
    if (!validateRegex(laneGlobalFilterBMCIgnoreAssertsUntil,
                       defaults.globalFilterBMCIgnoreAssertsUntil,
                       Regex("^[0-9]+$"),
                       "global_propagate_bmc_ignore_asserts_until",
                       "0-9 integer"))
      return false;
    if (!validateRegex(laneBMCOrigCacheMaxEntries, defaults.bmcOrigCacheMaxEntries,
                       Regex("^[0-9]+$"), "bmc_orig_cache_max_entries",
                       "0-9 integer"))
      return false;
    if (!validateRegex(laneBMCOrigCacheMaxBytes, defaults.bmcOrigCacheMaxBytes,
                       Regex("^[0-9]+$"), "bmc_orig_cache_max_bytes",
                       "0-9 integer"))
      return false;
    if (!validateRegex(laneBMCOrigCacheMaxAgeSeconds,
                       defaults.bmcOrigCacheMaxAgeSeconds, Regex("^[0-9]+$"),
                       "bmc_orig_cache_max_age_seconds", "0-9 integer"))
      return false;
    if (!validateRegex(laneBMCOrigCacheEvictionPolicy,
                       defaults.bmcOrigCacheEvictionPolicy,
                       Regex("^(lru|fifo|cost-lru)$"),
                       "bmc_orig_cache_eviction_policy", "lru|fifo|cost-lru"))
      return false;
    if (!validateBoolOverride(laneGlobalFilterAssumeKnownInputs,
                              "global_propagate_assume_known_inputs"))
      return false;
    if (!validateBoolOverride(laneGlobalFilterAcceptXpropOnly,
                              "global_propagate_accept_xprop_only"))
      return false;
    if (!validateBoolOverride(laneGlobalFilterBMCRunSMTLib,
                              "global_propagate_bmc_run_smtlib"))
      return false;
    if (!validateBoolOverride(laneGlobalFilterBMCAssumeKnownInputs,
                              "global_propagate_bmc_assume_known_inputs"))
      return false;
    if (!parseGateOverride(laneSkipBaseline, "skip_baseline"))
      return false;
    if (!parseGateOverride(laneFailOnUndetected,
                           "fail_on_undetected"))
      return false;
    if (!parseGateOverride(laneFailOnErrors, "fail_on_errors"))
      return false;
    bool hasAnyEffectiveGlobalFilterMode =
        !effectiveCmd.empty() || !effectiveLEC.empty() || !effectiveBMC.empty() ||
        !effectiveChain.empty();
    if (hasAnyEffectiveGlobalFilterMode &&
        (hasNonZeroDecimalValue(effectiveTimeoutSeconds) ||
         hasNonZeroDecimalValue(effectiveLECTimeoutSeconds) ||
         hasNonZeroDecimalValue(effectiveBMCTimeoutSeconds)))
      needsTimeoutTool = true;

    bool autoGenerateLane =
        mutationsFile == "-" && !generateCount.empty() && generateCount != "-";
    if (!autoGenerateLane)
      continue;

    uint64_t laneGenerateCount = 0;
    if (!parsePositiveUIntPreflight(generateCount, laneGenerateCount)) {
      error = (Twine("Invalid lane generate_count value in --lanes-tsv at "
                     "line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + generateCount +
               " (expected positive integer).")
                  .str();
      return false;
    }

    StringRef effectiveModeCounts =
        withDefault(laneMutationsModeCounts, defaults.mutationsModeCounts);
    StringRef effectiveModeWeights =
        withDefault(laneMutationsModeWeights, defaults.mutationsModeWeights);
    AllocationParseResult modeCounts =
        parseModeAllocationCSV(effectiveModeCounts);
    if (modeCounts.errorKind == AllocationParseErrorKind::InvalidEntry) {
      error =
          (Twine("Invalid lane mutations_mode_counts value in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + modeCounts.entry + " (expected NAME=COUNT).")
              .str();
      return false;
    }
    if (modeCounts.errorKind == AllocationParseErrorKind::InvalidValue) {
      error =
          (Twine("Invalid lane mutations_mode_counts value in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + modeCounts.modeName + "=" + modeCounts.value +
           " (expected NAME=COUNT with positive integer COUNT).")
              .str();
      return false;
    }
    AllocationParseResult modeWeights =
        parseModeAllocationCSV(effectiveModeWeights);
    if (modeWeights.errorKind == AllocationParseErrorKind::InvalidEntry) {
      error =
          (Twine("Invalid lane mutations_mode_weights value in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + modeWeights.entry +
           " (expected NAME=WEIGHT).")
              .str();
      return false;
    }
    if (modeWeights.errorKind == AllocationParseErrorKind::InvalidValue) {
      error =
          (Twine("Invalid lane mutations_mode_weights value in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + modeWeights.modeName + "=" +
           modeWeights.value +
           " (expected NAME=WEIGHT with positive integer WEIGHT).")
              .str();
      return false;
    }
    if (modeCounts.enabled && modeWeights.enabled) {
      error =
          (Twine("Lane mutation generation config conflict in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel +
           "): use only one of mutations_mode_counts or mutations_mode_weights.")
              .str();
      return false;
    }
    if (modeCounts.enabled && modeCounts.total != laneGenerateCount) {
      error =
          (Twine("Lane mutation generation config error in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): mutations_mode_counts total (" +
           Twine(modeCounts.total) + ") must match generate_count (" +
           Twine(laneGenerateCount) + ").")
              .str();
      return false;
    }

    StringRef requestedYosys = laneMutationsYosys;
    if (requestedYosys.empty() || requestedYosys == "-") {
      if (!defaults.mutationsYosys.empty())
        requestedYosys = defaults.mutationsYosys;
      else
        requestedYosys = "yosys";
    }

    if (!resolveToolPath(requestedYosys)) {
      error = (Twine("circt-mut matrix: unable to resolve lane mutations_yosys "
                     "executable in --lanes-tsv at line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + requestedYosys)
                  .str();
      return false;
    }
  }
  return true;
}

static MatrixRewriteResult rewriteMatrixArgs(const char *argv0,
                                             ArrayRef<StringRef> args) {
  MatrixRewriteResult result;
  MatrixLanePreflightDefaults defaults;
  bool hasDefaultGlobalFilterCmd = false;
  bool hasDefaultGlobalFilterLEC = false;
  bool hasDefaultGlobalFilterBMC = false;
  bool hasDefaultGlobalFilterChain = false;
  std::string defaultGlobalFilterChainMode;
  std::string lanesTSVPath;
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    auto valueFromArg = [&]() -> StringRef {
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos)
        return arg.substr(eqPos + 1);
      if (i + 1 < args.size())
        return args[i + 1];
      return StringRef();
    };

    auto resolveWithOptionalValue =
        [&](StringRef flag, StringRef toolName,
            std::string *resolvedOut) -> bool {
      std::string requested = "auto";
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        requested = arg.substr(eqPos + 1).str();
        if (requested.empty())
          requested = "auto";
      } else if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
        requested = args[++i].str();
      }
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, requested, toolName, "run_mutation_matrix.sh");
      if (!resolved) {
        result.error =
            (Twine("circt-mut matrix: unable to resolve ") + flag +
             " executable: " + requested +
             " (searched repo build/bin, install-tree sibling bin, and PATH).")
                .str();
        return false;
      }
      if (resolvedOut)
        *resolvedOut = *resolved;
      result.rewrittenArgs.push_back(flag.str());
      result.rewrittenArgs.push_back(*resolved);
      return true;
    };
    auto resolveWithRequiredValue = [&](StringRef flag,
                                        std::string *resolvedOut) -> bool {
      std::string requested;
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        requested = arg.substr(eqPos + 1).str();
      } else if (i + 1 < args.size()) {
        requested = args[++i].str();
      }
      if (requested.empty()) {
        result.error =
            (Twine("circt-mut matrix: missing value for ") + flag).str();
        return false;
      }
      auto resolved = resolveToolPath(requested);
      if (!resolved) {
        result.error = (Twine("circt-mut matrix: unable to resolve ") + flag +
                        " executable: " + requested)
                           .str();
        return false;
      }
      if (resolvedOut)
        *resolvedOut = *resolved;
      result.rewrittenArgs.push_back(flag.str());
      result.rewrittenArgs.push_back(*resolved);
      return true;
    };

    if (arg == "--default-formal-global-propagate-circt-lec" ||
        arg.starts_with("--default-formal-global-propagate-circt-lec=")) {
      hasDefaultGlobalFilterLEC = true;
      if (!resolveWithOptionalValue(
              "--default-formal-global-propagate-circt-lec", "circt-lec",
              &defaults.globalFilterCirctLEC))
        return result;
      continue;
    }
    if (arg == "--default-formal-global-propagate-circt-bmc" ||
        arg.starts_with("--default-formal-global-propagate-circt-bmc=")) {
      hasDefaultGlobalFilterBMC = true;
      if (!resolveWithOptionalValue(
              "--default-formal-global-propagate-circt-bmc", "circt-bmc",
              &defaults.globalFilterCirctBMC))
        return result;
      continue;
    }
    if (arg == "--default-formal-global-propagate-z3" ||
        arg.starts_with("--default-formal-global-propagate-z3=")) {
      if (!resolveWithRequiredValue("--default-formal-global-propagate-z3",
                                    &defaults.globalFilterZ3))
        return result;
      continue;
    }
    if (arg == "--default-formal-global-propagate-bmc-z3" ||
        arg.starts_with("--default-formal-global-propagate-bmc-z3=")) {
      if (!resolveWithRequiredValue("--default-formal-global-propagate-bmc-z3",
                                    &defaults.globalFilterBMCZ3))
        return result;
      continue;
    }
    if (arg == "--default-mutations-yosys" ||
        arg.starts_with("--default-mutations-yosys=")) {
      std::string requested;
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos)
        requested = arg.substr(eqPos + 1).str();
      else if (i + 1 < args.size())
        requested = args[++i].str();
      if (requested.empty()) {
        result.error =
            "circt-mut matrix: missing value for --default-mutations-yosys";
        return result;
      }
      auto resolved = resolveToolPath(requested);
      if (!resolved) {
        result.error =
            (Twine("circt-mut matrix: unable to resolve "
                   "--default-mutations-yosys executable: ") +
             requested)
                .str();
        return result;
      }
      defaults.mutationsYosys = *resolved;
      result.rewrittenArgs.push_back("--default-mutations-yosys");
      result.rewrittenArgs.push_back(*resolved);
      continue;
    }
    if (arg == "--default-mutations-mode-counts" ||
        arg.starts_with("--default-mutations-mode-counts=")) {
      defaults.mutationsModeCounts = valueFromArg().str();
    }
    if (arg == "--default-mutations-mode-weights" ||
        arg.starts_with("--default-mutations-mode-weights=")) {
      defaults.mutationsModeWeights = valueFromArg().str();
    }
    if (arg == "--lanes-tsv") {
      if (i + 1 >= args.size()) {
        result.error = "circt-mut matrix: missing value for --lanes-tsv";
        return result;
      }
      lanesTSVPath = args[i + 1].str();
    } else if (arg.starts_with("--lanes-tsv=")) {
      constexpr StringRef lanesPrefix = "--lanes-tsv=";
      lanesTSVPath = arg.substr(lanesPrefix.size()).str();
    }
    if (arg == "--default-formal-global-propagate-cmd" ||
        arg.starts_with("--default-formal-global-propagate-cmd=")) {
      hasDefaultGlobalFilterCmd = true;
      defaults.globalFilterCmd = valueFromArg().str();
    }
    if (arg == "--default-formal-global-propagate-timeout-seconds" ||
        arg.starts_with("--default-formal-global-propagate-timeout-seconds=")) {
      defaults.globalFilterTimeoutSeconds =
          valueFromArg().str();
    }
    if (arg == "--default-formal-global-propagate-lec-timeout-seconds" ||
        arg.starts_with(
            "--default-formal-global-propagate-lec-timeout-seconds=")) {
      defaults.globalFilterLECTimeoutSeconds =
          valueFromArg().str();
    }
    if (arg == "--default-formal-global-propagate-bmc-timeout-seconds" ||
        arg.starts_with(
            "--default-formal-global-propagate-bmc-timeout-seconds=")) {
      defaults.globalFilterBMCTimeoutSeconds =
          valueFromArg().str();
    }
    if (arg == "--default-formal-global-propagate-circt-chain" ||
        arg.starts_with("--default-formal-global-propagate-circt-chain=")) {
      constexpr StringRef chainPrefix =
          "--default-formal-global-propagate-circt-chain=";
      hasDefaultGlobalFilterChain = true;
      if (arg.starts_with(chainPrefix))
        defaultGlobalFilterChainMode = arg.substr(chainPrefix.size()).str();
      else if (i + 1 < args.size())
        defaultGlobalFilterChainMode = args[i + 1].str();
      else
        defaultGlobalFilterChainMode.clear();
      defaults.globalFilterCirctChain = defaultGlobalFilterChainMode;
    }
    if (arg == "--default-formal-global-propagate-assume-known-inputs")
      defaults.globalFilterAssumeKnownInputs = true;
    if (arg == "--default-formal-global-propagate-accept-xprop-only")
      defaults.globalFilterAcceptXpropOnly = true;
    if (arg == "--default-formal-global-propagate-bmc-run-smtlib")
      defaults.globalFilterBMCRunSMTLib = true;
    if (arg == "--default-formal-global-propagate-bmc-assume-known-inputs")
      defaults.globalFilterBMCAssumeKnownInputs = true;
    if (arg == "--default-formal-global-propagate-bmc-bound" ||
        arg.starts_with("--default-formal-global-propagate-bmc-bound=")) {
      defaults.globalFilterBMCBound =
          valueFromArg().str();
    }
    if (arg == "--default-formal-global-propagate-bmc-ignore-asserts-until" ||
        arg.starts_with(
            "--default-formal-global-propagate-bmc-ignore-asserts-until=")) {
      defaults.globalFilterBMCIgnoreAssertsUntil =
          valueFromArg().str();
    }
    if (arg == "--default-bmc-orig-cache-max-entries" ||
        arg.starts_with("--default-bmc-orig-cache-max-entries=")) {
      defaults.bmcOrigCacheMaxEntries =
          valueFromArg().str();
    }
    if (arg == "--default-bmc-orig-cache-max-bytes" ||
        arg.starts_with("--default-bmc-orig-cache-max-bytes=")) {
      defaults.bmcOrigCacheMaxBytes =
          valueFromArg().str();
    }
    if (arg == "--default-bmc-orig-cache-max-age-seconds" ||
        arg.starts_with("--default-bmc-orig-cache-max-age-seconds=")) {
      defaults.bmcOrigCacheMaxAgeSeconds =
          valueFromArg().str();
    }
    if (arg == "--default-bmc-orig-cache-eviction-policy" ||
        arg.starts_with("--default-bmc-orig-cache-eviction-policy=")) {
      defaults.bmcOrigCacheEvictionPolicy =
          valueFromArg().str();
    }
    if (arg == "--skip-baseline")
      defaults.skipBaseline = true;
    if (arg == "--fail-on-undetected")
      defaults.failOnUndetected = true;
    if (arg == "--fail-on-errors")
      defaults.failOnErrors = true;

    result.rewrittenArgs.push_back(arg.str());
  }

  AllocationParseResult defaultModeCounts =
      parseModeAllocationCSV(defaults.mutationsModeCounts);
  if (defaultModeCounts.errorKind == AllocationParseErrorKind::InvalidEntry) {
    result.error =
        (Twine("circt-mut matrix: invalid --default-mutations-mode-counts "
               "entry: ") +
         defaultModeCounts.entry + " (expected NAME=COUNT).")
            .str();
    return result;
  }
  if (defaultModeCounts.errorKind == AllocationParseErrorKind::InvalidValue) {
    result.error = (Twine("circt-mut matrix: invalid "
                          "--default-mutations-mode-count value for ") +
                    defaultModeCounts.modeName + ": " +
                    defaultModeCounts.value +
                    " (expected positive integer).")
                       .str();
    return result;
  }

  AllocationParseResult defaultModeWeights =
      parseModeAllocationCSV(defaults.mutationsModeWeights);
  if (defaultModeWeights.errorKind == AllocationParseErrorKind::InvalidEntry) {
    result.error =
        (Twine("circt-mut matrix: invalid --default-mutations-mode-weights "
               "entry: ") +
         defaultModeWeights.entry + " (expected NAME=WEIGHT).")
            .str();
    return result;
  }
  if (defaultModeWeights.errorKind == AllocationParseErrorKind::InvalidValue) {
    result.error = (Twine("circt-mut matrix: invalid "
                          "--default-mutations-mode-weight value for ") +
                    defaultModeWeights.modeName + ": " +
                    defaultModeWeights.value +
                    " (expected positive integer).")
                       .str();
    return result;
  }

  if (defaultModeCounts.enabled && defaultModeWeights.enabled) {
    result.error =
        "circt-mut matrix: use either --default-mutations-mode-counts or "
        "--default-mutations-mode-weights, not both.";
    return result;
  }

  if (hasDefaultGlobalFilterChain) {
    if (defaultGlobalFilterChainMode != "lec-then-bmc" &&
        defaultGlobalFilterChainMode != "bmc-then-lec" &&
        defaultGlobalFilterChainMode != "consensus" &&
        defaultGlobalFilterChainMode != "auto") {
      result.error =
          (Twine("Invalid --default-formal-global-propagate-circt-chain value: ") +
           defaultGlobalFilterChainMode +
           " (expected lec-then-bmc|bmc-then-lec|consensus|auto).")
              .str();
      return result;
    }
    if (hasDefaultGlobalFilterCmd) {
      result.error =
          "--default-formal-global-propagate-circt-chain cannot be combined "
          "with --default-formal-global-propagate-cmd.";
      return result;
    }
    if (!hasDefaultGlobalFilterLEC) {
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, "auto", "circt-lec", "run_mutation_matrix.sh");
      if (!resolved) {
        result.error =
            "circt-mut matrix: unable to resolve --default-formal-global-propagate-circt-lec executable: auto (searched repo build/bin, install-tree sibling bin, and PATH).";
        return result;
      }
      result.rewrittenArgs.push_back("--default-formal-global-propagate-circt-lec");
      result.rewrittenArgs.push_back(*resolved);
      hasDefaultGlobalFilterLEC = true;
    }
    if (!hasDefaultGlobalFilterBMC) {
      auto resolved = resolveCirctToolPathForWorkflow(
          argv0, "auto", "circt-bmc", "run_mutation_matrix.sh");
      if (!resolved) {
        result.error =
            "circt-mut matrix: unable to resolve --default-formal-global-propagate-circt-bmc executable: auto (searched repo build/bin, install-tree sibling bin, and PATH).";
        return result;
      }
      result.rewrittenArgs.push_back("--default-formal-global-propagate-circt-bmc");
      result.rewrittenArgs.push_back(*resolved);
      hasDefaultGlobalFilterBMC = true;
    }
  } else {
    int modeCount = 0;
    modeCount += hasDefaultGlobalFilterCmd ? 1 : 0;
    modeCount += hasDefaultGlobalFilterLEC ? 1 : 0;
    modeCount += hasDefaultGlobalFilterBMC ? 1 : 0;
    if (modeCount > 1) {
      result.error =
          "Use only one default global filter mode: --default-formal-global-propagate-cmd, --default-formal-global-propagate-circt-lec, --default-formal-global-propagate-circt-bmc, or --default-formal-global-propagate-circt-chain.";
      return result;
    }
  }

  auto validateDefaultRegex = [&](StringRef value, const Regex &pattern,
                                  StringRef flag, StringRef expected) -> bool {
    if (value.empty())
      return true;
    if (pattern.match(value))
      return true;
    result.error = (Twine("Invalid ") + flag + " value: " + value +
                    " (expected " + expected + ").")
                       .str();
    return false;
  };
  if (!validateDefaultRegex(defaults.globalFilterTimeoutSeconds,
                            Regex("^[0-9]+$"),
                            "--default-formal-global-propagate-timeout-seconds",
                            "0-9 integer"))
    return result;
  if (!validateDefaultRegex(
          defaults.globalFilterLECTimeoutSeconds, Regex("^[0-9]+$"),
          "--default-formal-global-propagate-lec-timeout-seconds",
          "0-9 integer"))
    return result;
  if (!validateDefaultRegex(
          defaults.globalFilterBMCTimeoutSeconds, Regex("^[0-9]+$"),
          "--default-formal-global-propagate-bmc-timeout-seconds",
          "0-9 integer"))
    return result;
  if (!validateDefaultRegex(defaults.globalFilterBMCBound,
                            Regex("^[1-9][0-9]*$"),
                            "--default-formal-global-propagate-bmc-bound",
                            "positive integer"))
    return result;
  if (!validateDefaultRegex(
          defaults.globalFilterBMCIgnoreAssertsUntil, Regex("^[0-9]+$"),
          "--default-formal-global-propagate-bmc-ignore-asserts-until",
          "0-9 integer"))
    return result;
  if (!validateDefaultRegex(defaults.bmcOrigCacheMaxEntries, Regex("^[0-9]+$"),
                            "--default-bmc-orig-cache-max-entries",
                            "0-9 integer"))
    return result;
  if (!validateDefaultRegex(defaults.bmcOrigCacheMaxBytes, Regex("^[0-9]+$"),
                            "--default-bmc-orig-cache-max-bytes",
                            "0-9 integer"))
    return result;
  if (!validateDefaultRegex(defaults.bmcOrigCacheMaxAgeSeconds,
                            Regex("^[0-9]+$"),
                            "--default-bmc-orig-cache-max-age-seconds",
                            "0-9 integer"))
    return result;
  if (!validateDefaultRegex(defaults.bmcOrigCacheEvictionPolicy,
                            Regex("^(lru|fifo|cost-lru)$"),
                            "--default-bmc-orig-cache-eviction-policy",
                            "lru|fifo|cost-lru"))
    return result;
  bool hasAnyDefaultGlobalFilterMode =
      hasDefaultGlobalFilterCmd || hasDefaultGlobalFilterLEC ||
      hasDefaultGlobalFilterBMC || hasDefaultGlobalFilterChain;
  bool needsTimeoutTool =
      hasAnyDefaultGlobalFilterMode &&
      (hasNonZeroDecimalValue(defaults.globalFilterTimeoutSeconds) ||
       hasNonZeroDecimalValue(defaults.globalFilterLECTimeoutSeconds) ||
       hasNonZeroDecimalValue(defaults.globalFilterBMCTimeoutSeconds));

  bool laneNeedsTimeoutTool = false;
  if (!lanesTSVPath.empty() &&
      !preflightMatrixLaneTools(argv0, lanesTSVPath, defaults, result.error,
                                laneNeedsTimeoutTool))
    return result;
  if ((needsTimeoutTool || laneNeedsTimeoutTool) &&
      !resolveToolPathFromEnvPath("timeout")) {
    result.error = "circt-mut matrix: unable to resolve timeout executable "
                   "required by non-zero global formal timeout settings; "
                   "install coreutils timeout or set PATH.";
    return result;
  }

  result.ok = true;
  return result;
}

static int runCoverFlow(const char *argv0, ArrayRef<StringRef> forwardedArgs) {
  CoverRewriteResult rewrite = rewriteCoverArgs(argv0, forwardedArgs);
  if (!rewrite.ok) {
    errs() << rewrite.error << "\n";
    return 1;
  }

  auto scriptPath = resolveScriptPath(argv0, "run_mutation_cover.sh");
  if (!scriptPath) {
    errs() << "circt-mut: unable to locate script 'run_mutation_cover.sh'.\n";
    errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree with"
              " utils scripts.\n";
    return 1;
  }

  SmallVector<StringRef, 32> rewrittenArgsRef;
  for (const std::string &arg : rewrite.rewrittenArgs)
    rewrittenArgsRef.push_back(arg);
  return dispatchToScript(*scriptPath, rewrittenArgsRef);
}

static int runMatrixFlow(const char *argv0, ArrayRef<StringRef> forwardedArgs) {
  MatrixRewriteResult rewrite = rewriteMatrixArgs(argv0, forwardedArgs);
  if (!rewrite.ok) {
    errs() << rewrite.error << "\n";
    return 1;
  }

  auto scriptPath = resolveScriptPath(argv0, "run_mutation_matrix.sh");
  if (!scriptPath) {
    errs() << "circt-mut: unable to locate script 'run_mutation_matrix.sh'.\n";
    errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree with"
              " utils scripts.\n";
    return 1;
  }

  SmallVector<StringRef, 32> rewrittenArgsRef;
  for (const std::string &arg : rewrite.rewrittenArgs)
    rewrittenArgsRef.push_back(arg);
  return dispatchToScript(*scriptPath, rewrittenArgsRef);
}

struct InitOptions {
  std::string projectDir = ".";
  std::string design = "design.il";
  std::string mutationsFile = "mutations.txt";
  std::string testsManifest = "tests.tsv";
  std::string lanesTSV = "lanes.tsv";
  std::string coverWorkDir = "out/cover";
  std::string matrixOutDir = "out/matrix";
  bool force = false;
};

struct InitParseResult {
  bool ok = false;
  bool showHelp = false;
  std::string error;
  InitOptions opts;
};

static std::string escapeTomlBasicString(StringRef value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    if (c == '\\' || c == '"')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

static bool writeGeneratedFile(StringRef path, StringRef content, bool force,
                               std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (!parent.empty()) {
    std::error_code mkdirEC = sys::fs::create_directories(parent);
    if (mkdirEC) {
      error = (Twine("circt-mut init: failed to create directory: ") + parent +
               ": " + mkdirEC.message())
                  .str();
      return false;
    }
  }

  if (!force && sys::fs::exists(path)) {
    error = (Twine("circt-mut init: file exists: ") + path +
             " (use --force to overwrite).")
                .str();
    return false;
  }

  std::error_code outEC;
  raw_fd_ostream out(path, outEC, sys::fs::OF_Text);
  if (outEC) {
    error = (Twine("circt-mut init: failed to write file: ") + path + ": " +
             outEC.message())
                .str();
    return false;
  }
  out << content;
  out.close();
  return true;
}

static std::string resolveProjectFilePath(StringRef projectDir, StringRef file) {
  if (sys::path::is_absolute(file))
    return file.str();
  SmallString<256> fullPath(projectDir);
  sys::path::append(fullPath, file);
  return std::string(fullPath.str());
}

static InitParseResult parseInitArgs(ArrayRef<StringRef> args) {
  InitParseResult result;

  auto consumeValue = [&](size_t &index, StringRef arg,
                          StringRef optName) -> std::optional<StringRef> {
    size_t eqPos = arg.find('=');
    if (eqPos != StringRef::npos) {
      StringRef value = arg.substr(eqPos + 1);
      if (value.empty()) {
        result.error =
            (Twine("circt-mut init: missing value for ") + optName).str();
        return std::nullopt;
      }
      return value;
    }
    if (index + 1 >= args.size()) {
      result.error = (Twine("circt-mut init: missing value for ") + optName).str();
      return std::nullopt;
    }
    return args[++index];
  };

  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == "-h" || arg == "--help") {
      result.showHelp = true;
      result.ok = true;
      return result;
    }
    if (arg == "--force") {
      result.opts.force = true;
      continue;
    }
    if (arg == "--project-dir" || arg.starts_with("--project-dir=")) {
      auto v = consumeValue(i, arg, "--project-dir");
      if (!v)
        return result;
      result.opts.projectDir = v->str();
      continue;
    }
    if (arg == "--design" || arg.starts_with("--design=")) {
      auto v = consumeValue(i, arg, "--design");
      if (!v)
        return result;
      result.opts.design = v->str();
      continue;
    }
    if (arg == "--mutations-file" || arg.starts_with("--mutations-file=")) {
      auto v = consumeValue(i, arg, "--mutations-file");
      if (!v)
        return result;
      result.opts.mutationsFile = v->str();
      continue;
    }
    if (arg == "--tests-manifest" || arg.starts_with("--tests-manifest=")) {
      auto v = consumeValue(i, arg, "--tests-manifest");
      if (!v)
        return result;
      result.opts.testsManifest = v->str();
      continue;
    }
    if (arg == "--lanes-tsv" || arg.starts_with("--lanes-tsv=")) {
      auto v = consumeValue(i, arg, "--lanes-tsv");
      if (!v)
        return result;
      result.opts.lanesTSV = v->str();
      continue;
    }
    if (arg == "--cover-work-dir" || arg.starts_with("--cover-work-dir=")) {
      auto v = consumeValue(i, arg, "--cover-work-dir");
      if (!v)
        return result;
      result.opts.coverWorkDir = v->str();
      continue;
    }
    if (arg == "--matrix-out-dir" || arg.starts_with("--matrix-out-dir=")) {
      auto v = consumeValue(i, arg, "--matrix-out-dir");
      if (!v)
        return result;
      result.opts.matrixOutDir = v->str();
      continue;
    }

    if (arg.starts_with("-")) {
      result.error = (Twine("circt-mut init: unknown option: ") + arg).str();
      return result;
    }
    result.error =
        (Twine("circt-mut init: unexpected positional argument: ") + arg).str();
    return result;
  }

  if (result.opts.projectDir.empty()) {
    result.error = "circt-mut init: --project-dir must not be empty";
    return result;
  }

  result.ok = true;
  return result;
}

static int runNativeInit(const InitOptions &opts) {
  std::error_code mkdirEC = sys::fs::create_directories(opts.projectDir);
  if (mkdirEC) {
    errs() << "circt-mut init: failed to create --project-dir: "
           << opts.projectDir << ": " << mkdirEC.message() << "\n";
    return 1;
  }

  std::string configPath = resolveProjectFilePath(opts.projectDir, "circt-mut.toml");
  std::string testsPath =
      resolveProjectFilePath(opts.projectDir, opts.testsManifest);
  std::string lanesPath = resolveProjectFilePath(opts.projectDir, opts.lanesTSV);

  std::string config;
  raw_string_ostream cfg(config);
  cfg << "# CIRCT mutation campaign configuration\n";
  cfg << "# Bootstrap generated by: circt-mut init\n";
  cfg << "#\n";
  cfg << "# MCY comparison:\n";
  cfg << "#   mcy init && mcy run -j8\n";
  cfg << "# Certitude comparison:\n";
  cfg << "#   certitude_run -config <cfg> -out <dir>\n\n";
  cfg << "[cover]\n";
  cfg << "design = \"" << escapeTomlBasicString(opts.design) << "\"\n";
  cfg << "mutations_file = \"" << escapeTomlBasicString(opts.mutationsFile)
      << "\"\n";
  cfg << "tests_manifest = \"" << escapeTomlBasicString(opts.testsManifest)
      << "\"\n";
  cfg << "work_dir = \"" << escapeTomlBasicString(opts.coverWorkDir) << "\"\n";
  cfg << "formal_global_propagate_circt_chain = \"auto\"\n";
  cfg << "formal_global_propagate_timeout_seconds = 60\n\n";
  cfg << "[matrix]\n";
  cfg << "lanes_tsv = \"" << escapeTomlBasicString(opts.lanesTSV) << "\"\n";
  cfg << "out_dir = \"" << escapeTomlBasicString(opts.matrixOutDir) << "\"\n";
  cfg << "default_formal_global_propagate_circt_chain = \"auto\"\n";
  cfg << "default_formal_global_propagate_timeout_seconds = 60\n";
  cfg.flush();

  std::string testsTemplate;
  raw_string_ostream testsOS(testsTemplate);
  testsOS << "# test_id<TAB>run_cmd<TAB>result_file<TAB>detect_regex<TAB>"
          << "survive_regex\n";
  testsOS << "sanity\tbash run_test.sh\tresult.txt\t^DETECTED$\t^SURVIVED$\n";
  testsOS.flush();

  std::string lanesTemplate;
  raw_string_ostream lanesOS(lanesTemplate);
  lanesOS << "# lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<TAB>"
          << "activate_cmd<TAB>propagate_cmd<TAB>coverage_threshold\n";
  lanesOS << "lane1\t" << opts.design << "\t" << opts.mutationsFile << "\t"
          << opts.testsManifest << "\t-\t-\t-\n";
  lanesOS.flush();

  std::string error;
  if (!writeGeneratedFile(configPath, config, opts.force, error)) {
    errs() << error << "\n";
    return 1;
  }
  if (!writeGeneratedFile(testsPath, testsTemplate, opts.force, error)) {
    errs() << error << "\n";
    return 1;
  }
  if (!writeGeneratedFile(lanesPath, lanesTemplate, opts.force, error)) {
    errs() << error << "\n";
    return 1;
  }

  outs() << "Initialized circt-mut project: " << opts.projectDir << "\n";
  outs() << "  Config: " << configPath << "\n";
  outs() << "  Tests manifest template: " << testsPath << "\n";
  outs() << "  Lanes template: " << lanesPath << "\n";
  return 0;
}

struct RunOptions {
  std::string projectDir = ".";
  std::string configPath;
  std::string mode = "all";
};

struct RunParseResult {
  bool ok = false;
  bool showHelp = false;
  std::string error;
  RunOptions opts;
};

struct RunConfigValues {
  StringMap<std::string> cover;
  StringMap<std::string> matrix;
};

static bool parseTomlQuotedString(StringRef value, std::string &out,
                                  std::string &error, StringRef path,
                                  size_t lineNo) {
  if (value.size() < 2 || !value.starts_with("\"") || !value.ends_with("\"")) {
    error = (Twine("circt-mut run: invalid TOML string in ") + path + ":" +
             Twine(static_cast<unsigned long long>(lineNo)) + ": " + value)
                .str();
    return false;
  }
  out.clear();
  for (size_t i = 1; i + 1 < value.size(); ++i) {
    char c = value[i];
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (i + 2 >= value.size()) {
      error = (Twine("circt-mut run: invalid TOML escape in ") + path + ":" +
               Twine(static_cast<unsigned long long>(lineNo)))
                  .str();
      return false;
    }
    char next = value[++i];
    switch (next) {
    case '\\':
    case '"':
      out.push_back(next);
      break;
    case 'n':
      out.push_back('\n');
      break;
    case 't':
      out.push_back('\t');
      break;
    default:
      error = (Twine("circt-mut run: unsupported TOML escape in ") + path +
               ":" + Twine(static_cast<unsigned long long>(lineNo)))
                  .str();
      return false;
    }
  }
  return true;
}

static bool parseRunConfigFile(StringRef path, RunConfigValues &cfg,
                               std::string &error) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut run: unable to read config file: ") + path).str();
    return false;
  }

  StringRef section;
  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  for (size_t i = 0; i < lines.size(); ++i) {
    StringRef rawLine = lines[i];
    StringRef line = rawLine.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    if (line.starts_with("[") && line.ends_with("]")) {
      section = line.drop_front().drop_back().trim();
      continue;
    }

    size_t eqPos = line.find('=');
    if (eqPos == StringRef::npos)
      continue;
    StringRef key = line.substr(0, eqPos).trim();
    StringRef value = line.substr(eqPos + 1).trim();
    if (key.empty())
      continue;

    if (!value.starts_with("\"")) {
      size_t hashPos = value.find('#');
      if (hashPos != StringRef::npos)
        value = value.substr(0, hashPos).trim();
    }

    std::string parsedValue;
    if (value.starts_with("\"")) {
      if (!parseTomlQuotedString(value, parsedValue, error, path, i + 1))
        return false;
    } else {
      parsedValue = value.str();
    }

    if (section == "cover")
      cfg.cover[key] = parsedValue;
    else if (section == "matrix")
      cfg.matrix[key] = parsedValue;
  }

  return true;
}

static bool appendRequiredConfigArg(SmallVectorImpl<std::string> &args,
                                    const StringMap<std::string> &sectionMap,
                                    StringRef key, StringRef optionFlag,
                                    StringRef sectionName, StringRef projectDir,
                                    bool resolveRelativePath,
                                    std::string &error) {
  auto it = sectionMap.find(key);
  if (it == sectionMap.end() || it->second.empty()) {
    error =
        (Twine("circt-mut run: missing required [") + sectionName + "] key '" +
         key + "' in config.")
            .str();
    return false;
  }
  std::string value = it->second;
  if (resolveRelativePath && !sys::path::is_absolute(value)) {
    SmallString<256> joined(projectDir);
    sys::path::append(joined, value);
    value = std::string(joined.str());
  }
  args.push_back(optionFlag.str());
  args.push_back(value);
  return true;
}

static void appendOptionalConfigPathArg(SmallVectorImpl<std::string> &args,
                                        const StringMap<std::string> &sectionMap,
                                        StringRef key, StringRef optionFlag,
                                        StringRef projectDir) {
  auto it = sectionMap.find(key);
  if (it == sectionMap.end() || it->second.empty())
    return;
  std::string value = it->second;
  if (!value.empty() && !sys::path::is_absolute(value) &&
      StringRef(value).contains('/')) {
    SmallString<256> joined(projectDir);
    sys::path::append(joined, value);
    value = std::string(joined.str());
  }
  args.push_back(optionFlag.str());
  args.push_back(value);
}

static void appendOptionalConfigArg(SmallVectorImpl<std::string> &args,
                                    const StringMap<std::string> &sectionMap,
                                    StringRef key, StringRef optionFlag) {
  auto it = sectionMap.find(key);
  if (it == sectionMap.end() || it->second.empty())
    return;
  args.push_back(optionFlag.str());
  args.push_back(it->second);
}

static bool appendOptionalConfigBoolFlagArg(
    SmallVectorImpl<std::string> &args, const StringMap<std::string> &sectionMap,
    StringRef key, StringRef optionFlag, StringRef sectionName,
    std::string &error) {
  auto it = sectionMap.find(key);
  if (it == sectionMap.end() || it->second.empty())
    return true;
  std::string lowered = StringRef(it->second).trim().lower();
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    args.push_back(optionFlag.str());
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off")
    return true;
  error = (Twine("circt-mut run: invalid boolean [") + sectionName + "] key '" +
           key + "' value '" + it->second +
           "' (expected 1|0|true|false|yes|no|on|off)")
              .str();
  return false;
}

static RunParseResult parseRunArgs(ArrayRef<StringRef> args) {
  RunParseResult result;

  auto consumeValue = [&](size_t &index, StringRef arg,
                          StringRef optName) -> std::optional<StringRef> {
    size_t eqPos = arg.find('=');
    if (eqPos != StringRef::npos) {
      StringRef value = arg.substr(eqPos + 1);
      if (value.empty()) {
        result.error =
            (Twine("circt-mut run: missing value for ") + optName).str();
        return std::nullopt;
      }
      return value;
    }
    if (index + 1 >= args.size()) {
      result.error = (Twine("circt-mut run: missing value for ") + optName).str();
      return std::nullopt;
    }
    return args[++index];
  };

  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == "-h" || arg == "--help") {
      result.showHelp = true;
      result.ok = true;
      return result;
    }
    if (arg == "--project-dir" || arg.starts_with("--project-dir=")) {
      auto v = consumeValue(i, arg, "--project-dir");
      if (!v)
        return result;
      result.opts.projectDir = v->str();
      continue;
    }
    if (arg == "--config" || arg.starts_with("--config=")) {
      auto v = consumeValue(i, arg, "--config");
      if (!v)
        return result;
      result.opts.configPath = v->str();
      continue;
    }
    if (arg == "--mode" || arg.starts_with("--mode=")) {
      auto v = consumeValue(i, arg, "--mode");
      if (!v)
        return result;
      result.opts.mode = v->str();
      continue;
    }

    if (arg.starts_with("-")) {
      result.error = (Twine("circt-mut run: unknown option: ") + arg).str();
      return result;
    }
    result.error =
        (Twine("circt-mut run: unexpected positional argument: ") + arg).str();
    return result;
  }

  if (result.opts.mode != "cover" && result.opts.mode != "matrix" &&
      result.opts.mode != "all") {
    result.error =
        (Twine("circt-mut run: invalid --mode value: ") + result.opts.mode +
         " (expected cover|matrix|all)")
            .str();
    return result;
  }

  result.ok = true;
  return result;
}

static int runNativeRun(const char *argv0, const RunOptions &opts) {
  SmallString<256> configPath;
  if (opts.configPath.empty()) {
    configPath = opts.projectDir;
    sys::path::append(configPath, "circt-mut.toml");
  } else if (sys::path::is_absolute(opts.configPath)) {
    configPath = opts.configPath;
  } else {
    configPath = opts.projectDir;
    sys::path::append(configPath, opts.configPath);
  }

  RunConfigValues cfg;
  std::string error;
  if (!parseRunConfigFile(configPath, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  auto runCoverFromConfig = [&]() -> int {
    SmallVector<std::string, 96> args;
    if (!appendRequiredConfigArg(args, cfg.cover, "design", "--design", "cover",
                                 opts.projectDir, /*resolveRelativePath=*/true,
                                 error) ||
        !appendRequiredConfigArg(args, cfg.cover, "tests_manifest",
                                 "--tests-manifest", "cover", opts.projectDir,
                                 /*resolveRelativePath=*/true, error) ||
        !appendRequiredConfigArg(args, cfg.cover, "work_dir", "--work-dir",
                                 "cover", opts.projectDir,
                                 /*resolveRelativePath=*/true, error)) {
      errs() << error << "\n";
      return 1;
    }

    auto mutationFileIt = cfg.cover.find("mutations_file");
    auto generateMutationsIt = cfg.cover.find("generate_mutations");
    bool hasMutationsFile =
        mutationFileIt != cfg.cover.end() && !mutationFileIt->second.empty();
    bool hasGenerateMutations = generateMutationsIt != cfg.cover.end() &&
                                !generateMutationsIt->second.empty();
    if (hasMutationsFile && hasGenerateMutations) {
      errs() << "circt-mut run: [cover] keys 'mutations_file' and "
                "'generate_mutations' are mutually exclusive.\n";
      return 1;
    }
    if (!hasMutationsFile && !hasGenerateMutations) {
      errs() << "circt-mut run: [cover] requires either 'mutations_file' or "
                "'generate_mutations'.\n";
      return 1;
    }
    if (hasMutationsFile) {
      if (!appendRequiredConfigArg(args, cfg.cover, "mutations_file",
                                   "--mutations-file", "cover",
                                   opts.projectDir,
                                   /*resolveRelativePath=*/true, error)) {
        errs() << error << "\n";
        return 1;
      }
    } else {
      appendOptionalConfigArg(args, cfg.cover, "generate_mutations",
                              "--generate-mutations");
      appendOptionalConfigArg(args, cfg.cover, "mutations_top", "--mutations-top");
      appendOptionalConfigArg(args, cfg.cover, "mutations_seed",
                              "--mutations-seed");
      appendOptionalConfigPathArg(args, cfg.cover, "mutations_yosys",
                                  "--mutations-yosys", opts.projectDir);
      appendOptionalConfigArg(args, cfg.cover, "mutations_modes",
                              "--mutations-modes");
      appendOptionalConfigArg(args, cfg.cover, "mutations_mode_counts",
                              "--mutations-mode-counts");
      appendOptionalConfigArg(args, cfg.cover, "mutations_mode_weights",
                              "--mutations-mode-weights");
      appendOptionalConfigArg(args, cfg.cover, "mutations_profiles",
                              "--mutations-profiles");
      appendOptionalConfigArg(args, cfg.cover, "mutations_cfg", "--mutations-cfg");
      appendOptionalConfigArg(args, cfg.cover, "mutations_select",
                              "--mutations-select");
    }

    appendOptionalConfigPathArg(args, cfg.cover, "create_mutated_script",
                                "--create-mutated-script", opts.projectDir);
    appendOptionalConfigArg(args, cfg.cover, "formal_activate_cmd",
                            "--formal-activate-cmd");
    appendOptionalConfigArg(args, cfg.cover, "formal_propagate_cmd",
                            "--formal-propagate-cmd");
    appendOptionalConfigArg(args, cfg.cover, "coverage_threshold",
                            "--coverage-threshold");
    appendOptionalConfigArg(args, cfg.cover, "jobs", "--jobs");
    appendOptionalConfigArg(args, cfg.cover, "reuse_compat_mode",
                            "--reuse-compat-mode");
    appendOptionalConfigPathArg(args, cfg.cover, "reuse_pair_file",
                                "--reuse-pair-file", opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.cover, "reuse_summary_file",
                                "--reuse-summary-file", opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.cover, "reuse_manifest_file",
                                "--reuse-manifest-file", opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.cover, "reuse_cache_dir",
                                "--reuse-cache-dir", opts.projectDir);
    appendOptionalConfigArg(args, cfg.cover, "reuse_cache_mode",
                            "--reuse-cache-mode");

    if (!appendOptionalConfigBoolFlagArg(args, cfg.cover, "resume", "--resume",
                                         "cover", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.cover, "skip_baseline",
                                         "--skip-baseline", "cover", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.cover, "fail_on_undetected",
                                         "--fail-on-undetected", "cover",
                                         error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.cover, "fail_on_errors",
                                         "--fail-on-errors", "cover", error)) {
      errs() << error << "\n";
      return 1;
    }

    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_cmd",
                            "--formal-global-propagate-cmd");
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_circt_chain",
                            "--formal-global-propagate-circt-chain");
    appendOptionalConfigArg(args, cfg.cover,
                            "formal_global_propagate_timeout_seconds",
                            "--formal-global-propagate-timeout-seconds");
    appendOptionalConfigArg(args, cfg.cover,
                            "formal_global_propagate_lec_timeout_seconds",
                            "--formal-global-propagate-lec-timeout-seconds");
    appendOptionalConfigArg(args, cfg.cover,
                            "formal_global_propagate_bmc_timeout_seconds",
                            "--formal-global-propagate-bmc-timeout-seconds");
    appendOptionalConfigPathArg(args, cfg.cover, "formal_global_propagate_circt_lec",
                                "--formal-global-propagate-circt-lec",
                                opts.projectDir);
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_circt_lec_args",
                            "--formal-global-propagate-circt-lec-args");
    appendOptionalConfigPathArg(args, cfg.cover, "formal_global_propagate_circt_bmc",
                                "--formal-global-propagate-circt-bmc",
                                opts.projectDir);
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_c1",
                            "--formal-global-propagate-c1");
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_c2",
                            "--formal-global-propagate-c2");
    appendOptionalConfigPathArg(args, cfg.cover, "formal_global_propagate_z3",
                                "--formal-global-propagate-z3", opts.projectDir);
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.cover, "formal_global_propagate_assume_known_inputs",
            "--formal-global-propagate-assume-known-inputs", "cover", error) ||
        !appendOptionalConfigBoolFlagArg(
            args, cfg.cover, "formal_global_propagate_accept_xprop_only",
            "--formal-global-propagate-accept-xprop-only", "cover", error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_bmc_args",
                            "--formal-global-propagate-circt-bmc-args");
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_bmc_bound",
                            "--formal-global-propagate-bmc-bound");
    appendOptionalConfigArg(args, cfg.cover, "formal_global_propagate_bmc_module",
                            "--formal-global-propagate-bmc-module");
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.cover, "formal_global_propagate_bmc_run_smtlib",
            "--formal-global-propagate-bmc-run-smtlib", "cover", error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigPathArg(args, cfg.cover, "formal_global_propagate_bmc_z3",
                                "--formal-global-propagate-bmc-z3",
                                opts.projectDir);
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.cover, "formal_global_propagate_bmc_assume_known_inputs",
            "--formal-global-propagate-bmc-assume-known-inputs", "cover",
            error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigArg(args, cfg.cover,
                            "formal_global_propagate_bmc_ignore_asserts_until",
                            "--formal-global-propagate-bmc-ignore-asserts-until");
    appendOptionalConfigArg(args, cfg.cover, "bmc_orig_cache_max_entries",
                            "--bmc-orig-cache-max-entries");
    appendOptionalConfigArg(args, cfg.cover, "bmc_orig_cache_max_bytes",
                            "--bmc-orig-cache-max-bytes");
    appendOptionalConfigArg(args, cfg.cover, "bmc_orig_cache_max_age_seconds",
                            "--bmc-orig-cache-max-age-seconds");
    appendOptionalConfigArg(args, cfg.cover, "bmc_orig_cache_eviction_policy",
                            "--bmc-orig-cache-eviction-policy");

    SmallVector<StringRef, 96> argRefs;
    for (const auto &arg : args)
      argRefs.push_back(arg);
    return runCoverFlow(argv0, argRefs);
  };

  auto runMatrixFromConfig = [&]() -> int {
    SmallVector<std::string, 96> args;
    if (!appendRequiredConfigArg(args, cfg.matrix, "lanes_tsv", "--lanes-tsv",
                                 "matrix", opts.projectDir,
                                 /*resolveRelativePath=*/true, error) ||
        !appendRequiredConfigArg(args, cfg.matrix, "out_dir", "--out-dir",
                                 "matrix", opts.projectDir,
                                 /*resolveRelativePath=*/true, error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigPathArg(args, cfg.matrix, "create_mutated_script",
                                "--create-mutated-script", opts.projectDir);
    appendOptionalConfigArg(args, cfg.matrix, "jobs_per_lane",
                            "--jobs-per-lane");
    appendOptionalConfigArg(args, cfg.matrix, "lane_jobs", "--lane-jobs");
    appendOptionalConfigArg(args, cfg.matrix, "lane_schedule_policy",
                            "--lane-schedule-policy");
    appendOptionalConfigArg(args, cfg.matrix, "results_file", "--results-file");
    appendOptionalConfigArg(args, cfg.matrix, "gate_summary_file",
                            "--gate-summary-file");
    appendOptionalConfigArg(args, cfg.matrix, "reuse_compat_mode",
                            "--reuse-compat-mode");
    appendOptionalConfigPathArg(args, cfg.matrix, "reuse_cache_dir",
                                "--reuse-cache-dir", opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.matrix, "default_reuse_pair_file",
                                "--default-reuse-pair-file", opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.matrix, "default_reuse_summary_file",
                                "--default-reuse-summary-file", opts.projectDir);
    appendOptionalConfigArg(args, cfg.matrix, "include_lane_regex",
                            "--include-lane-regex");
    appendOptionalConfigArg(args, cfg.matrix, "exclude_lane_regex",
                            "--exclude-lane-regex");
    if (!appendOptionalConfigBoolFlagArg(args, cfg.matrix, "skip_baseline",
                                         "--skip-baseline", "matrix", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.matrix, "fail_on_undetected",
                                         "--fail-on-undetected", "matrix",
                                         error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.matrix, "fail_on_errors",
                                         "--fail-on-errors", "matrix", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.matrix, "stop_on_fail",
                                         "--stop-on-fail", "matrix", error)) {
      errs() << error << "\n";
      return 1;
    }

    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_modes",
                            "--default-mutations-modes");
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_mode_counts",
                            "--default-mutations-mode-counts");
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_mode_weights",
                            "--default-mutations-mode-weights");
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_profiles",
                            "--default-mutations-profiles");
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_cfg",
                            "--default-mutations-cfg");
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_select",
                            "--default-mutations-select");
    appendOptionalConfigPathArg(args, cfg.matrix, "default_mutations_yosys",
                                "--default-mutations-yosys", opts.projectDir);

    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_cmd",
                            "--default-formal-global-propagate-cmd");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_circt_chain",
                            "--default-formal-global-propagate-circt-chain");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_timeout_seconds",
                            "--default-formal-global-propagate-timeout-seconds");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_lec_timeout_seconds",
                            "--default-formal-global-propagate-lec-timeout-seconds");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_bmc_timeout_seconds",
                            "--default-formal-global-propagate-bmc-timeout-seconds");
    appendOptionalConfigPathArg(args, cfg.matrix,
                            "default_formal_global_propagate_circt_lec",
                            "--default-formal-global-propagate-circt-lec",
                            opts.projectDir);
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_circt_lec_args",
                            "--default-formal-global-propagate-circt-lec-args");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_c1",
                            "--default-formal-global-propagate-c1");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_c2",
                            "--default-formal-global-propagate-c2");
    appendOptionalConfigPathArg(args, cfg.matrix,
                            "default_formal_global_propagate_z3",
                            "--default-formal-global-propagate-z3",
                            opts.projectDir);
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.matrix, "default_formal_global_propagate_assume_known_inputs",
            "--default-formal-global-propagate-assume-known-inputs", "matrix",
            error) ||
        !appendOptionalConfigBoolFlagArg(
            args, cfg.matrix, "default_formal_global_propagate_accept_xprop_only",
            "--default-formal-global-propagate-accept-xprop-only", "matrix",
            error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigPathArg(args, cfg.matrix,
                            "default_formal_global_propagate_circt_bmc",
                            "--default-formal-global-propagate-circt-bmc",
                            opts.projectDir);
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_circt_bmc_args",
                            "--default-formal-global-propagate-circt-bmc-args");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_bmc_bound",
                            "--default-formal-global-propagate-bmc-bound");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_bmc_module",
                            "--default-formal-global-propagate-bmc-module");
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.matrix, "default_formal_global_propagate_bmc_run_smtlib",
            "--default-formal-global-propagate-bmc-run-smtlib", "matrix",
            error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigPathArg(args, cfg.matrix,
                                "default_formal_global_propagate_bmc_z3",
                                "--default-formal-global-propagate-bmc-z3",
                                opts.projectDir);
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.matrix,
            "default_formal_global_propagate_bmc_assume_known_inputs",
            "--default-formal-global-propagate-bmc-assume-known-inputs",
            "matrix", error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_formal_global_propagate_bmc_ignore_asserts_until",
                            "--default-formal-global-propagate-bmc-ignore-asserts-until");
    appendOptionalConfigArg(args, cfg.matrix, "default_bmc_orig_cache_max_entries",
                            "--default-bmc-orig-cache-max-entries");
    appendOptionalConfigArg(args, cfg.matrix, "default_bmc_orig_cache_max_bytes",
                            "--default-bmc-orig-cache-max-bytes");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_bmc_orig_cache_max_age_seconds",
                            "--default-bmc-orig-cache-max-age-seconds");
    appendOptionalConfigArg(args, cfg.matrix,
                            "default_bmc_orig_cache_eviction_policy",
                            "--default-bmc-orig-cache-eviction-policy");

    SmallVector<StringRef, 96> argRefs;
    for (const auto &arg : args)
      argRefs.push_back(arg);
    return runMatrixFlow(argv0, argRefs);
  };

  if (opts.mode == "cover")
    return runCoverFromConfig();
  if (opts.mode == "matrix")
    return runMatrixFromConfig();

  int rc = runCoverFromConfig();
  if (rc != 0)
    return rc;
  return runMatrixFromConfig();
}

struct ReportOptions {
  std::string projectDir = ".";
  std::string configPath;
  bool configExplicit = false;
  std::string mode = "all";
  std::string coverWorkDir;
  std::string matrixOutDir;
  std::string compareFile;
  std::string compareHistoryLatestFile;
  std::string trendHistoryFile;
  uint64_t trendWindowRuns = 0;
  SmallVector<std::string, 4> policyProfiles;
  std::string appendHistoryFile;
  SmallVector<DeltaGateRule, 8> failIfDeltaGtRules;
  SmallVector<DeltaGateRule, 8> failIfDeltaLtRules;
  SmallVector<DeltaGateRule, 8> failIfTrendDeltaGtRules;
  SmallVector<DeltaGateRule, 8> failIfTrendDeltaLtRules;
  std::string outFile;
};

struct ReportParseResult {
  bool ok = false;
  bool showHelp = false;
  std::string error;
  ReportOptions opts;
};

static std::string rewriteErrorPrefix(StringRef error, StringRef oldPrefix,
                                      StringRef newPrefix) {
  if (!error.starts_with(oldPrefix))
    return error.str();
  return (Twine(newPrefix) + error.drop_front(oldPrefix.size())).str();
}

static void splitTSVLine(StringRef line, SmallVectorImpl<StringRef> &fields) {
  fields.clear();
  while (true) {
    size_t tabPos = line.find('\t');
    if (tabPos == StringRef::npos) {
      fields.push_back(line);
      return;
    }
    fields.push_back(line.substr(0, tabPos));
    line = line.substr(tabPos + 1);
  }
}

static std::string resolveRelativeTo(StringRef baseDir, StringRef path) {
  if (path.empty())
    return std::string(path);
  if (sys::path::is_absolute(path))
    return std::string(path);
  SmallString<256> joined(baseDir);
  sys::path::append(joined, path);
  return std::string(joined.str());
}

static bool parseKeyValueTSV(StringRef path, StringMap<std::string> &values,
                             std::string &error) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read file: ") + path).str();
    return false;
  }

  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.rtrim("\r").trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    size_t tabPos = line.find('\t');
    if (tabPos == StringRef::npos)
      continue;
    StringRef key = line.substr(0, tabPos).trim();
    StringRef value = line.substr(tabPos + 1).trim();
    if (key.empty() || key == "metric" || key == "key")
      continue;
    values[key] = value.str();
  }
  return true;
}

static void appendMetricRow(std::vector<std::pair<std::string, std::string>> &rows,
                            StringRef prefix,
                            const StringMap<std::string> &metrics, StringRef key,
                            StringRef fallback = "-") {
  auto it = metrics.find(key);
  if (it == metrics.end() || it->second.empty()) {
    rows.emplace_back((Twine(prefix) + "." + key).str(), fallback.str());
    return;
  }
  rows.emplace_back((Twine(prefix) + "." + key).str(), it->second);
}

static std::optional<double> parseOptionalDouble(StringRef value) {
  value = value.trim();
  if (value.empty() || value == "-")
    return std::nullopt;
  std::string tmp = value.str();
  char *end = nullptr;
  double parsed = std::strtod(tmp.c_str(), &end);
  if (end == tmp.c_str() || !end || *end != '\0')
    return std::nullopt;
  return parsed;
}

static std::string formatDouble2(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.2f", value);
  return std::string(buffer);
}

static std::string formatCurrentUTCISO8601() {
  auto now = std::chrono::system_clock::now();
  std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  std::tm tmBuf = {};
#if defined(_WIN32)
  gmtime_s(&tmBuf, &nowTime);
#else
  gmtime_r(&nowTime, &tmBuf);
#endif
  char out[32];
  std::strftime(out, sizeof(out), "%Y-%m-%dT%H:%M:%SZ", &tmBuf);
  return std::string(out);
}

static bool parseDeltaGateRule(StringRef value, StringRef optionName,
                               DeltaGateRule &rule, std::string &error) {
  size_t eqPos = value.find('=');
  if (eqPos == StringRef::npos || eqPos == 0 || eqPos + 1 >= value.size()) {
    error = (Twine("circt-mut report: invalid ") + optionName +
             " value '" + value + "' (expected <metric>=<number>)")
                .str();
    return false;
  }
  StringRef key = value.substr(0, eqPos).trim();
  StringRef thresholdStr = value.substr(eqPos + 1).trim();
  if (key.empty() || thresholdStr.empty()) {
    error = (Twine("circt-mut report: invalid ") + optionName +
             " value '" + value + "' (expected <metric>=<number>)")
                .str();
    return false;
  }
  std::string thresholdStorage = thresholdStr.str();
  char *end = nullptr;
  double parsed = std::strtod(thresholdStorage.c_str(), &end);
  if (end == thresholdStorage.c_str() || !end || *end != '\0') {
    error = (Twine("circt-mut report: invalid numeric threshold in ") +
             optionName + " value '" + value + "'")
                .str();
    return false;
  }
  rule.key = key.str();
  rule.threshold = parsed;
  return true;
}

static bool appendUniqueRule(SmallVectorImpl<DeltaGateRule> &rules, StringRef key,
                             double threshold) {
  for (const auto &rule : rules) {
    if (rule.key == key && rule.threshold == threshold)
      return false;
  }
  DeltaGateRule rule;
  rule.key = key.str();
  rule.threshold = threshold;
  rules.push_back(rule);
  return true;
}

static bool applyPolicyProfile(StringRef profile, ReportOptions &opts,
                               std::string &error) {
  if (profile == "formal-regression-basic") {
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "cover.global_filter_timeout_mutants", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "cover.global_filter_lec_unknown_mutants", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "cover.global_filter_bmc_unknown_mutants", 0.0);
    appendUniqueRule(opts.failIfDeltaLtRules, "cover.detected_mutants", 0.0);
    return true;
  }
  if (profile == "formal-regression-trend") {
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "cover.global_filter_timeout_mutants", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "cover.global_filter_lec_unknown_mutants", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "cover.global_filter_bmc_unknown_mutants", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules, "cover.detected_mutants",
                     0.0);
    return true;
  }
  error = (Twine("circt-mut report: unknown --policy-profile value: ") + profile +
           " (expected formal-regression-basic|formal-regression-trend)")
              .str();
  return false;
}

static void appendReportComparisonRows(
    ArrayRef<std::pair<std::string, std::string>> currentRows,
    const StringMap<std::string> &baselineValues, StringRef baselineLabel,
    std::vector<std::pair<std::string, std::string>> &rows,
    StringMap<double> &numericDeltas) {
  StringMap<std::string> currentValues;
  for (const auto &row : currentRows)
    currentValues[row.first] = row.second;

  uint64_t overlapKeys = 0;
  uint64_t addedKeys = 0;
  uint64_t missingKeys = 0;
  uint64_t numericOverlapKeys = 0;
  uint64_t exactChangedKeys = 0;

  for (const auto &it : currentValues) {
    StringRef key = it.getKey();
    if (key.starts_with("compare.") || key.starts_with("diff.") ||
        key.starts_with("policy."))
      continue;
    auto baselineIt = baselineValues.find(key);
    if (baselineIt == baselineValues.end()) {
      ++addedKeys;
      continue;
    }
    ++overlapKeys;
    StringRef currentValue = it.getValue();
    StringRef baselineValue = baselineIt->second;
    if (currentValue != baselineValue)
      ++exactChangedKeys;

    auto currentNum = parseOptionalDouble(currentValue);
    auto baselineNum = parseOptionalDouble(baselineValue);
    if (!currentNum || !baselineNum)
      continue;
    ++numericOverlapKeys;
    double delta = *currentNum - *baselineNum;
    numericDeltas[key] = delta;
    rows.emplace_back((Twine("diff.") + key + ".delta").str(), formatDouble2(delta));
    rows.emplace_back((Twine("diff.") + key + ".pct_change").str(),
                      *baselineNum != 0.0
                          ? formatDouble2((100.0 * delta) / *baselineNum)
                          : std::string("-"));
  }

  for (const auto &it : baselineValues) {
    StringRef key = it.getKey();
    if (key.starts_with("compare.") || key.starts_with("diff.") ||
        key.starts_with("policy."))
      continue;
    if (!currentValues.count(it.getKey()))
      ++missingKeys;
  }

  rows.emplace_back("compare.baseline_file", std::string(baselineLabel));
  rows.emplace_back("diff.overlap_keys", std::to_string(overlapKeys));
  rows.emplace_back("diff.numeric_overlap_keys", std::to_string(numericOverlapKeys));
  rows.emplace_back("diff.exact_changed_keys", std::to_string(exactChangedKeys));
  rows.emplace_back("diff.added_keys", std::to_string(addedKeys));
  rows.emplace_back("diff.missing_keys", std::to_string(missingKeys));
}

static bool appendReportComparison(
    ArrayRef<std::pair<std::string, std::string>> currentRows, StringRef baselinePath,
    std::vector<std::pair<std::string, std::string>> &rows,
    StringMap<double> &numericDeltas, std::string &error) {
  StringMap<std::string> baselineValues;
  if (!parseKeyValueTSV(baselinePath, baselineValues, error))
    return false;

  appendReportComparisonRows(currentRows, baselineValues, baselinePath, rows,
                             numericDeltas);
  return true;
}

static bool loadLatestHistorySnapshot(StringRef path, uint64_t &latestRunID,
                                      StringMap<std::string> &values,
                                      std::string &error) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read history file: ") + path).str();
    return false;
  }

  latestRunID = 0;
  bool haveRows = false;
  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  SmallVector<StringRef, 8> fields;
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.rtrim("\r").trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    splitTSVLine(line, fields);
    if (fields.size() < 4)
      continue;
    StringRef runIDField = fields[0].trim();
    if (runIDField.empty() || runIDField == "run_id")
      continue;
    uint64_t runID = 0;
    if (runIDField.getAsInteger(10, runID)) {
      error = (Twine("circt-mut report: invalid history run_id '") + runIDField +
               "' in " + path)
                  .str();
      return false;
    }
    StringRef key = fields[2].trim();
    StringRef value = fields[3].trim();
    if (key.empty())
      continue;
    if (!haveRows || runID > latestRunID) {
      latestRunID = runID;
      values.clear();
      haveRows = true;
    }
    if (runID == latestRunID)
      values[key] = value.str();
  }

  if (!haveRows) {
    error = (Twine("circt-mut report: no history snapshots found in: ") + path).str();
    return false;
  }
  return true;
}

static bool readHistoryMaxRunID(StringRef path, uint64_t &maxRunID,
                                std::string &error) {
  maxRunID = 0;
  if (!sys::fs::exists(path))
    return true;

  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read history file: ") + path).str();
    return false;
  }

  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  SmallVector<StringRef, 8> fields;
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.rtrim("\r").trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    splitTSVLine(line, fields);
    if (fields.size() < 1)
      continue;
    StringRef runIDField = fields[0].trim();
    if (runIDField.empty() || runIDField == "run_id")
      continue;
    uint64_t runID = 0;
    if (runIDField.getAsInteger(10, runID)) {
      error = (Twine("circt-mut report: invalid history run_id '") + runIDField +
               "' in " + path)
                  .str();
      return false;
    }
    maxRunID = std::max(maxRunID, runID);
  }
  return true;
}

static bool appendHistorySnapshot(
    StringRef path, uint64_t runID, StringRef timestampUTC,
    ArrayRef<std::pair<std::string, std::string>> rows, std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (!parent.empty()) {
    std::error_code dirEC = sys::fs::create_directories(parent);
    if (dirEC) {
      error = (Twine("circt-mut report: failed to create history directory: ") +
               parent + ": " + dirEC.message())
                  .str();
      return false;
    }
  }

  bool existed = sys::fs::exists(path);
  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_Text | sys::fs::OF_Append);
  if (ec) {
    error = (Twine("circt-mut report: failed to open history file: ") + path +
             ": " + ec.message())
                .str();
    return false;
  }
  if (!existed)
    os << "run_id\ttimestamp_utc\tkey\tvalue\n";

  for (const auto &row : rows) {
    StringRef key = row.first;
    if (key.starts_with("compare.") || key.starts_with("diff.") ||
        key.starts_with("history.") || key.starts_with("trend.") ||
        key.starts_with("policy.") ||
        key == "report.file")
      continue;
    os << runID << "\t" << timestampUTC << "\t" << row.first << "\t"
       << row.second << "\n";
  }
  return true;
}

struct HistorySnapshot {
  uint64_t runID = 0;
  std::string timestampUTC;
  StringMap<std::string> values;
};

static bool loadHistorySnapshots(StringRef path,
                                 std::vector<HistorySnapshot> &snapshots,
                                 std::string &error) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read history file: ") + path).str();
    return false;
  }

  std::map<uint64_t, HistorySnapshot> byRunID;
  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  SmallVector<StringRef, 8> fields;
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.rtrim("\r").trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    splitTSVLine(line, fields);
    if (fields.size() < 4)
      continue;
    StringRef runIDField = fields[0].trim();
    if (runIDField.empty() || runIDField == "run_id")
      continue;
    uint64_t runID = 0;
    if (runIDField.getAsInteger(10, runID)) {
      error = (Twine("circt-mut report: invalid history run_id '") + runIDField +
               "' in " + path)
                  .str();
      return false;
    }
    StringRef timestamp = fields[1].trim();
    StringRef key = fields[2].trim();
    StringRef value = fields[3].trim();
    if (key.empty())
      continue;
    HistorySnapshot &snapshot = byRunID[runID];
    snapshot.runID = runID;
    if (snapshot.timestampUTC.empty() && !timestamp.empty())
      snapshot.timestampUTC = timestamp.str();
    snapshot.values[key] = value.str();
  }

  snapshots.clear();
  snapshots.reserve(byRunID.size());
  for (auto &it : byRunID)
    snapshots.push_back(std::move(it.second));
  if (snapshots.empty()) {
    error = (Twine("circt-mut report: no history snapshots found in: ") + path).str();
    return false;
  }
  return true;
}

static void appendTrendRows(
    ArrayRef<std::pair<std::string, std::string>> currentRows,
    ArrayRef<HistorySnapshot> allSnapshots, StringRef historyPath,
    uint64_t windowRunsRequested,
    std::vector<std::pair<std::string, std::string>> &rows,
    StringMap<double> &trendDeltas) {
  size_t totalRuns = allSnapshots.size();
  size_t startIndex = 0;
  if (windowRunsRequested > 0 && windowRunsRequested < totalRuns)
    startIndex = totalRuns - static_cast<size_t>(windowRunsRequested);
  ArrayRef<HistorySnapshot> selected = allSnapshots.drop_front(startIndex);

  rows.emplace_back("trend.history_file", historyPath.str());
  rows.emplace_back("trend.history_runs_available", std::to_string(totalRuns));
  rows.emplace_back("trend.history_runs_selected",
                    std::to_string(selected.size()));
  rows.emplace_back("trend.history_window_requested",
                    std::to_string(windowRunsRequested));
  if (!selected.empty()) {
    rows.emplace_back("trend.history_run_id_min",
                      std::to_string(selected.front().runID));
    rows.emplace_back("trend.history_run_id_max",
                      std::to_string(selected.back().runID));
  }

  SmallVector<std::pair<std::string, double>, 64> numericCurrentRows;
  for (const auto &row : currentRows) {
    StringRef key = row.first;
    if (key.starts_with("compare.") || key.starts_with("diff.") ||
        key.starts_with("history.") || key.starts_with("trend.") ||
        key.starts_with("policy.") ||
        key == "report.file")
      continue;
    auto currentNum = parseOptionalDouble(row.second);
    if (!currentNum)
      continue;
    numericCurrentRows.emplace_back(key.str(), *currentNum);
  }

  uint64_t numericKeys = 0;
  for (const auto &currentRow : numericCurrentRows) {
    StringRef key = currentRow.first;
    double currentValue = currentRow.second;
    double sum = 0.0;
    double minValue = 0.0;
    double maxValue = 0.0;
    double latestValue = 0.0;
    uint64_t samples = 0;
    for (const auto &snapshot : selected) {
      auto it = snapshot.values.find(key);
      if (it == snapshot.values.end())
        continue;
      auto maybeNum = parseOptionalDouble(it->second);
      if (!maybeNum)
        continue;
      double value = *maybeNum;
      if (samples == 0) {
        minValue = value;
        maxValue = value;
      } else {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
      }
      latestValue = value;
      sum += value;
      ++samples;
    }
    if (samples == 0)
      continue;
    ++numericKeys;
    double mean = sum / static_cast<double>(samples);
    double delta = currentValue - mean;
    trendDeltas[key] = delta;
    rows.emplace_back((Twine("trend.") + key + ".samples").str(),
                      std::to_string(samples));
    rows.emplace_back((Twine("trend.") + key + ".mean").str(), formatDouble2(mean));
    rows.emplace_back((Twine("trend.") + key + ".min").str(),
                      formatDouble2(minValue));
    rows.emplace_back((Twine("trend.") + key + ".max").str(),
                      formatDouble2(maxValue));
    rows.emplace_back((Twine("trend.") + key + ".latest").str(),
                      formatDouble2(latestValue));
    rows.emplace_back((Twine("trend.") + key + ".delta_vs_mean").str(),
                      formatDouble2(delta));
    rows.emplace_back((Twine("trend.") + key + ".pct_vs_mean").str(),
                      mean != 0.0 ? formatDouble2((100.0 * delta) / mean)
                                  : std::string("-"));
  }
  rows.emplace_back("trend.numeric_keys", std::to_string(numericKeys));
}

static bool collectCoverReport(StringRef coverWorkDir,
                               std::vector<std::pair<std::string, std::string>> &rows,
                               std::string &error) {
  SmallString<256> metricsPath(coverWorkDir);
  sys::path::append(metricsPath, "metrics.tsv");
  if (!sys::fs::exists(metricsPath)) {
    error = (Twine("circt-mut report: cover metrics file not found: ") +
             metricsPath)
                .str();
    return false;
  }

  StringMap<std::string> metrics;
  if (!parseKeyValueTSV(metricsPath, metrics, error))
    return false;

  rows.emplace_back("cover.work_dir", std::string(coverWorkDir));
  rows.emplace_back("cover.metrics_file", std::string(metricsPath.str()));
  appendMetricRow(rows, "cover", metrics, "total_mutants");
  appendMetricRow(rows, "cover", metrics, "relevant_mutants");
  appendMetricRow(rows, "cover", metrics, "detected_mutants");
  appendMetricRow(rows, "cover", metrics, "propagated_not_detected_mutants");
  appendMetricRow(rows, "cover", metrics, "not_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "not_activated_mutants");
  appendMetricRow(rows, "cover", metrics, "errors");
  appendMetricRow(rows, "cover", metrics, "mutation_coverage_percent");
  appendMetricRow(rows, "cover", metrics, "global_filtered_not_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_lec_unknown_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_bmc_unknown_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_timeout_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_lec_timeout_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_bmc_timeout_mutants");
  appendMetricRow(rows, "cover", metrics, "global_filter_lec_runtime_ns");
  appendMetricRow(rows, "cover", metrics, "global_filter_bmc_runtime_ns");
  appendMetricRow(rows, "cover", metrics, "global_filter_cmd_runtime_ns");
  appendMetricRow(rows, "cover", metrics, "global_filter_lec_runs");
  appendMetricRow(rows, "cover", metrics, "global_filter_bmc_runs");
  appendMetricRow(rows, "cover", metrics, "global_filter_cmd_runs");
  appendMetricRow(rows, "cover", metrics, "chain_lec_unknown_fallbacks");
  appendMetricRow(rows, "cover", metrics,
                  "chain_bmc_resolved_not_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_bmc_resolved_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_bmc_unknown_fallbacks");
  appendMetricRow(rows, "cover", metrics,
                  "chain_lec_resolved_not_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_lec_resolved_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_lec_error_fallbacks");
  appendMetricRow(rows, "cover", metrics, "chain_bmc_error_fallbacks");
  appendMetricRow(rows, "cover", metrics, "chain_consensus_not_propagated_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_consensus_disagreement_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_consensus_error_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_auto_parallel_mutants");
  appendMetricRow(rows, "cover", metrics, "chain_auto_short_circuit_mutants");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_hit_mutants");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_miss_mutants");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_saved_runtime_ns");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_miss_runtime_ns");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_entries");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_pruned_entries");
  appendMetricRow(rows, "cover", metrics, "bmc_orig_cache_pruned_age_entries");
  appendMetricRow(rows, "cover", metrics, "generated_mutations_cache_status");
  appendMetricRow(rows, "cover", metrics, "generated_mutations_cache_hit");
  appendMetricRow(rows, "cover", metrics, "generated_mutations_cache_miss");
  appendMetricRow(rows, "cover", metrics, "generated_mutations_runtime_ns");
  appendMetricRow(rows, "cover", metrics,
                  "generated_mutations_cache_saved_runtime_ns");
  appendMetricRow(rows, "cover", metrics,
                  "generated_mutations_cache_lock_wait_ns");
  appendMetricRow(rows, "cover", metrics,
                  "generated_mutations_cache_lock_contended");
  return true;
}

static bool collectMatrixReport(
    StringRef matrixOutDir, std::vector<std::pair<std::string, std::string>> &rows,
    std::string &error) {
  SmallString<256> resultsPath(matrixOutDir);
  sys::path::append(resultsPath, "results.tsv");
  if (!sys::fs::exists(resultsPath)) {
    error = (Twine("circt-mut report: matrix results file not found: ") +
             resultsPath)
                .str();
    return false;
  }

  auto bufferOrErr = MemoryBuffer::getFile(resultsPath);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read matrix results file: ") +
             resultsPath)
                .str();
    return false;
  }

  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  if (lines.empty()) {
    error = (Twine("circt-mut report: empty matrix results file: ") + resultsPath)
                .str();
    return false;
  }

  SmallVector<StringRef, 32> fields;
  splitTSVLine(lines.front().rtrim("\r"), fields);
  StringMap<size_t> colIndex;
  for (size_t i = 0; i < fields.size(); ++i)
    colIndex[fields[i].trim()] = i;

  auto requireCol = [&](StringRef name, size_t &dst) -> bool {
    auto it = colIndex.find(name);
    if (it == colIndex.end()) {
      error = (Twine("circt-mut report: missing required matrix results column: ") +
               name + " in " + resultsPath)
                  .str();
      return false;
    }
    dst = it->second;
    return true;
  };

  size_t laneIDCol = 0, statusCol = 0, coverageCol = 0, gateCol = 0;
  if (!requireCol("lane_id", laneIDCol) || !requireCol("status", statusCol) ||
      !requireCol("coverage_percent", coverageCol) ||
      !requireCol("gate_status", gateCol))
    return false;
  size_t metricsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("metrics_file"); it != colIndex.end())
    metricsCol = it->second;

  uint64_t lanesTotal = 0;
  uint64_t lanesPass = 0;
  uint64_t lanesFail = 0;
  uint64_t gatePass = 0;
  uint64_t gateFail = 0;
  uint64_t lanesWithMetrics = 0;
  uint64_t lanesMissingMetrics = 0;
  uint64_t invalidMetricValues = 0;
  uint64_t totalMutantsSum = 0;
  uint64_t relevantMutantsSum = 0;
  uint64_t detectedMutantsSum = 0;
  uint64_t propagatedNotDetectedMutantsSum = 0;
  uint64_t notPropagatedMutantsSum = 0;
  uint64_t notActivatedMutantsSum = 0;
  uint64_t errorsSum = 0;
  double coverageSum = 0.0;
  uint64_t coverageCount = 0;
  StringMap<uint64_t> extraMetricSums;
  static constexpr const char *kExtraMetricKeys[] = {
      "global_filtered_not_propagated_mutants",
      "global_filter_lec_unknown_mutants",
      "global_filter_bmc_unknown_mutants",
      "global_filter_timeout_mutants",
      "global_filter_lec_timeout_mutants",
      "global_filter_bmc_timeout_mutants",
      "global_filter_lec_runtime_ns",
      "global_filter_bmc_runtime_ns",
      "global_filter_cmd_runtime_ns",
      "global_filter_lec_runs",
      "global_filter_bmc_runs",
      "global_filter_cmd_runs",
      "chain_lec_unknown_fallbacks",
      "chain_bmc_resolved_not_propagated_mutants",
      "chain_bmc_resolved_propagated_mutants",
      "chain_bmc_unknown_fallbacks",
      "chain_lec_resolved_not_propagated_mutants",
      "chain_lec_resolved_propagated_mutants",
      "chain_lec_error_fallbacks",
      "chain_bmc_error_fallbacks",
      "chain_consensus_not_propagated_mutants",
      "chain_consensus_disagreement_mutants",
      "chain_consensus_error_mutants",
      "chain_auto_parallel_mutants",
      "chain_auto_short_circuit_mutants",
      "bmc_orig_cache_hit_mutants",
      "bmc_orig_cache_miss_mutants",
      "bmc_orig_cache_saved_runtime_ns",
      "bmc_orig_cache_miss_runtime_ns",
      "bmc_orig_cache_entries",
      "bmc_orig_cache_pruned_entries",
      "bmc_orig_cache_pruned_age_entries",
      "generated_mutations_cache_hit",
      "generated_mutations_cache_miss",
      "generated_mutations_runtime_ns",
      "generated_mutations_cache_saved_runtime_ns",
      "generated_mutations_cache_lock_wait_ns",
      "generated_mutations_cache_lock_contended",
  };
  for (const char *key : kExtraMetricKeys)
    extraMetricSums[key] = 0;

  auto addMetric = [&](const StringMap<std::string> &metrics, StringRef key,
                       uint64_t &accumulator) {
    auto it = metrics.find(key);
    if (it == metrics.end() || it->second.empty())
      return;
    uint64_t parsed = 0;
    if (StringRef(it->second).trim().getAsInteger(10, parsed)) {
      ++invalidMetricValues;
      return;
    }
    accumulator += parsed;
  };

  for (size_t lineNo = 1; lineNo < lines.size(); ++lineNo) {
    StringRef line = lines[lineNo].rtrim("\r");
    if (line.trim().empty())
      continue;
    splitTSVLine(line, fields);

    auto getField = [&](size_t idx) -> StringRef {
      if (idx >= fields.size())
        return StringRef();
      return fields[idx].trim();
    };

    ++lanesTotal;
    StringRef laneID = getField(laneIDCol);
    StringRef status = getField(statusCol);
    StringRef gate = getField(gateCol);
    if (status == "PASS")
      ++lanesPass;
    else
      ++lanesFail;
    if (gate == "PASS")
      ++gatePass;
    else if (gate == "FAIL")
      ++gateFail;

    if (auto coverage = parseOptionalDouble(getField(coverageCol))) {
      coverageSum += *coverage;
      ++coverageCount;
    }

    std::string metricsPath;
    if (metricsCol != static_cast<size_t>(-1)) {
      StringRef metricsValue = getField(metricsCol);
      if (!metricsValue.empty() && metricsValue != "-")
        metricsPath = resolveRelativeTo(matrixOutDir, metricsValue);
    }
    if (metricsPath.empty() && !laneID.empty()) {
      SmallString<256> fallbackPath(matrixOutDir);
      sys::path::append(fallbackPath, laneID, "metrics.tsv");
      metricsPath = std::string(fallbackPath.str());
    }
    if (metricsPath.empty() || !sys::fs::exists(metricsPath)) {
      ++lanesMissingMetrics;
      continue;
    }

    StringMap<std::string> metrics;
    if (!parseKeyValueTSV(metricsPath, metrics, error))
      return false;
    ++lanesWithMetrics;
    addMetric(metrics, "total_mutants", totalMutantsSum);
    addMetric(metrics, "relevant_mutants", relevantMutantsSum);
    addMetric(metrics, "detected_mutants", detectedMutantsSum);
    addMetric(metrics, "propagated_not_detected_mutants",
              propagatedNotDetectedMutantsSum);
    addMetric(metrics, "not_propagated_mutants", notPropagatedMutantsSum);
    addMetric(metrics, "not_activated_mutants", notActivatedMutantsSum);
    addMetric(metrics, "errors", errorsSum);
    for (const char *key : kExtraMetricKeys)
      addMetric(metrics, key, extraMetricSums[key]);
  }

  rows.emplace_back("matrix.out_dir", std::string(matrixOutDir));
  rows.emplace_back("matrix.results_file", std::string(resultsPath.str()));
  rows.emplace_back("matrix.lanes_total", std::to_string(lanesTotal));
  rows.emplace_back("matrix.lanes_pass", std::to_string(lanesPass));
  rows.emplace_back("matrix.lanes_fail", std::to_string(lanesFail));
  rows.emplace_back("matrix.gate_pass", std::to_string(gatePass));
  rows.emplace_back("matrix.gate_fail", std::to_string(gateFail));
  rows.emplace_back("matrix.lanes_with_metrics", std::to_string(lanesWithMetrics));
  rows.emplace_back("matrix.lanes_missing_metrics",
                    std::to_string(lanesMissingMetrics));
  rows.emplace_back("matrix.invalid_metric_values",
                    std::to_string(invalidMetricValues));
  rows.emplace_back("matrix.total_mutants_sum", std::to_string(totalMutantsSum));
  rows.emplace_back("matrix.relevant_mutants_sum",
                    std::to_string(relevantMutantsSum));
  rows.emplace_back("matrix.detected_mutants_sum",
                    std::to_string(detectedMutantsSum));
  rows.emplace_back(
      "matrix.propagated_not_detected_mutants_sum",
      std::to_string(propagatedNotDetectedMutantsSum));
  rows.emplace_back("matrix.not_propagated_mutants_sum",
                    std::to_string(notPropagatedMutantsSum));
  rows.emplace_back("matrix.not_activated_mutants_sum",
                    std::to_string(notActivatedMutantsSum));
  rows.emplace_back("matrix.errors_sum", std::to_string(errorsSum));
  rows.emplace_back(
      "matrix.coverage_percent_avg",
      coverageCount ? formatDouble2(coverageSum / static_cast<double>(coverageCount))
                    : std::string("-"));
  rows.emplace_back(
      "matrix.coverage_percent_from_sums",
      relevantMutantsSum
          ? formatDouble2((100.0 * static_cast<double>(detectedMutantsSum)) /
                          static_cast<double>(relevantMutantsSum))
          : std::string("-"));
  for (const char *key : kExtraMetricKeys)
    rows.emplace_back((Twine("matrix.") + key + "_sum").str(),
                      std::to_string(extraMetricSums[key]));
  return true;
}

static bool writeReportFile(
    StringRef path, ArrayRef<std::pair<std::string, std::string>> rows,
    std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (!parent.empty()) {
    std::error_code dirEC = sys::fs::create_directories(parent);
    if (dirEC) {
      error = (Twine("circt-mut report: failed to create output directory: ") +
               parent + ": " + dirEC.message())
                  .str();
      return false;
    }
  }

  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_Text);
  if (ec) {
    error = (Twine("circt-mut report: failed to open --out file: ") + path +
             ": " + ec.message())
                .str();
    return false;
  }
  os << "key\tvalue\n";
  for (const auto &row : rows)
    os << row.first << "\t" << row.second << "\n";
  return true;
}

static ReportParseResult parseReportArgs(ArrayRef<StringRef> args) {
  ReportParseResult result;

  auto consumeValue = [&](size_t &index, StringRef arg,
                          StringRef optName) -> std::optional<StringRef> {
    size_t eqPos = arg.find('=');
    if (eqPos != StringRef::npos) {
      StringRef value = arg.substr(eqPos + 1);
      if (value.empty()) {
        result.error =
            (Twine("circt-mut report: missing value for ") + optName).str();
        return std::nullopt;
      }
      return value;
    }
    if (index + 1 >= args.size()) {
      result.error =
          (Twine("circt-mut report: missing value for ") + optName).str();
      return std::nullopt;
    }
    return args[++index];
  };

  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == "-h" || arg == "--help") {
      result.showHelp = true;
      result.ok = true;
      return result;
    }
    if (arg == "--project-dir" || arg.starts_with("--project-dir=")) {
      auto v = consumeValue(i, arg, "--project-dir");
      if (!v)
        return result;
      result.opts.projectDir = v->str();
      continue;
    }
    if (arg == "--config" || arg.starts_with("--config=")) {
      auto v = consumeValue(i, arg, "--config");
      if (!v)
        return result;
      result.opts.configPath = v->str();
      result.opts.configExplicit = true;
      continue;
    }
    if (arg == "--mode" || arg.starts_with("--mode=")) {
      auto v = consumeValue(i, arg, "--mode");
      if (!v)
        return result;
      result.opts.mode = v->str();
      continue;
    }
    if (arg == "--cover-work-dir" || arg.starts_with("--cover-work-dir=")) {
      auto v = consumeValue(i, arg, "--cover-work-dir");
      if (!v)
        return result;
      result.opts.coverWorkDir = v->str();
      continue;
    }
    if (arg == "--matrix-out-dir" || arg.starts_with("--matrix-out-dir=")) {
      auto v = consumeValue(i, arg, "--matrix-out-dir");
      if (!v)
        return result;
      result.opts.matrixOutDir = v->str();
      continue;
    }
    if (arg == "--compare" || arg.starts_with("--compare=")) {
      auto v = consumeValue(i, arg, "--compare");
      if (!v)
        return result;
      result.opts.compareFile = v->str();
      continue;
    }
    if (arg == "--compare-history-latest" ||
        arg.starts_with("--compare-history-latest=")) {
      auto v = consumeValue(i, arg, "--compare-history-latest");
      if (!v)
        return result;
      result.opts.compareHistoryLatestFile = v->str();
      continue;
    }
    if (arg == "--trend-history" || arg.starts_with("--trend-history=")) {
      auto v = consumeValue(i, arg, "--trend-history");
      if (!v)
        return result;
      result.opts.trendHistoryFile = v->str();
      continue;
    }
    if (arg == "--trend-window" || arg.starts_with("--trend-window=")) {
      auto v = consumeValue(i, arg, "--trend-window");
      if (!v)
        return result;
      uint64_t parsed = 0;
      if (StringRef(*v).trim().getAsInteger(10, parsed)) {
        result.error =
            (Twine("circt-mut report: invalid --trend-window value: ") + *v +
             " (expected non-negative integer)")
                .str();
        return result;
      }
      result.opts.trendWindowRuns = parsed;
      continue;
    }
    if (arg == "--policy-profile" || arg.starts_with("--policy-profile=")) {
      auto v = consumeValue(i, arg, "--policy-profile");
      if (!v)
        return result;
      result.opts.policyProfiles.push_back(v->str());
      continue;
    }
    if (arg == "--append-history" || arg.starts_with("--append-history=")) {
      auto v = consumeValue(i, arg, "--append-history");
      if (!v)
        return result;
      result.opts.appendHistoryFile = v->str();
      continue;
    }
    if (arg == "--fail-if-delta-gt" || arg.starts_with("--fail-if-delta-gt=")) {
      auto v = consumeValue(i, arg, "--fail-if-delta-gt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-delta-gt", rule, result.error))
        return result;
      result.opts.failIfDeltaGtRules.push_back(rule);
      continue;
    }
    if (arg == "--fail-if-delta-lt" || arg.starts_with("--fail-if-delta-lt=")) {
      auto v = consumeValue(i, arg, "--fail-if-delta-lt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-delta-lt", rule, result.error))
        return result;
      result.opts.failIfDeltaLtRules.push_back(rule);
      continue;
    }
    if (arg == "--fail-if-trend-delta-gt" ||
        arg.starts_with("--fail-if-trend-delta-gt=")) {
      auto v = consumeValue(i, arg, "--fail-if-trend-delta-gt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-trend-delta-gt", rule,
                              result.error))
        return result;
      result.opts.failIfTrendDeltaGtRules.push_back(rule);
      continue;
    }
    if (arg == "--fail-if-trend-delta-lt" ||
        arg.starts_with("--fail-if-trend-delta-lt=")) {
      auto v = consumeValue(i, arg, "--fail-if-trend-delta-lt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-trend-delta-lt", rule,
                              result.error))
        return result;
      result.opts.failIfTrendDeltaLtRules.push_back(rule);
      continue;
    }
    if (arg == "--out" || arg.starts_with("--out=")) {
      auto v = consumeValue(i, arg, "--out");
      if (!v)
        return result;
      result.opts.outFile = v->str();
      continue;
    }

    if (arg.starts_with("-")) {
      result.error = (Twine("circt-mut report: unknown option: ") + arg).str();
      return result;
    }
    result.error = (Twine("circt-mut report: unexpected positional argument: ") +
                    arg)
                       .str();
    return result;
  }

  if (result.opts.mode != "cover" && result.opts.mode != "matrix" &&
      result.opts.mode != "all") {
    result.error =
        (Twine("circt-mut report: invalid --mode value: ") + result.opts.mode +
         " (expected cover|matrix|all)")
            .str();
    return result;
  }
  if (!result.opts.compareFile.empty() &&
      !result.opts.compareHistoryLatestFile.empty()) {
    result.error = "circt-mut report: --compare and --compare-history-latest "
                   "are mutually exclusive";
    return result;
  }
  if (!result.opts.policyProfiles.empty()) {
    SmallVector<std::string, 4> uniqueProfiles;
    StringSet<> seen;
    for (const auto &profile : result.opts.policyProfiles) {
      if (!seen.insert(profile).second)
        continue;
      uniqueProfiles.push_back(profile);
      if (!applyPolicyProfile(profile, result.opts, result.error))
        return result;
    }
    result.opts.policyProfiles = std::move(uniqueProfiles);
  }

  result.ok = true;
  return result;
}

static int runNativeReport(const ReportOptions &opts) {
  SmallString<256> defaultCover(opts.projectDir);
  sys::path::append(defaultCover, "out", "cover");
  SmallString<256> defaultMatrix(opts.projectDir);
  sys::path::append(defaultMatrix, "out", "matrix");

  std::string coverWorkDir = std::string(defaultCover.str());
  std::string matrixOutDir = std::string(defaultMatrix.str());

  SmallString<256> configPath;
  if (opts.configPath.empty()) {
    configPath = opts.projectDir;
    sys::path::append(configPath, "circt-mut.toml");
  } else if (sys::path::is_absolute(opts.configPath)) {
    configPath = opts.configPath;
  } else {
    configPath = opts.projectDir;
    sys::path::append(configPath, opts.configPath);
  }

  bool haveConfig = sys::fs::exists(configPath);
  if (opts.configExplicit && !haveConfig) {
    errs() << "circt-mut report: config file not found: " << configPath << "\n";
    return 1;
  }

  if (haveConfig) {
    RunConfigValues cfg;
    std::string error;
    if (!parseRunConfigFile(configPath, cfg, error)) {
      errs() << rewriteErrorPrefix(error, "circt-mut run:", "circt-mut report:")
             << "\n";
      return 1;
    }
    if (auto it = cfg.cover.find("work_dir");
        it != cfg.cover.end() && !it->second.empty())
      coverWorkDir = resolveRelativeTo(opts.projectDir, it->second);
    if (auto it = cfg.matrix.find("out_dir");
        it != cfg.matrix.end() && !it->second.empty())
      matrixOutDir = resolveRelativeTo(opts.projectDir, it->second);
  }

  if (!opts.coverWorkDir.empty())
    coverWorkDir = resolveRelativeTo(opts.projectDir, opts.coverWorkDir);
  if (!opts.matrixOutDir.empty())
    matrixOutDir = resolveRelativeTo(opts.projectDir, opts.matrixOutDir);

  std::vector<std::pair<std::string, std::string>> rows;
  rows.emplace_back("report.mode", opts.mode);
  if (!opts.policyProfiles.empty()) {
    rows.emplace_back("policy.profile_count",
                      std::to_string(opts.policyProfiles.size()));
    for (size_t i = 0; i < opts.policyProfiles.size(); ++i)
      rows.emplace_back((Twine("policy.profile_") + Twine(i + 1)).str(),
                        opts.policyProfiles[i]);
  }
  int finalRC = 0;
  std::string error;
  if (opts.mode == "cover" || opts.mode == "all") {
    if (!collectCoverReport(coverWorkDir, rows, error)) {
      errs() << error << "\n";
      return 1;
    }
  }
  if (opts.mode == "matrix" || opts.mode == "all") {
    if (!collectMatrixReport(matrixOutDir, rows, error)) {
      errs() << error << "\n";
      return 1;
    }
  }

  if (opts.compareFile.empty() && opts.compareHistoryLatestFile.empty() &&
      (!opts.failIfDeltaGtRules.empty() || !opts.failIfDeltaLtRules.empty())) {
    errs() << "circt-mut report: --fail-if-delta-gt/--fail-if-delta-lt "
              "require --compare or --compare-history-latest\n";
    return 1;
  }
  if (opts.trendHistoryFile.empty() &&
      (!opts.failIfTrendDeltaGtRules.empty() ||
       !opts.failIfTrendDeltaLtRules.empty())) {
    errs() << "circt-mut report: --fail-if-trend-delta-gt/"
              "--fail-if-trend-delta-lt require --trend-history\n";
    return 1;
  }

  StringMap<double> numericDeltas;
  StringMap<double> trendDeltas;
  if (!opts.compareFile.empty()) {
    std::string baselinePath = resolveRelativeTo(opts.projectDir, opts.compareFile);
    if (!sys::fs::exists(baselinePath)) {
      errs() << "circt-mut report: compare baseline file not found: "
             << baselinePath << "\n";
      return 1;
    }
    if (!appendReportComparison(rows, baselinePath, rows, numericDeltas, error)) {
      errs() << error << "\n";
      return 1;
    }
  }
  if (!opts.compareHistoryLatestFile.empty()) {
    std::string historyPath =
        resolveRelativeTo(opts.projectDir, opts.compareHistoryLatestFile);
    if (!sys::fs::exists(historyPath)) {
      errs() << "circt-mut report: compare history file not found: "
             << historyPath << "\n";
      return 1;
    }
    StringMap<std::string> baselineValues;
    uint64_t baselineRunID = 0;
    if (!loadLatestHistorySnapshot(historyPath, baselineRunID, baselineValues,
                                   error)) {
      errs() << error << "\n";
      return 1;
    }
    std::string baselineLabel =
        (Twine(historyPath) + "#run_id=" + Twine(baselineRunID)).str();
    appendReportComparisonRows(rows, baselineValues, baselineLabel, rows,
                               numericDeltas);
    rows.emplace_back("compare.history_baseline_run_id",
                      std::to_string(baselineRunID));
  }
  if (!opts.trendHistoryFile.empty()) {
    std::string historyPath = resolveRelativeTo(opts.projectDir, opts.trendHistoryFile);
    if (!sys::fs::exists(historyPath)) {
      errs() << "circt-mut report: trend history file not found: " << historyPath
             << "\n";
      return 1;
    }
    std::vector<HistorySnapshot> snapshots;
    if (!loadHistorySnapshots(historyPath, snapshots, error)) {
      errs() << error << "\n";
      return 1;
    }
    appendTrendRows(rows, snapshots, historyPath, opts.trendWindowRuns, rows,
                    trendDeltas);
  }

  if (!opts.failIfDeltaGtRules.empty() || !opts.failIfDeltaLtRules.empty()) {
    SmallVector<std::string, 8> failures;
    auto evaluateRule = [&](const DeltaGateRule &rule, bool isUpperBound) -> bool {
      auto it = numericDeltas.find(rule.key);
      if (it == numericDeltas.end()) {
        errs() << "circt-mut report: numeric delta missing for gate rule key '"
               << rule.key
               << "' (run with --compare/--compare-history-latest and numeric baseline values)\n";
        return false;
      }
      double delta = it->second;
      if ((isUpperBound && delta > rule.threshold) ||
          (!isUpperBound && delta < rule.threshold)) {
        failures.push_back((Twine(rule.key) + " delta=" + formatDouble2(delta) +
                            (isUpperBound ? " > " : " < ") +
                            formatDouble2(rule.threshold))
                               .str());
      }
      return true;
    };

    for (const auto &rule : opts.failIfDeltaGtRules)
      if (!evaluateRule(rule, /*isUpperBound=*/true))
        return 1;
    for (const auto &rule : opts.failIfDeltaLtRules)
      if (!evaluateRule(rule, /*isUpperBound=*/false))
        return 1;

    rows.emplace_back("compare.gate_rules_total",
                      std::to_string(opts.failIfDeltaGtRules.size() +
                                     opts.failIfDeltaLtRules.size()));
    rows.emplace_back("compare.gate_failure_count",
                      std::to_string(failures.size()));
    rows.emplace_back("compare.gate_status", failures.empty() ? "pass" : "fail");
    for (size_t i = 0; i < failures.size(); ++i)
      rows.emplace_back((Twine("compare.gate_failure_") + Twine(i + 1)).str(),
                        failures[i]);
    if (!failures.empty())
      finalRC = 2;
  }
  if (!opts.failIfTrendDeltaGtRules.empty() ||
      !opts.failIfTrendDeltaLtRules.empty()) {
    SmallVector<std::string, 8> failures;
    auto evaluateTrendRule = [&](const DeltaGateRule &rule,
                                 bool isUpperBound) -> bool {
      auto it = trendDeltas.find(rule.key);
      if (it == trendDeltas.end()) {
        errs() << "circt-mut report: trend delta missing for gate rule key '"
               << rule.key
               << "' (run with --trend-history and numeric history values)\n";
        return false;
      }
      double delta = it->second;
      if ((isUpperBound && delta > rule.threshold) ||
          (!isUpperBound && delta < rule.threshold)) {
        failures.push_back((Twine(rule.key) + " trend_delta=" +
                            formatDouble2(delta) +
                            (isUpperBound ? " > " : " < ") +
                            formatDouble2(rule.threshold))
                               .str());
      }
      return true;
    };

    for (const auto &rule : opts.failIfTrendDeltaGtRules)
      if (!evaluateTrendRule(rule, /*isUpperBound=*/true))
        return 1;
    for (const auto &rule : opts.failIfTrendDeltaLtRules)
      if (!evaluateTrendRule(rule, /*isUpperBound=*/false))
        return 1;

    rows.emplace_back("trend.gate_rules_total",
                      std::to_string(opts.failIfTrendDeltaGtRules.size() +
                                     opts.failIfTrendDeltaLtRules.size()));
    rows.emplace_back("trend.gate_failure_count",
                      std::to_string(failures.size()));
    rows.emplace_back("trend.gate_status", failures.empty() ? "pass" : "fail");
    for (size_t i = 0; i < failures.size(); ++i)
      rows.emplace_back((Twine("trend.gate_failure_") + Twine(i + 1)).str(),
                        failures[i]);
    if (!failures.empty())
      finalRC = 2;
  }

  if (!opts.appendHistoryFile.empty()) {
    std::string historyPath =
        resolveRelativeTo(opts.projectDir, opts.appendHistoryFile);
    uint64_t maxRunID = 0;
    if (!readHistoryMaxRunID(historyPath, maxRunID, error)) {
      errs() << error << "\n";
      return 1;
    }
    uint64_t nextRunID = maxRunID + 1;
    std::string timestampUTC = formatCurrentUTCISO8601();
    if (!appendHistorySnapshot(historyPath, nextRunID, timestampUTC, rows,
                               error)) {
      errs() << error << "\n";
      return 1;
    }
    rows.emplace_back("history.file", historyPath);
    rows.emplace_back("history.appended_run_id", std::to_string(nextRunID));
    rows.emplace_back("history.appended_timestamp_utc", timestampUTC);
  }

  std::string outFile;
  if (!opts.outFile.empty())
    outFile = resolveRelativeTo(opts.projectDir, opts.outFile);
  if (!outFile.empty()) {
    if (!writeReportFile(outFile, rows, error)) {
      errs() << error << "\n";
      return 1;
    }
    rows.emplace_back("report.file", outFile);
  }

  outs() << "key\tvalue\n";
  for (const auto &row : rows)
    outs() << row.first << "\t" << row.second << "\n";
  return finalRC;
}

static void splitCSV(StringRef csv, SmallVectorImpl<std::string> &out) {
  SmallVector<StringRef, 8> parts;
  csv.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef part : parts) {
    part = part.trim();
    if (!part.empty())
      out.push_back(part.str());
  }
}

static bool parsePositiveUInt(StringRef value, uint64_t &out) {
  if (value.getAsInteger(10, out))
    return false;
  return out > 0;
}

static bool parseUInt(StringRef value, uint64_t &out) {
  return !value.getAsInteger(10, out);
}

static std::string quoteForYosys(StringRef value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char c : value) {
    if (c == '\\' || c == '"')
      out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

struct GenerateOptions {
  std::string design;
  std::string outFile;
  std::string cacheDir;
  std::string top;
  uint64_t count = 1000;
  uint64_t seed = 1;
  std::string yosys = "yosys";
  SmallVector<std::string, 8> modeList;
  SmallVector<std::string, 8> modeCountList;
  SmallVector<std::string, 8> modeWeightList;
  SmallVector<std::string, 8> profileList;
  SmallVector<std::string, 8> cfgList;
  SmallVector<std::string, 8> selectList;
};

struct GenerateParseResult {
  bool ok = false;
  bool showHelp = false;
  bool fallbackToScript = false;
  std::string error;
  GenerateOptions opts;
};

static GenerateParseResult parseGenerateArgs(ArrayRef<StringRef> args) {
  GenerateParseResult result;

  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    auto requireValue = [&](StringRef opt) -> std::optional<StringRef> {
      if (i + 1 >= args.size()) {
        result.error = (Twine("circt-mut generate: missing value for ") + opt).str();
        return std::nullopt;
      }
      return args[++i];
    };

    if (arg == "-h" || arg == "--help") {
      result.showHelp = true;
      result.ok = true;
      return result;
    }
    if (arg == "--design") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      result.opts.design = v->str();
      continue;
    }
    if (arg == "--out") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      result.opts.outFile = v->str();
      continue;
    }
    if (arg == "--top") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      result.opts.top = v->str();
      continue;
    }
    if (arg == "--count") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      if (!parsePositiveUInt(*v, result.opts.count)) {
        result.error = (Twine("circt-mut generate: invalid --count value: ") + *v).str();
        return result;
      }
      continue;
    }
    if (arg == "--seed") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      if (!parseUInt(*v, result.opts.seed)) {
        result.error = (Twine("circt-mut generate: invalid --seed value: ") + *v).str();
        return result;
      }
      continue;
    }
    if (arg == "--yosys") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      result.opts.yosys = v->str();
      continue;
    }
    if (arg == "--mode") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef mode = v->trim();
      if (!mode.empty())
        result.opts.modeList.push_back(mode.str());
      continue;
    }
    if (arg == "--modes") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.modeList);
      continue;
    }
    if (arg == "--mode-count") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef entry = v->trim();
      if (!entry.empty())
        result.opts.modeCountList.push_back(entry.str());
      continue;
    }
    if (arg == "--mode-counts") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.modeCountList);
      continue;
    }
    if (arg == "--mode-weight") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef entry = v->trim();
      if (!entry.empty())
        result.opts.modeWeightList.push_back(entry.str());
      continue;
    }
    if (arg == "--mode-weights") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.modeWeightList);
      continue;
    }
    if (arg == "--profile") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef entry = v->trim();
      if (!entry.empty())
        result.opts.profileList.push_back(entry.str());
      continue;
    }
    if (arg == "--profiles") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.profileList);
      continue;
    }
    if (arg == "--cfg") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef entry = v->trim();
      if (!entry.empty())
        result.opts.cfgList.push_back(entry.str());
      continue;
    }
    if (arg == "--cfgs") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.cfgList);
      continue;
    }
    if (arg == "--select") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      StringRef entry = v->trim();
      if (!entry.empty())
        result.opts.selectList.push_back(entry.str());
      continue;
    }
    if (arg == "--selects") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      splitCSV(*v, result.opts.selectList);
      continue;
    }

    if (arg == "--cache-dir") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      result.opts.cacheDir = v->str();
      continue;
    }
    // Keep full compatibility by deferring unsupported/unknown options to the
    // script backend while native migration is in progress.
    if (arg.starts_with("-")) {
      result.fallbackToScript = true;
      result.ok = true;
      return result;
    }

    result.error = (Twine("circt-mut generate: unexpected positional argument: ") + arg).str();
    return result;
  }

  if (result.opts.design.empty() || result.opts.outFile.empty()) {
    result.error = "circt-mut generate: missing required arguments --design and --out";
    return result;
  }

  result.ok = true;
  return result;
}

static void modeFamilyTargets(StringRef modeName,
                              SmallVectorImpl<std::string> &out) {
  if (modeName == "arith") {
    out.push_back("inv");
    out.push_back("const0");
    out.push_back("const1");
    return;
  }
  if (modeName == "control") {
    out.push_back("cnot0");
    out.push_back("cnot1");
    return;
  }
  if (modeName == "stuck") {
    out.push_back("const0");
    out.push_back("const1");
    return;
  }
  if (modeName == "invert") {
    out.push_back("inv");
    return;
  }
  if (modeName == "connect") {
    out.push_back("cnot0");
    out.push_back("cnot1");
    return;
  }
  if (modeName == "balanced" || modeName == "all") {
    out.push_back("inv");
    out.push_back("const0");
    out.push_back("const1");
    out.push_back("cnot0");
    out.push_back("cnot1");
    return;
  }
  out.push_back(modeName.str());
}

static bool isIndexInRotatedExtraPrefix(uint64_t index, uint64_t start,
                                        uint64_t extraCount,
                                        uint64_t totalCount) {
  if (extraCount == 0 || totalCount == 0)
    return false;
  if (extraCount >= totalCount)
    return true;
  start %= totalCount;
  uint64_t distance = (index + totalCount - start) % totalCount;
  return distance < extraCount;
}

static bool appendProfile(StringRef profileName,
                          SmallVectorImpl<std::string> &profileModes,
                          SmallVectorImpl<std::string> &profileCfgs,
                          std::string &error) {
  if (profileName == "arith-depth") {
    profileModes.push_back("arith");
    profileCfgs.push_back("weight_pq_w=5");
    profileCfgs.push_back("weight_pq_mw=5");
    profileCfgs.push_back("weight_pq_b=3");
    profileCfgs.push_back("weight_pq_mb=3");
    return true;
  }
  if (profileName == "control-depth") {
    profileModes.push_back("control");
    profileCfgs.push_back("weight_pq_c=5");
    profileCfgs.push_back("weight_pq_mc=5");
    profileCfgs.push_back("weight_pq_s=3");
    profileCfgs.push_back("weight_pq_ms=3");
    return true;
  }
  if (profileName == "balanced-depth") {
    profileModes.push_back("arith");
    profileModes.push_back("control");
    profileCfgs.push_back("weight_pq_w=4");
    profileCfgs.push_back("weight_pq_mw=4");
    profileCfgs.push_back("weight_pq_c=4");
    profileCfgs.push_back("weight_pq_mc=4");
    profileCfgs.push_back("weight_pq_s=2");
    profileCfgs.push_back("weight_pq_ms=2");
    return true;
  }
  if (profileName == "fault-basic") {
    profileModes.push_back("stuck");
    profileModes.push_back("invert");
    profileModes.push_back("connect");
    profileCfgs.push_back("weight_cover=5");
    profileCfgs.push_back("pick_cover_prcnt=80");
    return true;
  }
  if (profileName == "fault-stuck") {
    profileModes.push_back("stuck");
    profileModes.push_back("invert");
    profileCfgs.push_back("weight_cover=4");
    profileCfgs.push_back("pick_cover_prcnt=70");
    return true;
  }
  if (profileName == "fault-connect") {
    profileModes.push_back("connect");
    profileModes.push_back("invert");
    profileCfgs.push_back("weight_cover=4");
    profileCfgs.push_back("pick_cover_prcnt=70");
    return true;
  }
  if (profileName == "cover") {
    profileCfgs.push_back("weight_cover=5");
    profileCfgs.push_back("pick_cover_prcnt=80");
    return true;
  }
  if (profileName == "none")
    return true;

  error = (Twine("circt-mut generate: unknown --profile value: ") + profileName +
           " (expected arith-depth|control-depth|balanced-depth|fault-basic|"
           "fault-stuck|fault-connect|cover|none)")
              .str();
  return false;
}

class ScopedTempDir {
public:
  explicit ScopedTempDir(StringRef path) : path(path.str()) {}
  ~ScopedTempDir() {
    if (!path.empty())
      (void)sys::fs::remove_directories(path, /*IgnoreErrors=*/true);
  }

private:
  std::string path;
};

class ScopedCacheLock {
public:
  ~ScopedCacheLock() { release(); }

  void setLockDir(StringRef dir) { lockDir = dir.str(); }
  StringRef getLockDir() const { return lockDir; }
  void setHeld(bool value) { held = value; }
  bool isHeld() const { return held; }

  void release() {
    if (!held || lockDir.empty())
      return;
    std::error_code ec;
    SmallString<256> pidFile(lockDir);
    sys::path::append(pidFile, "pid");
    (void)sys::fs::remove(pidFile);
    (void)sys::fs::remove(lockDir);
    held = false;
    lockDir.clear();
  }

private:
  std::string lockDir;
  bool held = false;
};

static std::string hashSHA256(StringRef input) {
  SHA256 sha;
  sha.update(input);
  auto digest = sha.final();
  return toHex(ArrayRef<uint8_t>(digest), /*LowerCase=*/true);
}

static std::optional<std::string> hashFileSHA256(StringRef path) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return std::nullopt;
  return hashSHA256(bufferOrErr.get()->getBuffer());
}

static std::string joinWithTrailingNewline(ArrayRef<std::string> items) {
  std::string out;
  if (items.empty()) {
    out.push_back('\n');
    return out;
  }
  for (const std::string &item : items) {
    out += item;
    out.push_back('\n');
  }
  return out;
}

static uint64_t parseGenerationRuntimeFromMeta(StringRef path) {
  constexpr StringRef key = "generation_runtime_ns\t";
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return 0;
  SmallVector<StringRef, 16> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    if (!line.starts_with(key))
      continue;
    uint64_t value = 0;
    if (!line.substr(key.size()).getAsInteger(10, value))
      return value;
  }
  return 0;
}

static std::error_code copyFileReplace(StringRef from, StringRef to) {
  (void)sys::fs::remove(to);
  return sys::fs::copy_file(from, to);
}

static uint64_t countNonEmptyLines(StringRef path) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return 0;
  uint64_t count = 0;
  SmallVector<StringRef, 64> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    if (!line.trim().empty())
      ++count;
  }
  return count;
}

static int runNativeGenerate(const GenerateOptions &opts) {
  auto start = std::chrono::steady_clock::now();
  auto elapsedNs = [&]() -> uint64_t {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start)
        .count();
  };

  if (!sys::fs::exists(opts.design)) {
    errs() << "circt-mut generate: design file not found: " << opts.design << "\n";
    return 1;
  }
  auto yosysResolved = resolveToolPath(opts.yosys);
  if (!yosysResolved) {
    errs() << "circt-mut generate: unable to resolve --yosys executable: "
           << opts.yosys << "\n";
    return 1;
  }
  std::string yosysExec = *yosysResolved;

  SmallVector<std::string, 8> profileModes;
  SmallVector<std::string, 8> profileCfgs;
  for (const std::string &profile : opts.profileList) {
    std::string profileErr;
    if (!appendProfile(profile, profileModes, profileCfgs, profileErr)) {
      errs() << profileErr << "\n";
      return 1;
    }
  }

  bool modeCountsEnabled = false;
  uint64_t modeCountsTotal = 0;
  SmallVector<std::string, 8> modeCountKeys;
  StringMap<uint64_t> modeCountByMode;
  for (const std::string &entry : opts.modeCountList) {
    StringRef ref(entry);
    auto split = ref.split('=');
    StringRef modeName = split.first.trim();
    StringRef countRef = split.second.trim();
    if (modeName.empty() || countRef == split.first) {
      errs() << "circt-mut generate: invalid --mode-count entry: " << entry
             << " (expected NAME=COUNT)\n";
      return 1;
    }
    uint64_t modeCountValue = 0;
    if (!parsePositiveUInt(countRef, modeCountValue)) {
      errs() << "circt-mut generate: invalid --mode-count count for " << modeName
             << ": " << countRef << " (expected positive integer)\n";
      return 1;
    }
    if (!modeCountByMode.count(modeName))
      modeCountKeys.push_back(modeName.str());
    modeCountByMode[modeName] += modeCountValue;
    modeCountsTotal += modeCountValue;
    modeCountsEnabled = true;
  }
  if (modeCountsEnabled && modeCountsTotal != opts.count) {
    errs() << "circt-mut generate: mode-count total (" << modeCountsTotal
           << ") must match --count (" << opts.count << ")\n";
    return 1;
  }
  bool modeWeightsEnabled = false;
  uint64_t modeWeightsTotal = 0;
  SmallVector<std::string, 8> modeWeightKeys;
  StringMap<uint64_t> modeWeightByMode;
  for (const std::string &entry : opts.modeWeightList) {
    StringRef ref(entry);
    auto split = ref.split('=');
    StringRef modeName = split.first.trim();
    StringRef weightRef = split.second.trim();
    if (modeName.empty() || weightRef == split.first) {
      errs() << "circt-mut generate: invalid --mode-weight entry: " << entry
             << " (expected NAME=WEIGHT)\n";
      return 1;
    }
    uint64_t modeWeightValue = 0;
    if (!parsePositiveUInt(weightRef, modeWeightValue)) {
      errs() << "circt-mut generate: invalid --mode-weight weight for "
             << modeName << ": " << weightRef
             << " (expected positive integer)\n";
      return 1;
    }
    if (!modeWeightByMode.count(modeName))
      modeWeightKeys.push_back(modeName.str());
    modeWeightByMode[modeName] += modeWeightValue;
    modeWeightsTotal += modeWeightValue;
    modeWeightsEnabled = true;
  }
  if (modeCountsEnabled && modeWeightsEnabled) {
    errs() << "circt-mut generate: use either --mode-count(s) or "
              "--mode-weight(s), not both\n";
    return 1;
  }
  if (modeWeightsEnabled) {
    if (modeWeightsTotal == 0) {
      errs() << "circt-mut generate: mode-weight total must be positive\n";
      return 1;
    }
    modeCountsEnabled = true;
    modeCountsTotal = 0;
    modeCountKeys = modeWeightKeys;
    for (const std::string &modeName : modeWeightKeys) {
      uint64_t modeWeightValue = modeWeightByMode[modeName];
      uint64_t modeCountValue = (opts.count * modeWeightValue) / modeWeightsTotal;
      modeCountByMode[modeName] = modeCountValue;
      modeCountsTotal += modeCountValue;
    }
    uint64_t modeRemainder = opts.count - modeCountsTotal;
    if (modeRemainder > 0 && !modeWeightKeys.empty()) {
      uint64_t start = opts.seed % modeWeightKeys.size();
      for (uint64_t i = 0; i < modeRemainder; ++i) {
        const std::string &modeName =
            modeWeightKeys[(start + i) % modeWeightKeys.size()];
        modeCountByMode[modeName] += 1;
      }
    }
  }

  SmallVector<std::string, 16> combinedModes;
  combinedModes.append(profileModes.begin(), profileModes.end());
  combinedModes.append(opts.modeList.begin(), opts.modeList.end());
  if (modeCountsEnabled)
    combinedModes.append(modeCountKeys.begin(), modeCountKeys.end());

  StringSet<> seenModes;
  SmallVector<std::string, 16> finalModes;
  for (const std::string &mode : combinedModes) {
    StringRef m(mode);
    m = m.trim();
    if (m.empty())
      continue;
    if (seenModes.insert(m).second)
      finalModes.push_back(m.str());
  }
  if (finalModes.empty())
    finalModes.push_back(std::string());

  SmallVector<std::string, 16> combinedCfgs;
  combinedCfgs.append(profileCfgs.begin(), profileCfgs.end());
  combinedCfgs.append(opts.cfgList.begin(), opts.cfgList.end());

  SmallVector<std::string, 16> cfgKeyOrder;
  StringMap<std::string> cfgByKey;
  for (const std::string &cfgEntry : combinedCfgs) {
    StringRef ref(cfgEntry);
    auto split = ref.split('=');
    StringRef key = split.first.trim();
    StringRef value = split.second.trim();
    if (key.empty() || value == split.first) {
      errs() << "circt-mut generate: invalid --cfg entry: " << cfgEntry
             << " (expected KEY=VALUE)\n";
      return 1;
    }
    int64_t numericValue = 0;
    if (value.getAsInteger(10, numericValue)) {
      errs() << "circt-mut generate: invalid --cfg value for " << key << ": "
             << value << " (expected integer)\n";
      return 1;
    }
    if (!cfgByKey.count(key))
      cfgKeyOrder.push_back(key.str());
    cfgByKey[key] = std::to_string(numericValue);
  }

  SmallVector<std::pair<std::string, std::string>, 16> finalCfgList;
  for (const std::string &key : cfgKeyOrder)
    finalCfgList.push_back({key, cfgByKey[key]});

  StringSet<> seenSelects;
  SmallVector<std::string, 16> finalSelects;
  for (const std::string &sel : opts.selectList) {
    StringRef select = StringRef(sel).trim();
    if (select.empty())
      continue;
    if (seenSelects.insert(select).second)
      finalSelects.push_back(select.str());
  }

  SmallVector<std::string, 16> modeTargetList;
  SmallVector<uint64_t, 16> modeTargetCounts;

  uint64_t modeCount = finalModes.size();
  uint64_t baseCount = 0;
  uint64_t extraCount = 0;
  uint64_t modeExtraStart = 0;
  if (!modeCountsEnabled) {
    baseCount = opts.count / modeCount;
    extraCount = opts.count % modeCount;
    modeExtraStart = opts.seed % modeCount;
  }

  for (size_t i = 0; i < finalModes.size(); ++i) {
    uint64_t listCount = 0;
    StringRef mode = finalModes[i];
    if (modeCountsEnabled) {
      auto it = modeCountByMode.find(mode);
      if (it != modeCountByMode.end())
        listCount = it->second;
    } else {
      listCount = baseCount;
      if (isIndexInRotatedExtraPrefix(i, modeExtraStart, extraCount, modeCount))
        ++listCount;
    }
    if (listCount == 0)
      continue;

    SmallVector<std::string, 8> familyTargets;
    modeFamilyTargets(mode, familyTargets);
    uint64_t familyCount = familyTargets.size();
    uint64_t familyBase = listCount / familyCount;
    uint64_t familyExtra = listCount % familyCount;
    uint64_t familyExtraStart = 0;
    if (familyCount > 0)
      familyExtraStart = (opts.seed + i) % familyCount;
    for (size_t j = 0; j < familyTargets.size(); ++j) {
      uint64_t familyListCount = familyBase;
      if (isIndexInRotatedExtraPrefix(j, familyExtraStart, familyExtra,
                                      familyCount))
        ++familyListCount;
      if (familyListCount == 0)
        continue;
      modeTargetList.push_back(familyTargets[j]);
      modeTargetCounts.push_back(familyListCount);
    }
  }

  if (modeTargetList.empty()) {
    errs() << "circt-mut generate: no mutation targets selected after mode expansion\n";
    return 1;
  }

  bool cacheEnabled = !opts.cacheDir.empty();
  std::string cacheFile;
  std::string cacheMetaFile;
  uint64_t cacheSavedRuntimeNs = 0;
  uint64_t cacheLockWaitNs = 0;
  int cacheLockContended = 0;
  ScopedCacheLock cacheLock;

  if (cacheEnabled) {
    std::error_code ec = sys::fs::create_directories(opts.cacheDir);
    if (ec) {
      errs() << "circt-mut generate: failed to create cache directory: "
             << opts.cacheDir << ": " << ec.message() << "\n";
      return 1;
    }

    auto designHash = hashFileSHA256(opts.design);
    if (!designHash) {
      errs() << "circt-mut generate: failed to hash design file: " << opts.design
             << "\n";
      return 1;
    }

    SmallVector<std::string, 16> cfgEntries;
    for (const auto &cfg : finalCfgList)
      cfgEntries.push_back((Twine(cfg.first) + "=" + cfg.second).str());

    std::string modePayload = joinWithTrailingNewline(finalModes);
    std::string modeCountPayload = joinWithTrailingNewline(opts.modeCountList);
    std::string modeWeightPayload = joinWithTrailingNewline(opts.modeWeightList);
    std::string profilePayload = joinWithTrailingNewline(opts.profileList);
    std::string cfgPayload = joinWithTrailingNewline(cfgEntries);
    std::string selectPayload = joinWithTrailingNewline(finalSelects);

    std::string cachePayload;
    raw_string_ostream cachePayloadOS(cachePayload);
    cachePayloadOS << "v1\n";
    cachePayloadOS << "design_hash=" << *designHash << "\n";
    cachePayloadOS << "top=" << opts.top << "\n";
    cachePayloadOS << "count=" << opts.count << "\n";
    cachePayloadOS << "seed=" << opts.seed << "\n";
    cachePayloadOS << "yosys_bin=" << yosysExec << "\n";
    cachePayloadOS << "modes=" << modePayload;
    cachePayloadOS << "mode_counts=" << modeCountPayload;
    cachePayloadOS << "mode_weights=" << modeWeightPayload;
    cachePayloadOS << "profiles=" << profilePayload;
    cachePayloadOS << "cfg=" << cfgPayload;
    cachePayloadOS << "select=" << selectPayload;
    cachePayloadOS.flush();

    std::string cacheKey = hashSHA256(cachePayload);
    cacheFile = (Twine(opts.cacheDir) + "/" + cacheKey + ".mutations.txt").str();
    cacheMetaFile = cacheFile + ".meta";

    auto tryCacheHit = [&](bool afterLock) -> std::optional<int> {
      uint64_t cacheSize = 0;
      if (sys::fs::file_size(cacheFile, cacheSize) || cacheSize == 0)
        return std::nullopt;
      std::error_code copyEc = copyFileReplace(cacheFile, opts.outFile);
      if (copyEc) {
        errs() << "circt-mut generate: failed to copy cache file: " << cacheFile
               << " -> " << opts.outFile << ": " << copyEc.message() << "\n";
        return 1;
      }
      cacheSavedRuntimeNs = parseGenerationRuntimeFromMeta(cacheMetaFile);
      uint64_t generated = countNonEmptyLines(opts.outFile);
      outs() << "Generated mutations: " << generated << " (cache hit)\n";
      outs() << "Mutation file: " << opts.outFile << "\n";
      outs() << "Mutation generation runtime_ns: " << elapsedNs() << "\n";
      outs() << "Mutation cache saved_runtime_ns: " << cacheSavedRuntimeNs
             << "\n";
      outs() << "Mutation cache lock_wait_ns: " << cacheLockWaitNs << "\n";
      outs() << "Mutation cache lock_contended: " << cacheLockContended << "\n";
      outs() << "Mutation cache status: hit\n";
      outs() << "Mutation cache file: " << cacheFile << "\n";
      if (afterLock)
        cacheLock.release();
      return 0;
    };

    if (auto rc = tryCacheHit(/*afterLock=*/false))
      return *rc;

    cacheLock.setLockDir(cacheFile + ".lock");
    auto lockStart = std::chrono::steady_clock::now();
    while (true) {
      std::error_code lockEc = sys::fs::create_directory(cacheLock.getLockDir());
      if (!lockEc) {
        cacheLock.setHeld(true);
        auto lockEnd = std::chrono::steady_clock::now();
        cacheLockWaitNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(lockEnd -
                                                                  lockStart)
                .count();
        break;
      }
      if (lockEc != std::errc::file_exists) {
        errs() << "circt-mut generate: failed to acquire cache lock: "
               << cacheLock.getLockDir() << ": " << lockEc.message() << "\n";
        return 1;
      }
      cacheLockContended = 1;

      sys::fs::file_status lockStatus;
      std::error_code statusEc = sys::fs::status(cacheLock.getLockDir(), lockStatus);
      if (!statusEc) {
        auto now = std::chrono::system_clock::now();
        auto ageSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                              now - lockStatus.getLastModificationTime())
                              .count();
        if (ageSeconds >= 3600) {
          SmallString<256> pidFile(cacheLock.getLockDir());
          sys::path::append(pidFile, "pid");
          (void)sys::fs::remove(pidFile);
          (void)sys::fs::remove(cacheLock.getLockDir());
          continue;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (auto rc = tryCacheHit(/*afterLock=*/true))
      return *rc;
  }

  SmallString<256> outParent(opts.outFile);
  sys::path::remove_filename(outParent);
  if (!outParent.empty()) {
    std::error_code ec = sys::fs::create_directories(outParent);
    if (ec) {
      errs() << "circt-mut generate: failed to create output directory: "
             << outParent << ": " << ec.message() << "\n";
      return 1;
    }
  }

  std::error_code ec;
  raw_fd_ostream out(opts.outFile, ec, sys::fs::OF_Text);
  if (ec) {
    errs() << "circt-mut generate: failed to open output file: " << opts.outFile
           << ": " << ec.message() << "\n";
    return 1;
  }

  SmallString<128> workDir;
  if (auto dirEC = sys::fs::createUniqueDirectory("circt-mut-generate", workDir)) {
    errs() << "circt-mut generate: failed to create temporary directory: "
           << dirEC.message() << "\n";
    return 1;
  }
  ScopedTempDir cleanup(workDir);

  std::string readCmd;
  StringRef designRef(opts.design);
  if (designRef.ends_with(".il"))
    readCmd = (Twine("read_rtlil ") + quoteForYosys(opts.design)).str();
  else if (designRef.ends_with(".sv"))
    readCmd = (Twine("read_verilog -sv ") + quoteForYosys(opts.design)).str();
  else if (designRef.ends_with(".v"))
    readCmd = (Twine("read_verilog ") + quoteForYosys(opts.design)).str();
  else {
    errs() << "circt-mut generate: unsupported design extension for "
           << opts.design << " (expected .il/.v/.sv)\n";
    return 1;
  }

  std::string prepCmd = "prep";
  if (!opts.top.empty())
    prepCmd = (Twine("prep -top ") + opts.top).str();

  auto runGenerationRound =
      [&](StringRef roundTag, uint64_t roundSeed, ArrayRef<uint64_t> roundCounts,
          SmallVectorImpl<std::string> &roundOutFiles) -> bool {
    roundOutFiles.clear();

    SmallString<256> scriptFile(workDir);
    sys::path::append(scriptFile, (Twine("mutate.") + roundTag + ".ys").str());
    SmallString<256> logFile(workDir);
    sys::path::append(logFile, (Twine("mutate.") + roundTag + ".log").str());

    std::error_code scriptEC;
    raw_fd_ostream script(scriptFile, scriptEC, sys::fs::OF_Text);
    if (scriptEC) {
      errs() << "circt-mut generate: failed to write yosys script: " << scriptFile
             << ": " << scriptEC.message() << "\n";
      return false;
    }
    script << readCmd << "\n";
    script << prepCmd << "\n";

    bool anyTarget = false;
    for (size_t i = 0; i < modeTargetList.size(); ++i) {
      uint64_t listCount = roundCounts[i];
      if (listCount == 0)
        continue;

      anyTarget = true;
      SmallString<256> sourcesFile(workDir);
      sys::path::append(sourcesFile,
                        (Twine("sources.") + roundTag + "." + Twine(i) + ".txt").str());
      SmallString<256> modeOutFile(workDir);
      sys::path::append(modeOutFile,
                        (Twine("mutations.") + roundTag + "." + Twine(i) + ".txt").str());
      roundOutFiles.push_back(std::string(modeOutFile.str()));

      std::string mutateCmd =
          (Twine("mutate -list ") + Twine(listCount) + " -seed " +
           Twine(roundSeed) + " -none")
              .str();
      for (const auto &cfg : finalCfgList)
        mutateCmd += (Twine(" -cfg ") + cfg.first + " " + cfg.second).str();

      StringRef mode = modeTargetList[i];
      if (!mode.empty())
        mutateCmd += (Twine(" -mode ") + mode).str();

      mutateCmd += (Twine(" -o ") + quoteForYosys(modeOutFile.str()) + " -s " +
                    quoteForYosys(sourcesFile.str()))
                       .str();
      for (const std::string &sel : finalSelects)
        mutateCmd += (Twine(" ") + sel).str();

      script << mutateCmd << "\n";
    }
    script.close();

    if (!anyTarget)
      return true;

    std::string scriptPath = std::string(scriptFile.str());
    std::string logPath = std::string(logFile.str());
    SmallVector<StringRef, 8> yosysArgs;
    yosysArgs.push_back(yosysExec);
    yosysArgs.push_back("-ql");
    yosysArgs.push_back(logPath);
    yosysArgs.push_back(scriptPath);

    std::string errMsg;
    int rc = sys::ExecuteAndWait(yosysExec, yosysArgs, /*Env=*/std::nullopt,
                                 /*Redirects=*/{}, /*SecondsToWait=*/0,
                                 /*MemoryLimit=*/0, &errMsg);
    if (!errMsg.empty())
      errs() << "circt-mut generate: execution error: " << errMsg << "\n";
    if (rc != 0) {
      errs() << "circt-mut generate: yosys failed (exit=" << rc
             << "), see log: " << logPath << "\n";
      return false;
    }

    for (const std::string &outPath : roundOutFiles) {
      if (!sys::fs::exists(outPath)) {
        errs() << "circt-mut generate: output file missing: " << outPath << "\n";
        return false;
      }
    }
    return true;
  };

  StringSet<> seenSpecs;
  uint64_t nextID = 1;

  auto consumeGenerated = [&](ArrayRef<std::string> roundOutFiles) {
    for (const std::string &outPath : roundOutFiles) {
      if (nextID > opts.count)
        return;
      auto bufferOrErr = MemoryBuffer::getFile(outPath);
      if (!bufferOrErr) {
        errs() << "circt-mut generate: failed to read " << outPath << "\n";
        continue;
      }

      StringRef content = bufferOrErr.get()->getBuffer();
      SmallVector<StringRef, 64> lines;
      content.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (StringRef line : lines) {
        if (nextID > opts.count)
          return;
        line = line.trim();
        if (line.empty() || line.front() == '#')
          continue;

        size_t splitPos = line.find_first_of(" \t");
        if (splitPos == StringRef::npos)
          continue;
        StringRef mutSpec = line.substr(splitPos + 1).trim();
        if (mutSpec.empty())
          continue;
        if (!seenSpecs.insert(mutSpec).second)
          continue;

        out << nextID << ' ' << mutSpec << '\n';
        ++nextID;
      }
    }
  };

  SmallVector<std::string, 16> roundOutFiles;
  if (!runGenerationRound("base", opts.seed, modeTargetCounts, roundOutFiles))
    return 1;
  consumeGenerated(roundOutFiles);

  constexpr uint64_t maxTopupRounds = 8;
  for (uint64_t round = 1; round <= maxTopupRounds && nextID <= opts.count;
       ++round) {
    uint64_t needed = opts.count - nextID + 1;
    uint64_t targetCount = modeTargetList.size();
    uint64_t topupBase = needed / targetCount;
    uint64_t topupExtra = needed % targetCount;
    uint64_t topupExtraStart = (opts.seed + round) % targetCount;

    SmallVector<uint64_t, 16> topupCounts;
    topupCounts.reserve(targetCount);
    for (uint64_t i = 0; i < targetCount; ++i) {
      uint64_t count = topupBase;
      if (isIndexInRotatedExtraPrefix(i, topupExtraStart, topupExtra,
                                      targetCount))
        ++count;
      topupCounts.push_back(count);
    }

    if (!runGenerationRound((Twine("topup") + Twine(round)).str(),
                            opts.seed + round, topupCounts, roundOutFiles))
      return 1;
    consumeGenerated(roundOutFiles);
  }

  out.close();

  if (nextID == 1) {
    errs() << "circt-mut generate: generation failed: empty mutation set\n";
    return 1;
  }
  if (nextID <= opts.count) {
    uint64_t generated = nextID - 1;
    errs() << "circt-mut generate: unable to produce requested count after "
              "dedup/top-up (requested="
           << opts.count << " generated=" << generated << ")\n";
    return 1;
  }

  uint64_t generated = nextID - 1;

  if (cacheEnabled) {
    std::string cacheTmp = cacheFile + ".tmp." + std::to_string(sys::Process::getProcessId());
    std::string cacheMetaTmp = cacheMetaFile + ".tmp." + std::to_string(sys::Process::getProcessId());

    std::error_code copyEc = copyFileReplace(opts.outFile, cacheTmp);
    if (copyEc) {
      errs() << "circt-mut generate: failed to write cache file: " << cacheTmp
             << ": " << copyEc.message() << "\n";
      return 1;
    }

    std::error_code metaEc;
    raw_fd_ostream metaOut(cacheMetaTmp, metaEc, sys::fs::OF_Text);
    if (metaEc) {
      errs() << "circt-mut generate: failed to write cache metadata: "
             << cacheMetaTmp << ": " << metaEc.message() << "\n";
      return 1;
    }
    metaOut << "generation_runtime_ns\t" << elapsedNs() << "\n";
    metaOut.close();

    std::error_code renameEc = sys::fs::rename(cacheTmp, cacheFile);
    if (renameEc) {
      errs() << "circt-mut generate: failed to publish cache file: " << cacheTmp
             << " -> " << cacheFile << ": " << renameEc.message() << "\n";
      return 1;
    }
    renameEc = sys::fs::rename(cacheMetaTmp, cacheMetaFile);
    if (renameEc) {
      errs() << "circt-mut generate: failed to publish cache metadata: "
             << cacheMetaTmp << " -> " << cacheMetaFile << ": "
             << renameEc.message() << "\n";
      return 1;
    }
    cacheLock.release();
  }

  outs() << "Generated mutations: " << generated << "\n";
  outs() << "Mutation file: " << opts.outFile << "\n";
  outs() << "Mutation generation runtime_ns: " << elapsedNs() << "\n";
  outs() << "Mutation cache saved_runtime_ns: 0\n";
  outs() << "Mutation cache lock_wait_ns: " << cacheLockWaitNs << "\n";
  outs() << "Mutation cache lock_contended: " << cacheLockContended << "\n";
  if (cacheEnabled) {
    outs() << "Mutation cache status: miss\n";
    outs() << "Mutation cache file: " << cacheFile << "\n";
  } else {
    outs() << "Mutation cache status: disabled\n";
  }

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  if (argc < 2) {
    printHelp(outs());
    return 1;
  }

  StringRef firstArg(argv[1]);
  if (firstArg == "-h" || firstArg == "--help" || firstArg == "help") {
    printHelp(outs());
    return 0;
  }
  if (firstArg == "--version") {
    outs() << "circt-mut " << circt::getCirctVersion() << "\n";
    return 0;
  }

  if (firstArg == "init") {
    SmallVector<StringRef, 16> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);

    InitParseResult parseResult = parseInitArgs(forwardedArgs);
    if (!parseResult.ok) {
      errs() << parseResult.error << "\n";
      return 1;
    }
    if (parseResult.showHelp) {
      printInitHelp(outs());
      return 0;
    }
    return runNativeInit(parseResult.opts);
  }

  if (firstArg == "run") {
    SmallVector<StringRef, 16> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);

    RunParseResult parseResult = parseRunArgs(forwardedArgs);
    if (!parseResult.ok) {
      errs() << parseResult.error << "\n";
      return 1;
    }
    if (parseResult.showHelp) {
      printRunHelp(outs());
      return 0;
    }
    return runNativeRun(argv[0], parseResult.opts);
  }

  if (firstArg == "report") {
    SmallVector<StringRef, 16> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);

    ReportParseResult parseResult = parseReportArgs(forwardedArgs);
    if (!parseResult.ok) {
      errs() << parseResult.error << "\n";
      return 1;
    }
    if (parseResult.showHelp) {
      printReportHelp(outs());
      return 0;
    }
    return runNativeReport(parseResult.opts);
  }

  if (firstArg == "generate") {
    SmallVector<StringRef, 16> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);

    GenerateParseResult parseResult = parseGenerateArgs(forwardedArgs);
    if (!parseResult.ok) {
      errs() << parseResult.error << "\n";
      return 1;
    }
    if (parseResult.showHelp) {
      printGenerateHelp(outs());
      return 0;
    }
    if (parseResult.fallbackToScript) {
      auto scriptPath = resolveScriptPath(argv[0], "generate_mutations_yosys.sh");
      if (!scriptPath) {
        errs() << "circt-mut: native generate path does not support this option set "
                  "yet and script backend is unavailable.\n";
        errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree "
                  "with utils scripts.\n";
        return 1;
      }
      return dispatchToScript(*scriptPath, forwardedArgs);
    }
    return runNativeGenerate(parseResult.opts);
  }

  if (firstArg == "cover") {
    SmallVector<StringRef, 32> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);
    return runCoverFlow(argv[0], forwardedArgs);
  }

  if (firstArg == "matrix") {
    SmallVector<StringRef, 32> forwardedArgs;
    for (int i = 2; i < argc; ++i)
      forwardedArgs.push_back(argv[i]);
    return runMatrixFlow(argv[0], forwardedArgs);
  }

  auto scriptName = mapSubcommandToScript(firstArg);
  if (!scriptName) {
    errs() << "circt-mut: unknown subcommand: " << firstArg << "\n";
    printHelp(errs());
    return 1;
  }

  auto scriptPath = resolveScriptPath(argv[0], *scriptName);
  if (!scriptPath) {
    errs() << "circt-mut: unable to locate script '" << *scriptName << "'.\n";
    errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree with"
              " utils scripts.\n";
    return 1;
  }

  SmallVector<StringRef, 16> forwardedArgs;
  for (int i = 2; i < argc; ++i)
    forwardedArgs.push_back(argv[i]);

  return dispatchToScript(*scriptPath, forwardedArgs);
}
