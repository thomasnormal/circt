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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <set>
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
  os << "  --matrix-native-dispatch BOOL\n";
  os << "                           Enable native matrix dispatch in generated\n";
  os << "                           config (default: false)\n";
  os << "  --matrix-native-global-filter-prequalify BOOL\n";
  os << "                           Enable native matrix prequalification in\n";
  os << "                           generated config (default: false)\n";
  os << "  --report-policy-mode MODE\n";
  os << "                           Report policy mode for generated config\n";
  os << "                           (smoke|nightly|strict|trend-nightly|trend-strict|\n";
  os << "                            native-trend-nightly|native-trend-strict|\n";
  os << "                            provenance-guard|provenance-strict|\n";
  os << "                            native-lifecycle-strict|native-smoke|\n";
  os << "                            native-nightly|native-strict|\n";
  os << "                            native-strict-formal|strict-formal|\n";
  os << "                            native-strict-formal-summary|\n";
  os << "                            strict-formal-summary|\n";
  os << "                            native-strict-formal-summary-v1|\n";
  os << "                            strict-formal-summary-v1,\n";
  os << "                            default: smoke)\n";
  os << "  --report-policy-stop-on-fail BOOL\n";
  os << "                           Enable stop-on-fail report guard profile in\n";
  os << "                           generated config (default: true)\n";
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
  os << "  --with-report            Run 'circt-mut report' after run completes\n";
  os << "  --with-report-on-fail    Run report even if run flow fails\n";
  os << "  --report-mode MODE       cover|matrix|all (default: same as --mode)\n";
  os << "  --report-compare FILE    Override post-run report --compare\n";
  os << "  --report-compare-history-latest FILE\n";
  os << "                           Override post-run report --compare-history-latest\n";
  os << "  --report-history FILE    Override post-run report --history\n";
  os << "  --report-append-history FILE\n";
  os << "                           Override post-run report --append-history\n";
  os << "  --report-trend-history FILE\n";
  os << "                           Override post-run report --trend-history\n";
  os << "  --report-trend-window N  Override post-run report --trend-window\n";
  os << "  --report-history-max-runs N\n";
  os << "                           Override post-run report --history-max-runs\n";
  os << "  --report-out FILE        Override post-run report --out\n";
  os << "  --report-fail-if-value-gt RULE\n";
  os << "                           Repeatable override for report --fail-if-value-gt\n";
  os << "  --report-fail-if-value-lt RULE\n";
  os << "                           Repeatable override for report --fail-if-value-lt\n";
  os << "  --report-fail-if-delta-gt RULE\n";
  os << "                           Repeatable override for report --fail-if-delta-gt\n";
  os << "  --report-fail-if-delta-lt RULE\n";
  os << "                           Repeatable override for report --fail-if-delta-lt\n";
  os << "  --report-fail-if-trend-delta-gt RULE\n";
  os << "                           Repeatable override for report --fail-if-trend-delta-gt\n";
  os << "  --report-fail-if-trend-delta-lt RULE\n";
  os << "                           Repeatable override for report --fail-if-trend-delta-lt\n";
  os << "  --report-history-bootstrap\n";
  os << "                           Enable post-run report --history-bootstrap\n";
  os << "  --report-no-history-bootstrap\n";
  os << "                           Disable post-run report history bootstrap\n";
  os << "  --report-policy-profile NAME\n";
  os << "                           Repeatable post-run report policy profile\n";
  os << "  --report-policy-mode MODE\n";
  os << "                           smoke|nightly|strict|trend-nightly|trend-strict|\n";
  os << "                           native-trend-nightly|native-trend-strict|\n";
  os << "                           provenance-guard|provenance-strict|\n";
  os << "                           native-lifecycle-strict|native-smoke|\n";
  os << "                           native-nightly|native-strict|\n";
  os << "                           native-strict-formal|strict-formal|\n";
  os << "                           native-strict-formal-summary|\n";
  os << "                           strict-formal-summary|\n";
  os << "                           native-strict-formal-summary-v1|\n";
  os << "                           strict-formal-summary-v1\n";
  os << "                           (maps to report policy profile)\n";
  os << "  --report-external-formal-results FILE\n";
  os << "                           Repeatable override for report\n";
  os << "                           --external-formal-results\n";
  os << "  --report-external-formal-out-dir DIR\n";
  os << "                           Override report --external-formal-out-dir\n";
  os << "  --report-policy-stop-on-fail BOOL\n";
  os << "                           1|0|true|false|yes|no|on|off\n";
  os << "  --report-fail-on-prequalify-drift\n";
  os << "                           Force-enable report prequalify drift gate\n";
  os << "  --report-no-fail-on-prequalify-drift\n";
  os << "                           Force-disable report prequalify drift gate\n";
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
  os << "  --external-formal-results FILE\n";
  os << "                           Repeatable external formal results file\n";
  os << "  --external-formal-out-dir DIR\n";
  os << "                           Discover external formal results from out dir\n";
  os << "  --compare FILE           Compare against baseline report TSV\n";
  os << "  --compare-history-latest FILE\n";
  os << "                           Compare against latest snapshot in history TSV\n";
  os << "  --history FILE           Shorthand for compare-latest + trend + append history\n";
  os << "  --history-bootstrap      Allow missing --history file by skipping compare/trend gates on bootstrap run\n";
  os << "  --history-max-runs N     Keep only latest N runs in history after append\n";
  os << "  --trend-history FILE     Compute trend summary from history TSV\n";
  os << "  --trend-window N         Use latest N history runs for trends (0=all)\n";
  os << "  --policy-profile NAME    Apply built-in report policy profile\n";
  os << "  --policy-mode MODE       smoke|nightly|strict|trend-nightly|trend-strict|\n";
  os << "                           native-trend-nightly|native-trend-strict|\n";
  os << "                           provenance-guard|provenance-strict|\n";
  os << "                           native-lifecycle-strict|native-smoke|\n";
  os << "                           native-nightly|native-strict|\n";
  os << "                           native-strict-formal|strict-formal|\n";
  os << "                           native-strict-formal-summary|\n";
  os << "                           strict-formal-summary|\n";
  os << "                           native-strict-formal-summary-v1|\n";
  os << "                           strict-formal-summary-v1\n";
  os << "                           (maps to report policy profile)\n";
  os << "  --policy-stop-on-fail BOOL\n";
  os << "                           1|0|true|false|yes|no|on|off\n";
  os << "                           formal-regression-basic|formal-regression-trend|\n";
  os << "                           formal-regression-matrix-basic|\n";
  os << "                           formal-regression-matrix-trend|\n";
  os << "                           formal-regression-matrix-guard|\n";
  os << "                           formal-regression-matrix-trend-guard|\n";
  os << "                           formal-regression-matrix-guard-smoke|\n";
  os << "                           formal-regression-matrix-guard-nightly|\n";
  os << "                           formal-regression-matrix-stop-on-fail-guard-smoke|\n";
  os << "                           formal-regression-matrix-stop-on-fail-guard-nightly|\n";
  os << "                           formal-regression-matrix-guard-strict|\n";
  os << "                           formal-regression-matrix-nightly|\n";
  os << "                           formal-regression-matrix-strict|\n";
  os << "                           formal-regression-matrix-stop-on-fail-basic|\n";
  os << "                           formal-regression-matrix-stop-on-fail-trend|\n";
  os << "                           formal-regression-matrix-stop-on-fail-strict|\n";
  os << "                           formal-regression-matrix-full-lanes-strict|\n";
  os << "                           formal-regression-matrix-lane-drift-nightly|\n";
  os << "                           formal-regression-matrix-lane-drift-strict|\n";
  os << "                           formal-regression-matrix-lane-trend-nightly|\n";
  os << "                           formal-regression-matrix-lane-trend-strict|\n";
  os << "                           formal-regression-matrix-external-formal-guard|\n";
  os << "                           formal-regression-matrix-external-formal-summary-guard|\n";
  os << "                           formal-regression-matrix-external-formal-summary-v1-guard|\n";
  os << "                           formal-regression-matrix-provenance-guard|\n";
  os << "                           formal-regression-matrix-provenance-strict|\n";
  os << "                           formal-regression-matrix-native-lifecycle-strict|\n";
  os << "                           formal-regression-matrix-policy-mode-native-strict-contract|\n";
  os << "                           formal-regression-matrix-policy-mode-native-family-contract|\n";
  os << "                           formal-regression-matrix-runtime-smoke|\n";
  os << "                           formal-regression-matrix-runtime-nightly|\n";
  os << "                           formal-regression-matrix-runtime-trend|\n";
  os << "                           formal-regression-matrix-trend-history-quality|\n";
  os << "                           formal-regression-matrix-trend-history-quality-strict|\n";
  os << "                           formal-regression-matrix-runtime-strict|\n";
  os << "                           formal-regression-matrix-composite-smoke|\n";
  os << "                           formal-regression-matrix-composite-nightly|\n";
  os << "                           formal-regression-matrix-composite-strict|\n";
  os << "                           formal-regression-matrix-composite-native-strict|\n";
  os << "                           formal-regression-matrix-composite-trend-nightly|\n";
  os << "                           formal-regression-matrix-composite-trend-strict|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-smoke|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-nightly|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-strict|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-native-strict|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-trend-nightly|\n";
  os << "                           formal-regression-matrix-composite-stop-on-fail-trend-strict\n";
  os << "  --append-history FILE    Append current report rows to history TSV\n";
  os << "  --fail-if-value-gt RULE  Fail if current numeric value exceeds threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
  os << "  --fail-if-value-lt RULE  Fail if current numeric value is below threshold\n";
  os << "                           RULE format: <metric>=<value>\n";
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
  os << "  --fail-on-prequalify-drift\n";
  os << "                           Fail if matrix prequalify results.tsv counters\n";
  os << "                           drift from native prequalify summary counters\n";
  os << "  --no-fail-on-prequalify-drift\n";
  os << "                           Explicitly disable prequalify drift gate\n";
  os << "  --lane-budget-out FILE   Write per-lane matrix budget TSV artifact\n";
  os << "  --skip-budget-out FILE   Write matrix skip-budget TSV artifact\n";
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

static bool isKnownMutationMode(StringRef mode) {
  return mode == "inv" || mode == "const0" || mode == "const1" ||
         mode == "cnot0" || mode == "cnot1" || mode == "arith" ||
         mode == "control" || mode == "balanced" || mode == "all" ||
         mode == "stuck" || mode == "invert" || mode == "connect";
}

static std::optional<std::string>
firstUnknownMutationModeCSV(StringRef csv) {
  if (csv.empty())
    return std::nullopt;
  SmallVector<StringRef, 16> entries;
  csv.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef raw : entries) {
    StringRef mode = raw.trim();
    if (mode.empty())
      continue;
    if (!isKnownMutationMode(mode))
      return mode.str();
  }
  return std::nullopt;
}

static std::optional<std::string>
firstUnknownMutationModeInAllocationCSV(StringRef csv) {
  if (csv.empty())
    return std::nullopt;
  SmallVector<StringRef, 16> entries;
  csv.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef rawEntry : entries) {
    StringRef entry = rawEntry.trim();
    if (entry.empty())
      continue;
    auto split = entry.split('=');
    StringRef modeName = split.first.trim();
    if (modeName.empty() || split.second == split.first)
      continue;
    if (!isKnownMutationMode(modeName))
      return modeName.str();
  }
  return std::nullopt;
}

static bool isKnownMutationProfile(StringRef profile) {
  return profile == "arith-depth" || profile == "control-depth" ||
         profile == "balanced-depth" || profile == "fault-basic" ||
         profile == "fault-stuck" || profile == "fault-connect" ||
         profile == "cover" || profile == "none";
}

static std::optional<std::string> firstUnknownMutationProfile(StringRef csv) {
  if (csv.empty())
    return std::nullopt;
  SmallVector<StringRef, 16> entries;
  csv.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef raw : entries) {
    StringRef profile = raw.trim();
    if (profile.empty())
      continue;
    if (!isKnownMutationProfile(profile))
      return profile.str();
  }
  return std::nullopt;
}

struct CoverRewriteResult {
  bool ok = false;
  std::string error;
  SmallVector<std::string, 32> rewrittenArgs;
  bool nativeGlobalFilterProbe = false;
  std::string nativeGlobalFilterProbeMutant;
  std::string nativeGlobalFilterProbeLog;
  bool nativeGlobalFilterPrequalify = false;
  bool nativeGlobalFilterPrequalifyOnly = false;
  std::string nativeGlobalFilterPrequalifyPairFile;
};

static CoverRewriteResult rewriteCoverArgs(const char *argv0,
                                           ArrayRef<StringRef> args) {
  CoverRewriteResult result;
  bool hasMutationsFile = false;
  bool hasGenerateMutations = false;
  bool wantsHelp = false;
  std::string generateMutations;
  std::string mutationsSeed;
  std::string mutationsModes;
  std::string mutationsProfiles;
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
  std::string reuseCacheMode;
  bool hasGlobalFilterCmd = false;
  bool hasGlobalFilterLEC = false;
  bool hasGlobalFilterBMC = false;
  bool hasGlobalFilterChain = false;
  bool hasReusePairFile = false;
  std::string globalFilterChainMode;
  bool nativeGlobalFilterProbe = false;
  std::string nativeGlobalFilterProbeMutant;
  std::string nativeGlobalFilterProbeLog;
  bool nativeGlobalFilterPrequalify = false;
  bool nativeGlobalFilterPrequalifyOnly = false;
  std::string nativeGlobalFilterPrequalifyPairFile;
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
    auto resolveWithRequiredValue =
        [&](StringRef flag,
            std::optional<StringRef> autoToolName = std::nullopt) -> bool {
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
      std::optional<std::string> resolved;
      if (requested == "auto") {
        if (!autoToolName) {
          result.error = (Twine("circt-mut cover: invalid value for ") + flag +
                          ": auto")
                             .str();
          return false;
        }
        resolved = resolveToolPathFromEnvPath(*autoToolName);
        if (!resolved) {
          result.error = (Twine("circt-mut cover: unable to resolve ") + flag +
                          " executable: auto (searched PATH).")
                             .str();
          return false;
        }
      } else {
        resolved = resolveToolPath(requested);
      }
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
      if (!resolveWithRequiredValue("--formal-global-propagate-z3", "z3"))
        return result;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-z3" ||
        arg.starts_with("--formal-global-propagate-bmc-z3=")) {
      if (!resolveWithRequiredValue("--formal-global-propagate-bmc-z3", "z3"))
        return result;
      continue;
    }
    if (arg == "--mutations-yosys" || arg.starts_with("--mutations-yosys=")) {
      if (!resolveWithRequiredValue("--mutations-yosys"))
        return result;
      continue;
    }
    if (arg == "--native-global-filter-probe-mutant" ||
        arg.starts_with("--native-global-filter-probe-mutant=")) {
      nativeGlobalFilterProbe = true;
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        nativeGlobalFilterProbeMutant = arg.substr(eqPos + 1).str();
      } else if (i + 1 < args.size()) {
        nativeGlobalFilterProbeMutant = args[++i].str();
      }
      if (nativeGlobalFilterProbeMutant.empty()) {
        result.error =
            "circt-mut cover: missing value for --native-global-filter-probe-mutant.";
        return result;
      }
      continue;
    }
    if (arg == "--native-global-filter-probe-log" ||
        arg.starts_with("--native-global-filter-probe-log=")) {
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        nativeGlobalFilterProbeLog = arg.substr(eqPos + 1).str();
      } else if (i + 1 < args.size()) {
        nativeGlobalFilterProbeLog = args[++i].str();
      }
      if (nativeGlobalFilterProbeLog.empty()) {
        result.error =
            "circt-mut cover: missing value for --native-global-filter-probe-log.";
        return result;
      }
      continue;
    }
    if (arg == "--native-global-filter-prequalify") {
      nativeGlobalFilterPrequalify = true;
      continue;
    }
    if (arg == "--native-global-filter-prequalify-only") {
      nativeGlobalFilterPrequalify = true;
      nativeGlobalFilterPrequalifyOnly = true;
      continue;
    }
    if (arg == "--native-global-filter-prequalify-pair-file" ||
        arg.starts_with("--native-global-filter-prequalify-pair-file=")) {
      size_t eqPos = arg.find('=');
      if (eqPos != StringRef::npos) {
        nativeGlobalFilterPrequalifyPairFile = arg.substr(eqPos + 1).str();
      } else if (i + 1 < args.size()) {
        nativeGlobalFilterPrequalifyPairFile = args[++i].str();
      }
      if (nativeGlobalFilterPrequalifyPairFile.empty()) {
        result.error =
            "circt-mut cover: missing value for --native-global-filter-prequalify-pair-file.";
        return result;
      }
      continue;
    }
    if (arg == "--mutations-file" || arg.starts_with("--mutations-file="))
      hasMutationsFile = true;
    if (arg == "--reuse-pair-file" || arg.starts_with("--reuse-pair-file="))
      hasReusePairFile = true;
    if (arg == "-h" || arg == "--help")
      wantsHelp = true;
    if (arg == "--generate-mutations" ||
        arg.starts_with("--generate-mutations=")) {
      hasGenerateMutations = true;
      generateMutations = valueFromArg().str();
    }
    if (arg == "--mutations-seed" || arg.starts_with("--mutations-seed="))
      mutationsSeed = valueFromArg().str();
    if (arg == "--mutations-modes" || arg.starts_with("--mutations-modes="))
      mutationsModes = valueFromArg().str();
    if (arg == "--mutations-mode-counts" ||
        arg.starts_with("--mutations-mode-counts="))
      mutationsModeCounts = valueFromArg().str();
    if (arg == "--mutations-profiles" ||
        arg.starts_with("--mutations-profiles="))
      mutationsProfiles = valueFromArg().str();
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
    if (arg == "--reuse-cache-mode" ||
        arg.starts_with("--reuse-cache-mode=")) {
      reuseCacheMode = valueFromArg().str();
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
    if (auto unknown = firstUnknownMutationProfile(mutationsProfiles)) {
      result.error = (Twine("circt-mut cover: unknown --mutations-profiles "
                            "value: ") +
                      *unknown +
                      " (expected arith-depth|control-depth|balanced-depth|"
                      "fault-basic|fault-stuck|fault-connect|cover|none).")
                         .str();
      return result;
    }
    if (auto unknown = firstUnknownMutationModeCSV(mutationsModes)) {
      result.error = (Twine("circt-mut cover: unknown --mutations-modes "
                            "value: ") +
                      *unknown +
                      " (expected inv|const0|const1|cnot0|cnot1|"
                      "arith|control|balanced|all|stuck|invert|connect).")
                         .str();
      return result;
    }
    if (!mutationsSeed.empty() && !Regex("^[0-9]+$").match(mutationsSeed)) {
      result.error = (Twine("circt-mut cover: invalid --mutations-seed value: ") +
                      mutationsSeed + " (expected 0-9 integer).")
                         .str();
      return result;
    }
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
    if (auto unknown =
            firstUnknownMutationModeInAllocationCSV(mutationsModeCounts)) {
      result.error =
          (Twine("circt-mut cover: unknown --mutations-mode-counts mode: ") +
           *unknown +
           " (expected inv|const0|const1|cnot0|cnot1|"
           "arith|control|balanced|all|stuck|invert|connect).")
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
    if (auto unknown =
            firstUnknownMutationModeInAllocationCSV(mutationsModeWeights)) {
      result.error =
          (Twine("circt-mut cover: unknown --mutations-mode-weights mode: ") +
           *unknown +
           " (expected inv|const0|const1|cnot0|cnot1|"
           "arith|control|balanced|all|stuck|invert|connect).")
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
  if (!validateCoverRegex(reuseCacheMode, Regex("^(off|read|read-write)$"),
                          "--reuse-cache-mode", "off|read|read-write"))
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
  if (!wantsHelp && !nativeGlobalFilterProbe && !nativeGlobalFilterPrequalify &&
      !hasMutationsFile && !hasGenerateMutations) {
    result.error = "circt-mut cover: requires either --mutations-file or "
                   "--generate-mutations.";
    return result;
  }
  if (nativeGlobalFilterProbe && !hasGlobalFilterCmd && !hasGlobalFilterLEC &&
      !hasGlobalFilterBMC && !hasGlobalFilterChain) {
    result.error =
        "circt-mut cover: --native-global-filter-probe-mutant requires "
        "a global filter mode (--formal-global-propagate-cmd, "
        "--formal-global-propagate-circt-lec, "
        "--formal-global-propagate-circt-bmc, or "
        "--formal-global-propagate-circt-chain).";
    return result;
  }
  if (nativeGlobalFilterProbe && nativeGlobalFilterPrequalify) {
    result.error =
        "circt-mut cover: use either --native-global-filter-probe-mutant or "
        "--native-global-filter-prequalify, not both.";
    return result;
  }
  if (nativeGlobalFilterPrequalify && !hasMutationsFile &&
      !hasGenerateMutations) {
    result.error = "circt-mut cover: --native-global-filter-prequalify "
                   "requires either --mutations-file or "
                   "--generate-mutations.";
    return result;
  }
  if (nativeGlobalFilterPrequalify && hasReusePairFile) {
    result.error = "circt-mut cover: --native-global-filter-prequalify cannot "
                   "be combined with --reuse-pair-file.";
    return result;
  }
  if (nativeGlobalFilterPrequalify && !hasGlobalFilterCmd &&
      !hasGlobalFilterLEC && !hasGlobalFilterBMC && !hasGlobalFilterChain) {
    result.error =
        "circt-mut cover: --native-global-filter-prequalify requires "
        "a global filter mode (--formal-global-propagate-cmd, "
        "--formal-global-propagate-circt-lec, "
        "--formal-global-propagate-circt-bmc, or "
        "--formal-global-propagate-circt-chain).";
    return result;
  }

  result.nativeGlobalFilterProbe = nativeGlobalFilterProbe;
  result.nativeGlobalFilterProbeMutant = nativeGlobalFilterProbeMutant;
  result.nativeGlobalFilterProbeLog = nativeGlobalFilterProbeLog;
  result.nativeGlobalFilterPrequalify = nativeGlobalFilterPrequalify;
  result.nativeGlobalFilterPrequalifyOnly = nativeGlobalFilterPrequalifyOnly;
  result.nativeGlobalFilterPrequalifyPairFile =
      nativeGlobalFilterPrequalifyPairFile;
  result.ok = true;
  return result;
}

struct ProbeRawResult {
  std::string state;
  int rc = -1;
  std::string source;
};

struct CoverGlobalFilterProbeConfig {
  std::string design;
  std::string mutantDesign;
  std::string logFile;
  std::string globalFilterLEC;
  std::string globalFilterBMC;
  std::string globalFilterChain;
  std::string globalFilterCmd;
  std::string globalFilterLECToolArgs;
  std::string globalFilterBmcToolArgs;
  std::string globalFilterZ3;
  std::string globalFilterBMCZ3;
  std::string globalFilterC1 = "top";
  std::string globalFilterC2 = "top";
  std::string globalFilterBMCModule = "top";
  bool globalFilterAssumeKnownInputs = false;
  bool globalFilterAcceptXpropOnly = false;
  bool globalFilterBMCRunSmtlib = false;
  bool globalFilterBMCAssumeKnownInputs = false;
  uint64_t globalFilterLECTimeoutSeconds = 0;
  uint64_t globalFilterBMCTimeoutSeconds = 0;
  uint64_t globalFilterCmdTimeoutSeconds = 0;
  uint64_t globalFilterBMCBound = 20;
  uint64_t globalFilterBMCIgnoreAssertsUntil = 0;
};

struct CoverGlobalFilterProbeOutcome {
  std::string classification;
  int finalRC = -1;
  std::string cmdSource;
};

static std::string shellQuote(StringRef value) {
  std::string quoted;
  quoted.reserve(value.size() + 8);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'')
      quoted += "'\"'\"'";
    else
      quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

static bool isTimeoutExitCode(int rc) {
  return rc == 124 || rc == 137 || rc == 143;
}

static bool ensureParentDirForFile(StringRef path, std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (parent.empty())
    return true;
  std::error_code ec = sys::fs::create_directories(parent);
  if (ec) {
    error = (Twine("circt-mut cover: failed to create directory for ") + path +
             ": " + ec.message())
                .str();
    return false;
  }
  return true;
}

static bool runArgvToLog(ArrayRef<std::string> argv, StringRef logPath,
                         uint64_t timeoutSeconds, int &rc,
                         std::string &error) {
  std::string cmd;
  if (timeoutSeconds > 0) {
    cmd += "timeout --signal=TERM --kill-after=5 ";
    cmd += std::to_string(timeoutSeconds);
    cmd += " ";
  }
  for (const std::string &arg : argv) {
    cmd += shellQuote(arg);
    cmd.push_back(' ');
  }
  cmd += ">";
  cmd += shellQuote(logPath);
  cmd += " 2>&1";

  SmallVector<StringRef, 8> shArgs;
  shArgs.push_back("env");
  shArgs.push_back("bash");
  shArgs.push_back("-lc");
  shArgs.push_back(cmd);
  std::string errMsg;
  rc = sys::ExecuteAndWait("/usr/bin/env", shArgs, /*Env=*/std::nullopt,
                           /*Redirects=*/{},
                           /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (!errMsg.empty()) {
    error = (Twine("circt-mut cover: execution error: ") + errMsg).str();
    return false;
  }
  return true;
}

static std::string readTextFileOrEmpty(StringRef path);

static bool
runCommandStringToLogWithMutationEnv(StringRef runDir, StringRef command,
                                     StringRef logPath, uint64_t timeoutSeconds,
                                     StringRef design, StringRef mutantDesign,
                                     StringRef mutationID, StringRef mutationSpec,
                                     int &rc, std::string &error) {
  std::string shell;
  shell += "cd ";
  shell += shellQuote(runDir);
  shell += " && ";
  shell += "export ORIG_DESIGN=";
  shell += shellQuote(design);
  shell += " MUTANT_DESIGN=";
  shell += shellQuote(mutantDesign);
  shell += " MUTATION_ID=";
  shell += shellQuote(mutationID);
  shell += " MUTATION_SPEC=";
  shell += shellQuote(mutationSpec);
  shell += " MUTATION_WORKDIR=";
  shell += shellQuote(runDir);
  shell += " && ";
  if (timeoutSeconds > 0) {
    shell += "timeout --signal=TERM --kill-after=5 ";
    shell += std::to_string(timeoutSeconds);
    shell += " ";
  }
  shell += "bash -lc ";
  shell += shellQuote(command);
  shell += " >";
  shell += shellQuote(logPath);
  shell += " 2>&1";

  SmallVector<StringRef, 8> shArgs;
  shArgs.push_back("env");
  shArgs.push_back("bash");
  shArgs.push_back("-lc");
  shArgs.push_back(shell);
  std::string errMsg;
  rc = sys::ExecuteAndWait("/usr/bin/env", shArgs, /*Env=*/std::nullopt,
                           /*Redirects=*/{},
                           /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (!errMsg.empty()) {
    error = (Twine("circt-mut cover: execution error: ") + errMsg).str();
    return false;
  }
  return true;
}

static ProbeRawResult runCoverGlobalFilterCmdRaw(
    StringRef runDir, StringRef command, StringRef logPath,
    uint64_t timeoutSeconds, StringRef design, StringRef mutantDesign,
    StringRef mutationID, StringRef mutationSpec, std::string &error) {
  int rc = -1;
  if (!runCommandStringToLogWithMutationEnv(runDir, command, logPath,
                                            timeoutSeconds, design,
                                            mutantDesign, mutationID,
                                            mutationSpec, rc, error))
    return ProbeRawResult{"error", rc, "exec_error"};

  std::string logText = readTextFileOrEmpty(logPath);
  Regex notPropagatedToken("(^|[^[:alnum:]_])NOT_PROPAGATED([^[:alnum:]_]|$)",
                           Regex::IgnoreCase);
  Regex propagatedToken("(^|[^[:alnum:]_])PROPAGATED([^[:alnum:]_]|$)",
                        Regex::IgnoreCase);
  if (notPropagatedToken.match(logText))
    return ProbeRawResult{"not_propagated", rc, "token_not_propagated"};
  if (propagatedToken.match(logText))
    return ProbeRawResult{"propagated", rc, "token_propagated"};
  if (timeoutSeconds > 0 && isTimeoutExitCode(rc))
    return ProbeRawResult{"propagated", rc, "timeout"};
  if (rc == 0)
    return ProbeRawResult{"not_propagated", rc, "rc0"};
  if (rc == 1)
    return ProbeRawResult{"propagated", rc, "rc1"};
  return ProbeRawResult{"error", rc, "error"};
}

static std::string readTextFileOrEmpty(StringRef path) {
  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return std::string();
  return std::string(bufferOrErr.get()->getBuffer());
}

static std::string extractLastLecResultToken(StringRef text) {
  std::string token;
  SmallVector<StringRef, 128> lines;
  text.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    if (line.contains("LEC_RESULT=EQ"))
      token = "eq";
    else if (line.contains("LEC_RESULT=NEQ"))
      token = "neq";
    else if (line.contains("LEC_RESULT=UNKNOWN"))
      token = "unknown";
  }
  return token;
}

static std::string extractLastBMCResultToken(StringRef text) {
  std::string token;
  SmallVector<StringRef, 128> lines;
  text.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef line : lines) {
    if (line.contains("BMC_RESULT=SAT"))
      token = "sat";
    else if (line.contains("BMC_RESULT=UNSAT"))
      token = "unsat";
    else if (line.contains("BMC_RESULT=UNKNOWN"))
      token = "unknown";
  }
  return token;
}

static void appendSplitArgs(StringRef raw, SmallVectorImpl<std::string> &out) {
  if (raw.empty())
    return;
  SmallVector<StringRef, 16> parts;
  raw.split(parts, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef part : parts) {
    StringRef trimmed = part.trim();
    if (!trimmed.empty())
      out.push_back(trimmed.str());
  }
}

static ProbeRawResult
runCoverGlobalFilterLECRaw(const CoverGlobalFilterProbeConfig &cfg,
                           StringRef logPath, std::string &error) {
  ProbeRawResult result;
  if (cfg.globalFilterLEC.empty()) {
    result.state = "error";
    result.rc = -1;
    return result;
  }
  SmallVector<std::string, 16> cmd;
  cmd.push_back(cfg.globalFilterLEC);
  cmd.push_back("--run-smtlib");
  if (!cfg.globalFilterZ3.empty())
    cmd.push_back((Twine("--z3-path=") + cfg.globalFilterZ3).str());
  if (cfg.globalFilterAssumeKnownInputs)
    cmd.push_back("--assume-known-inputs");
  if (cfg.globalFilterAcceptXpropOnly)
    cmd.push_back("--accept-xprop-only");
  appendSplitArgs(cfg.globalFilterLECToolArgs, cmd);
  cmd.push_back((Twine("-c1=") + cfg.globalFilterC1).str());
  cmd.push_back((Twine("-c2=") + cfg.globalFilterC2).str());
  cmd.push_back(cfg.design);
  cmd.push_back(cfg.mutantDesign);

  int rc = -1;
  if (!runArgvToLog(cmd, logPath, cfg.globalFilterLECTimeoutSeconds, rc, error))
    return ProbeRawResult{"error", rc};
  std::string logText = readTextFileOrEmpty(logPath);
  std::string token = extractLastLecResultToken(logText);
  if (!token.empty())
    return ProbeRawResult{token, rc};
  if (isTimeoutExitCode(rc))
    return ProbeRawResult{"timeout", rc};
  return ProbeRawResult{"error", rc};
}

static ProbeRawResult
runCoverGlobalFilterBMCRaw(const CoverGlobalFilterProbeConfig &cfg,
                           StringRef logPath, std::string &error) {
  ProbeRawResult result;
  if (cfg.globalFilterBMC.empty()) {
    result.state = "error";
    result.rc = -1;
    return result;
  }

  SmallString<256> origLog(logPath);
  origLog += ".orig";
  SmallString<256> mutantLog(logPath);
  mutantLog += ".mutant";

  SmallVector<std::string, 16> commonCmd;
  commonCmd.push_back(cfg.globalFilterBMC);
  commonCmd.push_back("-b");
  commonCmd.push_back(std::to_string(cfg.globalFilterBMCBound));
  commonCmd.push_back((Twine("--module=") + cfg.globalFilterBMCModule).str());
  commonCmd.push_back((Twine("--ignore-asserts-until=") +
                       Twine(cfg.globalFilterBMCIgnoreAssertsUntil))
                          .str());
  if (cfg.globalFilterBMCRunSmtlib)
    commonCmd.push_back("--run-smtlib");
  if (!cfg.globalFilterBMCZ3.empty())
    commonCmd.push_back((Twine("--z3-path=") + cfg.globalFilterBMCZ3).str());
  if (cfg.globalFilterBMCAssumeKnownInputs)
    commonCmd.push_back("--assume-known-inputs");
  appendSplitArgs(cfg.globalFilterBmcToolArgs, commonCmd);

  SmallVector<std::string, 20> origCmd(commonCmd.begin(), commonCmd.end());
  origCmd.push_back(cfg.design);
  int origRC = -1;
  if (!runArgvToLog(origCmd, origLog, cfg.globalFilterBMCTimeoutSeconds, origRC,
                    error))
    return ProbeRawResult{"error", origRC};

  SmallVector<std::string, 20> mutantCmd(commonCmd.begin(), commonCmd.end());
  mutantCmd.push_back(cfg.mutantDesign);
  int mutantRC = -1;
  if (!runArgvToLog(mutantCmd, mutantLog, cfg.globalFilterBMCTimeoutSeconds,
                    mutantRC, error))
    return ProbeRawResult{"error", mutantRC};

  std::string origText = readTextFileOrEmpty(origLog);
  std::string mutantText = readTextFileOrEmpty(mutantLog);
  {
    std::error_code ec;
    raw_fd_ostream merged(logPath, ec, sys::fs::OF_Text);
    if (ec) {
      error = (Twine("circt-mut cover: failed to write probe log: ") + logPath +
               ": " + ec.message())
                  .str();
      return ProbeRawResult{"error", -1};
    }
    merged << "# bmc_probe_orig_rc=" << origRC << "\n";
    merged << origText << "\n";
    merged << "# bmc_probe_mutant_rc=" << mutantRC << "\n";
    merged << mutantText << "\n";
  }

  std::string origResult = extractLastBMCResultToken(origText);
  std::string mutantResult = extractLastBMCResultToken(mutantText);

  int finalRC = mutantRC;
  if (finalRC == 0)
    finalRC = origRC;

  if (origResult.empty() && isTimeoutExitCode(origRC))
    return ProbeRawResult{"timeout", finalRC};
  if (mutantResult.empty() && isTimeoutExitCode(mutantRC))
    return ProbeRawResult{"timeout", finalRC};
  if (origResult == "unknown" || mutantResult == "unknown")
    return ProbeRawResult{"unknown", finalRC};
  if (origResult.empty() || mutantResult.empty())
    return ProbeRawResult{"error", finalRC};
  if (origResult == mutantResult)
    return ProbeRawResult{"equal", finalRC};
  return ProbeRawResult{"different", finalRC};
}

static std::string classifyChainConsensusLike(StringRef lecState,
                                              StringRef bmcState) {
  if (lecState == "eq" && bmcState == "equal")
    return "not_propagated";
  if (lecState == "neq" || lecState == "unknown" || lecState == "timeout" ||
      bmcState == "different" || bmcState == "unknown" ||
      bmcState == "timeout")
    return "propagated";
  if (lecState == "error" && bmcState == "equal")
    return "propagated";
  if (bmcState == "error" && lecState == "eq")
    return "propagated";
  return "error";
}

static bool parseCoverGlobalFilterProbeConfig(
    const CoverRewriteResult &rewrite, CoverGlobalFilterProbeConfig &cfg,
    std::string &error) {
  cfg.mutantDesign = rewrite.nativeGlobalFilterProbeMutant;
  cfg.logFile = rewrite.nativeGlobalFilterProbeLog.empty()
                    ? "global_filter_probe.log"
                    : rewrite.nativeGlobalFilterProbeLog;

  auto parseUIntValue = [&](StringRef value, StringRef flag, uint64_t &out,
                            uint64_t defaultValue) -> bool {
    if (value.empty()) {
      out = defaultValue;
      return true;
    }
    if (value.getAsInteger(10, out)) {
      error = (Twine("circt-mut cover: invalid value for ") + flag + ": " +
               value + " (expected integer).")
                  .str();
      return false;
    }
    return true;
  };

  ArrayRef<std::string> args = rewrite.rewrittenArgs;
  uint64_t globalTimeoutSeconds = 0;
  bool hasGlobalTimeoutSeconds = false;
  bool hasLECTimeoutOverride = false;
  bool hasBMCTimeoutOverride = false;
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    auto consumeValue = [&](StringRef optName,
                            std::string &outValue) -> bool {
      std::string withEq = (optName + "=").str();
      if (arg.starts_with(withEq)) {
        outValue = arg.substr(withEq.size()).str();
        return true;
      }
      if (arg == optName) {
        if (i + 1 >= args.size()) {
          error = (Twine("circt-mut cover: missing value for ") + optName).str();
          return false;
        }
        outValue = args[++i];
        return true;
      }
      return true;
    };

    if (arg == "--design" || arg.starts_with("--design=")) {
      if (!consumeValue("--design", cfg.design))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-lec" ||
        arg.starts_with("--formal-global-propagate-circt-lec=")) {
      if (!consumeValue("--formal-global-propagate-circt-lec",
                        cfg.globalFilterLEC))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-bmc" ||
        arg.starts_with("--formal-global-propagate-circt-bmc=")) {
      if (!consumeValue("--formal-global-propagate-circt-bmc",
                        cfg.globalFilterBMC))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-chain" ||
        arg.starts_with("--formal-global-propagate-circt-chain=")) {
      if (!consumeValue("--formal-global-propagate-circt-chain",
                        cfg.globalFilterChain))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-cmd" ||
        arg.starts_with("--formal-global-propagate-cmd=")) {
      if (!consumeValue("--formal-global-propagate-cmd", cfg.globalFilterCmd))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-lec-args" ||
        arg.starts_with("--formal-global-propagate-circt-lec-args=")) {
      if (!consumeValue("--formal-global-propagate-circt-lec-args",
                        cfg.globalFilterLECToolArgs))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-circt-bmc-args" ||
        arg.starts_with("--formal-global-propagate-circt-bmc-args=")) {
      if (!consumeValue("--formal-global-propagate-circt-bmc-args",
                        cfg.globalFilterBmcToolArgs))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-z3" ||
        arg.starts_with("--formal-global-propagate-z3=")) {
      if (!consumeValue("--formal-global-propagate-z3", cfg.globalFilterZ3))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-z3" ||
        arg.starts_with("--formal-global-propagate-bmc-z3=")) {
      if (!consumeValue("--formal-global-propagate-bmc-z3",
                        cfg.globalFilterBMCZ3))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-c1" ||
        arg.starts_with("--formal-global-propagate-c1=")) {
      if (!consumeValue("--formal-global-propagate-c1", cfg.globalFilterC1))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-c2" ||
        arg.starts_with("--formal-global-propagate-c2=")) {
      if (!consumeValue("--formal-global-propagate-c2", cfg.globalFilterC2))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-timeout-seconds=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-timeout-seconds", raw))
        return false;
      uint64_t parsed = 0;
      if (!parseUIntValue(raw, "--formal-global-propagate-timeout-seconds",
                          parsed, 0))
        return false;
      globalTimeoutSeconds = parsed;
      hasGlobalTimeoutSeconds = true;
      continue;
    }
    if (arg == "--formal-global-propagate-lec-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-lec-timeout-seconds=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-lec-timeout-seconds", raw))
        return false;
      if (!parseUIntValue(raw, "--formal-global-propagate-lec-timeout-seconds",
                          cfg.globalFilterLECTimeoutSeconds, 0))
        return false;
      hasLECTimeoutOverride = true;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-bmc-timeout-seconds=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-bmc-timeout-seconds", raw))
        return false;
      if (!parseUIntValue(raw, "--formal-global-propagate-bmc-timeout-seconds",
                          cfg.globalFilterBMCTimeoutSeconds, 0))
        return false;
      hasBMCTimeoutOverride = true;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-bound" ||
        arg.starts_with("--formal-global-propagate-bmc-bound=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-bmc-bound", raw))
        return false;
      if (!parseUIntValue(raw, "--formal-global-propagate-bmc-bound",
                          cfg.globalFilterBMCBound, 20))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-module" ||
        arg.starts_with("--formal-global-propagate-bmc-module=")) {
      if (!consumeValue("--formal-global-propagate-bmc-module",
                        cfg.globalFilterBMCModule))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-ignore-asserts-until" ||
        arg.starts_with("--formal-global-propagate-bmc-ignore-asserts-until=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-bmc-ignore-asserts-until",
                        raw))
        return false;
      if (!parseUIntValue(raw,
                          "--formal-global-propagate-bmc-ignore-asserts-until",
                          cfg.globalFilterBMCIgnoreAssertsUntil, 0))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-assume-known-inputs") {
      cfg.globalFilterAssumeKnownInputs = true;
      continue;
    }
    if (arg == "--formal-global-propagate-accept-xprop-only") {
      cfg.globalFilterAcceptXpropOnly = true;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-run-smtlib") {
      cfg.globalFilterBMCRunSmtlib = true;
      continue;
    }
    if (arg == "--formal-global-propagate-bmc-assume-known-inputs") {
      cfg.globalFilterBMCAssumeKnownInputs = true;
      continue;
    }
  }

  if (cfg.design.empty()) {
    error = "circt-mut cover: --native-global-filter-probe-mutant requires "
            "--design.";
    return false;
  }
  if (cfg.globalFilterChain.empty()) {
    if (cfg.globalFilterCmd.empty() && cfg.globalFilterLEC.empty() &&
        cfg.globalFilterBMC.empty()) {
      error = "circt-mut cover: no global filter configured for probe.";
      return false;
    }
  } else if (cfg.globalFilterChain != "lec-then-bmc" &&
             cfg.globalFilterChain != "bmc-then-lec" &&
             cfg.globalFilterChain != "consensus" &&
             cfg.globalFilterChain != "auto") {
    error = (Twine("circt-mut cover: invalid chain mode for probe: ") +
             cfg.globalFilterChain)
                .str();
    return false;
  }
  if (hasGlobalTimeoutSeconds && !hasLECTimeoutOverride)
    cfg.globalFilterLECTimeoutSeconds = globalTimeoutSeconds;
  if (hasGlobalTimeoutSeconds && !hasBMCTimeoutOverride)
    cfg.globalFilterBMCTimeoutSeconds = globalTimeoutSeconds;
  if (hasGlobalTimeoutSeconds)
    cfg.globalFilterCmdTimeoutSeconds = globalTimeoutSeconds;
  return true;
}

static bool executeNativeCoverGlobalFilterProbe(
    const CoverGlobalFilterProbeConfig &cfg,
    CoverGlobalFilterProbeOutcome &outcome, std::string &error) {
  std::string classification;
  int finalRC = -1;
  std::string cmdSource;

  auto classifyLEC = [&](StringRef state) {
    if (state == "eq")
      classification = "not_propagated";
    else if (state == "neq" || state == "unknown" || state == "timeout")
      classification = "propagated";
    else
      classification = "error";
  };
  auto classifyBMC = [&](StringRef state) {
    if (state == "equal")
      classification = "not_propagated";
    else if (state == "different" || state == "unknown" || state == "timeout")
      classification = "propagated";
    else
      classification = "error";
  };

  if (!cfg.globalFilterCmd.empty()) {
    ProbeRawResult cmdRaw =
        runCoverGlobalFilterCmdRaw(/*runDir=*/".", cfg.globalFilterCmd,
                                   cfg.logFile, cfg.globalFilterCmdTimeoutSeconds,
                                   cfg.design, cfg.mutantDesign,
                                   /*mutationID=*/"-", /*mutationSpec=*/"-",
                                   error);
    if (!error.empty())
      return false;
    if (cmdRaw.state == "not_propagated")
      classification = "not_propagated";
    else if (cmdRaw.state == "propagated")
      classification = "propagated";
    else
      classification = "error";
    finalRC = cmdRaw.rc;
    cmdSource = cmdRaw.source;
  } else if (!cfg.globalFilterChain.empty()) {
    if (cfg.globalFilterLEC.empty() || cfg.globalFilterBMC.empty()) {
      error = "circt-mut cover: probe chain mode requires both "
              "--formal-global-propagate-circt-lec and "
              "--formal-global-propagate-circt-bmc.";
      return false;
    }
    ProbeRawResult lecRaw =
        runCoverGlobalFilterLECRaw(cfg, cfg.logFile + ".lec", error);
    if (!error.empty())
      return false;
    ProbeRawResult bmcRaw =
        runCoverGlobalFilterBMCRaw(cfg, cfg.logFile + ".bmc", error);
    if (!error.empty())
      return false;
    finalRC = bmcRaw.rc;
    if (finalRC == 0)
      finalRC = lecRaw.rc;

    if (cfg.globalFilterChain == "lec-then-bmc") {
      if (lecRaw.state == "eq")
        classification = "not_propagated";
      else if (lecRaw.state == "neq")
        classification = "propagated";
      else if (lecRaw.state == "error") {
        if (bmcRaw.state == "equal" || bmcRaw.state == "different" ||
            bmcRaw.state == "unknown" || bmcRaw.state == "timeout")
          classification = "propagated";
        else
          classification = "error";
      } else if (bmcRaw.state == "equal")
        classification = "not_propagated";
      else if (bmcRaw.state == "different" || bmcRaw.state == "unknown" ||
               bmcRaw.state == "timeout")
        classification = "propagated";
      else
        classification = "error";
    } else if (cfg.globalFilterChain == "bmc-then-lec") {
      if (bmcRaw.state == "equal")
        classification = "not_propagated";
      else if (bmcRaw.state == "different")
        classification = "propagated";
      else if (bmcRaw.state == "error") {
        if (lecRaw.state == "eq" || lecRaw.state == "neq" ||
            lecRaw.state == "unknown" || lecRaw.state == "timeout")
          classification = "propagated";
        else
          classification = "error";
      } else if (lecRaw.state == "eq")
        classification = "not_propagated";
      else if (lecRaw.state == "neq" || lecRaw.state == "unknown" ||
               lecRaw.state == "timeout")
        classification = "propagated";
      else
        classification = "error";
    } else {
      classification = classifyChainConsensusLike(lecRaw.state, bmcRaw.state);
    }

    std::error_code ec;
    raw_fd_ostream out(cfg.logFile, ec, sys::fs::OF_Text);
    if (ec) {
      error = (Twine("circt-mut cover: failed to write probe log: ") +
               cfg.logFile + ": " + ec.message())
                  .str();
      return false;
    }
    out << "# chain_mode=" << cfg.globalFilterChain << " lec_state="
        << lecRaw.state << " lec_rc=" << lecRaw.rc << "\n";
    out << readTextFileOrEmpty(cfg.logFile + ".lec");
    out << "\n# chain_bmc_state=" << bmcRaw.state << " bmc_rc=" << bmcRaw.rc
        << "\n";
    out << readTextFileOrEmpty(cfg.logFile + ".bmc");
    out << "\n";
    sys::fs::remove(cfg.logFile + ".lec");
    sys::fs::remove(cfg.logFile + ".bmc");
    sys::fs::remove(cfg.logFile + ".bmc.orig");
    sys::fs::remove(cfg.logFile + ".bmc.mutant");
  } else if (!cfg.globalFilterLEC.empty()) {
    ProbeRawResult lecRaw = runCoverGlobalFilterLECRaw(cfg, cfg.logFile, error);
    if (!error.empty())
      return false;
    classifyLEC(lecRaw.state);
    finalRC = lecRaw.rc;
  } else if (!cfg.globalFilterBMC.empty()) {
    ProbeRawResult bmcRaw = runCoverGlobalFilterBMCRaw(cfg, cfg.logFile, error);
    if (!error.empty())
      return false;
    classifyBMC(bmcRaw.state);
    finalRC = bmcRaw.rc;
    sys::fs::remove(cfg.logFile + ".orig");
    sys::fs::remove(cfg.logFile + ".mutant");
  } else {
    error = "circt-mut cover: no global filter configured for native probe.";
    return false;
  }

  outcome.classification = classification;
  outcome.finalRC = finalRC;
  outcome.cmdSource = cmdSource;
  return true;
}

static int runNativeCoverGlobalFilterProbe(const CoverRewriteResult &rewrite) {
  CoverGlobalFilterProbeConfig cfg;
  std::string error;
  if (!parseCoverGlobalFilterProbeConfig(rewrite, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  if (!ensureParentDirForFile(cfg.logFile, error)) {
    errs() << error << "\n";
    return 1;
  }

  CoverGlobalFilterProbeOutcome outcome;
  if (!executeNativeCoverGlobalFilterProbe(cfg, outcome, error)) {
    errs() << error << "\n";
    return 1;
  }

  outs() << "classification\t" << outcome.classification << "\n";
  outs() << "global_filter_rc\t" << outcome.finalRC << "\n";
  if (!outcome.cmdSource.empty())
    outs() << "global_filter_cmd_source\t" << outcome.cmdSource << "\n";
  outs() << "global_filter_log\t" << cfg.logFile << "\n";
  return outcome.classification == "error" ? 1 : 0;
}

struct MutationRow {
  std::string id;
  std::string spec;
};

struct CoverNativePrequalifyConfig {
  CoverGlobalFilterProbeConfig probeCfg;
  bool useGlobalFilterCmd = false;
  std::string globalFilterCmd;
  std::string design;
  uint64_t globalFilterTimeoutSeconds = 0;
  std::string mutationsFile;
  bool useGeneratedMutations = false;
  uint64_t generateMutations = 0;
  uint64_t mutationsSeed = 1;
  std::string mutationsTop;
  std::string mutationsYosys;
  std::string mutationsModes;
  std::string mutationsModeCounts;
  std::string mutationsModeWeights;
  std::string mutationsProfiles;
  std::string mutationsCfg;
  std::string mutationsSelect;
  std::string reuseCacheDir;
  std::string reuseCacheMode = "read-write";
  std::string workDir = "mutation-cover-results";
  std::string createMutatedScript;
  std::string mutantFormat = "il";
  uint64_t jobs = 1;
  uint64_t mutationLimit = 0;
  std::string pairFile;
  std::string generateLogFile;
};

static bool parseMutationRowsForPrequalify(StringRef mutationsFile,
                                           uint64_t mutationLimit,
                                           std::vector<MutationRow> &rows,
                                           std::string &error) {
  auto bufferOrErr = MemoryBuffer::getFile(mutationsFile);
  if (!bufferOrErr) {
    error = (Twine("circt-mut cover: unable to read --mutations-file: ") +
             mutationsFile)
                .str();
    return false;
  }
  StringRef contents = bufferOrErr.get()->getBuffer();
  SmallVector<StringRef, 256> lines;
  contents.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef rawLine : lines) {
    StringRef line = rawLine.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    StringRef id = line.take_until([](char c) { return isspace(c); }).trim();
    StringRef rest = line.drop_front(id.size()).trim();
    if (id.empty() || rest.empty()) {
      error = (Twine("circt-mut cover: malformed mutation line in ") +
               mutationsFile + ": '" + line + "'")
                  .str();
      return false;
    }
    rows.push_back(MutationRow{id.str(), rest.str()});
    if (mutationLimit > 0 && rows.size() >= mutationLimit)
      break;
  }
  if (rows.empty()) {
    error = (Twine("circt-mut cover: no usable mutations found in ") +
             mutationsFile)
                .str();
    return false;
  }
  return true;
}

static bool isSupportedMutantFormat(StringRef fmt) {
  return fmt == "il" || fmt == "v" || fmt == "sv";
}

static std::string joinPath2(StringRef a, StringRef b) {
  SmallString<256> path(a);
  sys::path::append(path, b);
  return std::string(path.str());
}

static bool parseCoverNativePrequalifyConfig(
    const char *argv0, const CoverRewriteResult &rewrite,
    CoverNativePrequalifyConfig &cfg, std::string &error) {
  auto parseUIntArg = [&](StringRef value, StringRef flag, uint64_t &out) {
    if (value.getAsInteger(10, out)) {
      error = (Twine("circt-mut cover: invalid value for ") + flag + ": " +
               value + " (expected integer).")
                  .str();
      return false;
    }
    return true;
  };

  ArrayRef<std::string> args = rewrite.rewrittenArgs;
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    auto consumeValue = [&](StringRef optName,
                            std::string &outValue) -> bool {
      std::string withEq = (optName + "=").str();
      if (arg.starts_with(withEq)) {
        outValue = arg.substr(withEq.size()).str();
        return true;
      }
      if (arg == optName) {
        if (i + 1 >= args.size()) {
          error = (Twine("circt-mut cover: missing value for ") + optName).str();
          return false;
        }
        outValue = args[++i];
        return true;
      }
      return true;
    };

    if (arg == "--mutations-file" || arg.starts_with("--mutations-file=")) {
      if (!consumeValue("--mutations-file", cfg.mutationsFile))
        return false;
      continue;
    }
    if (arg == "--design" || arg.starts_with("--design=")) {
      if (!consumeValue("--design", cfg.design))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-cmd" ||
        arg.starts_with("--formal-global-propagate-cmd=")) {
      if (!consumeValue("--formal-global-propagate-cmd", cfg.globalFilterCmd))
        return false;
      continue;
    }
    if (arg == "--formal-global-propagate-timeout-seconds" ||
        arg.starts_with("--formal-global-propagate-timeout-seconds=")) {
      std::string raw;
      if (!consumeValue("--formal-global-propagate-timeout-seconds", raw))
        return false;
      if (!parseUIntArg(raw, "--formal-global-propagate-timeout-seconds",
                        cfg.globalFilterTimeoutSeconds))
        return false;
      continue;
    }
    if (arg == "--generate-mutations" ||
        arg.starts_with("--generate-mutations=")) {
      std::string raw;
      if (!consumeValue("--generate-mutations", raw))
        return false;
      if (!parseUIntArg(raw, "--generate-mutations", cfg.generateMutations))
        return false;
      continue;
    }
    if (arg == "--mutations-top" || arg.starts_with("--mutations-top=")) {
      if (!consumeValue("--mutations-top", cfg.mutationsTop))
        return false;
      continue;
    }
    if (arg == "--mutations-seed" || arg.starts_with("--mutations-seed=")) {
      std::string raw;
      if (!consumeValue("--mutations-seed", raw))
        return false;
      if (!parseUIntArg(raw, "--mutations-seed", cfg.mutationsSeed))
        return false;
      continue;
    }
    if (arg == "--mutations-yosys" || arg.starts_with("--mutations-yosys=")) {
      if (!consumeValue("--mutations-yosys", cfg.mutationsYosys))
        return false;
      continue;
    }
    if (arg == "--mutations-modes" || arg.starts_with("--mutations-modes=")) {
      if (!consumeValue("--mutations-modes", cfg.mutationsModes))
        return false;
      continue;
    }
    if (arg == "--mutations-mode-counts" ||
        arg.starts_with("--mutations-mode-counts=")) {
      if (!consumeValue("--mutations-mode-counts", cfg.mutationsModeCounts))
        return false;
      continue;
    }
    if (arg == "--mutations-mode-weights" ||
        arg.starts_with("--mutations-mode-weights=")) {
      if (!consumeValue("--mutations-mode-weights", cfg.mutationsModeWeights))
        return false;
      continue;
    }
    if (arg == "--mutations-profiles" ||
        arg.starts_with("--mutations-profiles=")) {
      if (!consumeValue("--mutations-profiles", cfg.mutationsProfiles))
        return false;
      continue;
    }
    if (arg == "--mutations-cfg" || arg.starts_with("--mutations-cfg=")) {
      if (!consumeValue("--mutations-cfg", cfg.mutationsCfg))
        return false;
      continue;
    }
    if (arg == "--mutations-select" ||
        arg.starts_with("--mutations-select=")) {
      if (!consumeValue("--mutations-select", cfg.mutationsSelect))
        return false;
      continue;
    }
    if (arg == "--reuse-cache-dir" || arg.starts_with("--reuse-cache-dir=")) {
      if (!consumeValue("--reuse-cache-dir", cfg.reuseCacheDir))
        return false;
      continue;
    }
    if (arg == "--reuse-cache-mode" || arg.starts_with("--reuse-cache-mode=")) {
      if (!consumeValue("--reuse-cache-mode", cfg.reuseCacheMode))
        return false;
      continue;
    }
    if (arg == "--work-dir" || arg.starts_with("--work-dir=")) {
      if (!consumeValue("--work-dir", cfg.workDir))
        return false;
      continue;
    }
    if (arg == "--jobs" || arg.starts_with("--jobs=")) {
      std::string raw;
      if (!consumeValue("--jobs", raw))
        return false;
      if (!parseUIntArg(raw, "--jobs", cfg.jobs))
        return false;
      continue;
    }
    if (arg == "--create-mutated-script" ||
        arg.starts_with("--create-mutated-script=")) {
      if (!consumeValue("--create-mutated-script", cfg.createMutatedScript))
        return false;
      continue;
    }
    if (arg == "--mutant-format" || arg.starts_with("--mutant-format=")) {
      if (!consumeValue("--mutant-format", cfg.mutantFormat))
        return false;
      continue;
    }
    if (arg == "--mutation-limit" || arg.starts_with("--mutation-limit=")) {
      std::string raw;
      if (!consumeValue("--mutation-limit", raw))
        return false;
      if (!parseUIntArg(raw, "--mutation-limit", cfg.mutationLimit))
        return false;
      continue;
    }
  }

  if (!cfg.mutationsFile.empty() && cfg.generateMutations > 0) {
    error = "circt-mut cover: --native-global-filter-prequalify requires "
            "exactly one mutation source (--mutations-file or "
            "--generate-mutations).";
    return false;
  }
  if (cfg.mutationsFile.empty() && cfg.generateMutations == 0) {
    error = "circt-mut cover: --native-global-filter-prequalify requires "
            "either --mutations-file or --generate-mutations.";
    return false;
  }
  cfg.useGeneratedMutations = cfg.generateMutations > 0;
  if (!isSupportedMutantFormat(cfg.mutantFormat)) {
    error = (Twine("circt-mut cover: unsupported --mutant-format value for "
                   "native prequalification: ") +
             cfg.mutantFormat + " (expected il|v|sv).")
                .str();
    return false;
  }
  if (cfg.jobs == 0) {
    error = "circt-mut cover: invalid --jobs value for native prequalification: "
            "0 (expected positive integer).";
    return false;
  }

  if (cfg.createMutatedScript.empty()) {
    auto defaultScript = resolveScriptPath(argv0, "create_mutated_yosys.sh");
    if (!defaultScript) {
      error = "circt-mut cover: unable to locate default "
              "'create_mutated_yosys.sh' for native prequalification.";
      return false;
    }
    cfg.createMutatedScript = *defaultScript;
  } else {
    auto resolvedCreateScript = resolveToolPath(cfg.createMutatedScript);
    if (!resolvedCreateScript) {
      error = (Twine("circt-mut cover: unable to resolve --create-mutated-script "
                     "for native prequalification: ") +
               cfg.createMutatedScript)
                  .str();
      return false;
    }
    cfg.createMutatedScript = *resolvedCreateScript;
  }

  if (!rewrite.nativeGlobalFilterPrequalifyPairFile.empty()) {
    cfg.pairFile = rewrite.nativeGlobalFilterPrequalifyPairFile;
  } else {
    cfg.pairFile = joinPath2(cfg.workDir, "native_global_filter_prequalify.tsv");
  }
  if (cfg.useGeneratedMutations) {
    cfg.mutationsFile = joinPath2(cfg.workDir, "generated_mutations.txt");
    cfg.generateLogFile = joinPath2(cfg.workDir, "generate_mutations.log");
  }

  cfg.useGlobalFilterCmd = !cfg.globalFilterCmd.empty();
  if (cfg.useGlobalFilterCmd) {
    if (cfg.design.empty()) {
      error = "circt-mut cover: --native-global-filter-prequalify requires "
              "--design when using --formal-global-propagate-cmd.";
      return false;
    }
    return true;
  }

  CoverRewriteResult probeRewrite = rewrite;
  probeRewrite.nativeGlobalFilterProbeMutant = "__native_prequalify_dummy__";
  probeRewrite.nativeGlobalFilterProbeLog = "__native_prequalify_dummy__.log";
  if (!parseCoverGlobalFilterProbeConfig(probeRewrite, cfg.probeCfg, error))
    return false;
  cfg.probeCfg.mutantDesign.clear();
  cfg.probeCfg.logFile.clear();
  if (cfg.design.empty())
    cfg.design = cfg.probeCfg.design;

  return true;
}

static bool runNativeGenerateForPrequalify(const char *argv0,
                                           const CoverNativePrequalifyConfig &cfg,
                                           std::string &error) {
  if (!cfg.useGeneratedMutations)
    return true;

  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (mainExec.empty()) {
    error = "circt-mut cover: unable to locate circt-mut executable for native "
            "prequalification generation.";
    return false;
  }

  if (!ensureParentDirForFile(cfg.mutationsFile, error))
    return false;
  if (!ensureParentDirForFile(cfg.generateLogFile, error))
    return false;

  SmallVector<std::string, 32> genCmd;
  genCmd.push_back(mainExec);
  genCmd.push_back("generate");
  genCmd.push_back("--design");
  genCmd.push_back(cfg.design);
  genCmd.push_back("--out");
  genCmd.push_back(cfg.mutationsFile);
  genCmd.push_back("--count");
  genCmd.push_back(std::to_string(cfg.generateMutations));
  genCmd.push_back("--seed");
  genCmd.push_back(std::to_string(cfg.mutationsSeed));
  if (!cfg.mutationsTop.empty()) {
    genCmd.push_back("--top");
    genCmd.push_back(cfg.mutationsTop);
  }
  if (!cfg.mutationsYosys.empty()) {
    genCmd.push_back("--yosys");
    genCmd.push_back(cfg.mutationsYosys);
  }
  if (!cfg.mutationsModes.empty()) {
    genCmd.push_back("--modes");
    genCmd.push_back(cfg.mutationsModes);
  }
  if (!cfg.mutationsModeCounts.empty()) {
    genCmd.push_back("--mode-counts");
    genCmd.push_back(cfg.mutationsModeCounts);
  }
  if (!cfg.mutationsModeWeights.empty()) {
    genCmd.push_back("--mode-weights");
    genCmd.push_back(cfg.mutationsModeWeights);
  }
  if (!cfg.mutationsProfiles.empty()) {
    genCmd.push_back("--profiles");
    genCmd.push_back(cfg.mutationsProfiles);
  }
  if (!cfg.mutationsCfg.empty()) {
    genCmd.push_back("--cfgs");
    genCmd.push_back(cfg.mutationsCfg);
  }
  if (!cfg.mutationsSelect.empty()) {
    genCmd.push_back("--selects");
    genCmd.push_back(cfg.mutationsSelect);
  }
  if (!cfg.reuseCacheDir.empty() && cfg.reuseCacheMode != "off") {
    genCmd.push_back("--cache-dir");
    genCmd.push_back(joinPath2(cfg.reuseCacheDir, "generated_mutations"));
  }

  int rc = -1;
  if (!runArgvToLog(genCmd, cfg.generateLogFile, /*timeoutSeconds=*/0, rc, error))
    return false;
  if (rc != 0) {
    error = (Twine("circt-mut cover: native prequalification generation failed "
                   "(exit=") +
             Twine(rc) + "), see log: " + cfg.generateLogFile)
                .str();
    return false;
  }
  return true;
}

static SmallVector<std::string, 64>
rewriteCoverArgsForPrequalifyDispatch(const CoverRewriteResult &rewrite,
                                      const CoverNativePrequalifyConfig &cfg) {
  auto shouldDropGenerateArg = [&](StringRef arg, bool &consumeNext) -> bool {
    auto match = [&](StringRef opt) -> bool {
      std::string withEq = (opt + "=").str();
      if (arg == opt) {
        consumeNext = true;
        return true;
      }
      return arg.starts_with(withEq);
    };
    return match("--mutations-file") || match("--generate-mutations") ||
           match("--mutations-top") || match("--mutations-seed") ||
           match("--mutations-yosys") || match("--mutations-modes") ||
           match("--mutations-mode-counts") ||
           match("--mutations-mode-weights") || match("--mutations-profiles") ||
           match("--mutations-cfg") || match("--mutations-select");
  };

  SmallVector<std::string, 64> rewrittenArgs;
  if (cfg.useGeneratedMutations) {
    for (size_t i = 0; i < rewrite.rewrittenArgs.size(); ++i) {
      StringRef arg = rewrite.rewrittenArgs[i];
      bool consumeNext = false;
      if (shouldDropGenerateArg(arg, consumeNext)) {
        if (consumeNext && i + 1 < rewrite.rewrittenArgs.size())
          ++i;
        continue;
      }
      rewrittenArgs.push_back(arg.str());
    }
    rewrittenArgs.push_back("--mutations-file");
    rewrittenArgs.push_back(cfg.mutationsFile);
  } else {
    rewrittenArgs.append(rewrite.rewrittenArgs.begin(),
                         rewrite.rewrittenArgs.end());
  }
  return rewrittenArgs;
}

static int runNativeCoverGlobalFilterPrequalifyAndDispatch(
    const char *argv0, const CoverRewriteResult &rewrite) {
  CoverNativePrequalifyConfig cfg;
  std::string error;
  if (!parseCoverNativePrequalifyConfig(argv0, rewrite, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  if (!ensureParentDirForFile(cfg.pairFile, error)) {
    errs() << error << "\n";
    return 1;
  }
  if (!runNativeGenerateForPrequalify(argv0, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  std::vector<MutationRow> rows;
  if (!parseMutationRowsForPrequalify(cfg.mutationsFile, cfg.mutationLimit, rows,
                                      error)) {
    errs() << error << "\n";
    return 1;
  }

  std::string prequalifyRoot = joinPath2(cfg.workDir, "native_global_filter_prequalify");
  std::error_code mkdirEC = sys::fs::create_directories(prequalifyRoot);
  if (mkdirEC) {
    errs() << "circt-mut cover: failed to create native prequalification root: "
           << prequalifyRoot << ": " << mkdirEC.message() << "\n";
    return 1;
  }

  struct PrequalifyRowResult {
    std::string mutationID;
    std::string propagation = "propagated";
    int propagateRC = -1;
    std::string note = "global_filter_propagated;native_prequalify=1";
    std::string cmdSource;
    bool createMutatedError = false;
    bool probeError = false;
  };

  std::vector<PrequalifyRowResult> rowResults(rows.size());
  std::atomic<size_t> nextIndex{0};
  std::atomic<bool> failed{false};
  std::string fatalError;
  std::mutex fatalMutex;
  auto setFatalError = [&](StringRef message) {
    std::lock_guard<std::mutex> lock(fatalMutex);
    if (fatalError.empty())
      fatalError = message.str();
    failed.store(true);
  };

  auto processRow = [&](size_t idx) {
    if (failed.load())
      return;
    const MutationRow &row = rows[idx];
    PrequalifyRowResult result;
    result.mutationID = row.id;

    std::string mutationRoot = joinPath2(prequalifyRoot, row.id);
    std::error_code rowEC = sys::fs::create_directories(mutationRoot);
    if (rowEC) {
      setFatalError((Twine("circt-mut cover: failed to create mutation "
                           "prequalification directory: ") +
                     mutationRoot + ": " + rowEC.message())
                        .str());
      return;
    }
    std::string mutationInput = joinPath2(mutationRoot, "mutation_input.txt");
    std::string mutantDesign =
        joinPath2(mutationRoot, (Twine("mutant.") + cfg.mutantFormat).str());
    std::string createLog = joinPath2(mutationRoot, "create_mutated.log");
    std::string probeLog = joinPath2(mutationRoot, "global_propagate.log");

    {
      std::error_code inputEC;
      raw_fd_ostream inputOut(mutationInput, inputEC, sys::fs::OF_Text);
      if (inputEC) {
        setFatalError((Twine("circt-mut cover: failed to write mutation input: ") +
                       mutationInput + ": " + inputEC.message())
                          .str());
        return;
      }
      inputOut << row.id << " " << row.spec << "\n";
    }

    SmallVector<std::string, 16> createCmd;
    createCmd.push_back(cfg.createMutatedScript);
    createCmd.push_back("-d");
    createCmd.push_back(cfg.design);
    createCmd.push_back("-i");
    createCmd.push_back(mutationInput);
    createCmd.push_back("-o");
    createCmd.push_back(mutantDesign);
    int createRC = -1;
    std::string createError;
    if (!runArgvToLog(createCmd, createLog, /*timeoutSeconds=*/0, createRC,
                      createError)) {
      result.note += ";native_prequalify_create_mutated_exec_error=1";
      result.createMutatedError = true;
    } else if (createRC != 0) {
      result.propagateRC = createRC;
      result.note += ";native_prequalify_create_mutated_error=1";
      result.createMutatedError = true;
    } else {
      if (cfg.useGlobalFilterCmd) {
        std::string probeError;
        ProbeRawResult cmdOutcome = runCoverGlobalFilterCmdRaw(
            mutationRoot, cfg.globalFilterCmd, probeLog,
            cfg.globalFilterTimeoutSeconds, cfg.design, mutantDesign, row.id,
            row.spec, probeError);
        if (!probeError.empty()) {
          result.note += ";native_prequalify_probe_exec_error=1";
          result.probeError = true;
        } else if (cmdOutcome.state == "not_propagated") {
          result.propagation = "not_propagated";
          result.propagateRC = cmdOutcome.rc;
          result.note = "global_filter_not_propagated;native_prequalify=1";
          result.cmdSource = cmdOutcome.source;
          if (!result.cmdSource.empty())
            result.note += ";global_filter_cmd_source=" + result.cmdSource;
        } else if (cmdOutcome.state == "propagated") {
          result.propagation = "propagated";
          result.propagateRC = cmdOutcome.rc;
          result.cmdSource = cmdOutcome.source;
          if (!result.cmdSource.empty())
            result.note += ";global_filter_cmd_source=" + result.cmdSource;
        } else {
          result.propagation = "propagated";
          result.propagateRC = cmdOutcome.rc;
          result.note += ";native_prequalify_probe_error=1";
          result.cmdSource = cmdOutcome.source;
          if (!result.cmdSource.empty())
            result.note += ";global_filter_cmd_source=" + result.cmdSource;
          result.probeError = true;
        }
      } else {
        CoverGlobalFilterProbeConfig probeCfg = cfg.probeCfg;
        probeCfg.mutantDesign = mutantDesign;
        probeCfg.logFile = probeLog;
        CoverGlobalFilterProbeOutcome outcome;
        std::string probeError;
        if (!executeNativeCoverGlobalFilterProbe(probeCfg, outcome,
                                                 probeError)) {
          result.note += ";native_prequalify_probe_exec_error=1";
          result.probeError = true;
        } else if (outcome.classification == "not_propagated") {
          result.propagation = "not_propagated";
          result.propagateRC = outcome.finalRC;
          result.note = "global_filter_not_propagated;native_prequalify=1";
        } else if (outcome.classification == "propagated") {
          result.propagation = "propagated";
          result.propagateRC = outcome.finalRC;
        } else {
          result.propagation = "propagated";
          result.propagateRC = outcome.finalRC;
          result.note += ";native_prequalify_probe_error=1";
          result.probeError = true;
        }
      }
    }

    rowResults[idx] = std::move(result);
  };

  auto worker = [&]() {
    while (true) {
      size_t idx = nextIndex.fetch_add(1);
      if (idx >= rows.size())
        return;
      processRow(idx);
      if (failed.load())
        return;
    }
  };

  size_t workerCount =
      std::min<size_t>(static_cast<size_t>(cfg.jobs), rows.size());
  if (workerCount == 0)
    workerCount = 1;
  if (workerCount == 1) {
    worker();
  } else {
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i)
      workers.emplace_back(worker);
    for (std::thread &thread : workers)
      thread.join();
  }

  if (failed.load()) {
    errs() << fatalError << "\n";
    return 1;
  }

  uint64_t prequalifyTotalMutants = rows.size();
  uint64_t prequalifyNotPropagatedMutants = 0;
  uint64_t prequalifyPropagatedMutants = 0;
  uint64_t prequalifyCreateMutatedErrorMutants = 0;
  uint64_t prequalifyProbeErrorMutants = 0;
  uint64_t prequalifyCmdTokenNotPropagatedMutants = 0;
  uint64_t prequalifyCmdTokenPropagatedMutants = 0;
  uint64_t prequalifyCmdRCNotPropagatedMutants = 0;
  uint64_t prequalifyCmdRCPropagatedMutants = 0;
  uint64_t prequalifyCmdTimeoutPropagatedMutants = 0;
  uint64_t prequalifyCmdErrorMutants = 0;
  for (const PrequalifyRowResult &result : rowResults) {
    if (result.propagation == "not_propagated")
      ++prequalifyNotPropagatedMutants;
    else
      ++prequalifyPropagatedMutants;
    if (result.createMutatedError)
      ++prequalifyCreateMutatedErrorMutants;
    if (result.probeError)
      ++prequalifyProbeErrorMutants;
    if (result.cmdSource == "token_not_propagated")
      ++prequalifyCmdTokenNotPropagatedMutants;
    else if (result.cmdSource == "token_propagated")
      ++prequalifyCmdTokenPropagatedMutants;
    else if (result.cmdSource == "rc0")
      ++prequalifyCmdRCNotPropagatedMutants;
    else if (result.cmdSource == "rc1")
      ++prequalifyCmdRCPropagatedMutants;
    else if (result.cmdSource == "timeout")
      ++prequalifyCmdTimeoutPropagatedMutants;
    else if (result.cmdSource == "error" || result.cmdSource == "exec_error")
      ++prequalifyCmdErrorMutants;
  }

  std::error_code pairEC;
  raw_fd_ostream pairOut(cfg.pairFile, pairEC, sys::fs::OF_Text);
  if (pairEC) {
    errs() << "circt-mut cover: failed to open prequalification pair file: "
           << cfg.pairFile << ": " << pairEC.message() << "\n";
    return 1;
  }
  pairOut
      << "mutation_id\ttest_id\tactivation\tpropagation\tactivate_exit\t"
         "propagate_exit\tnote\n";
  for (const PrequalifyRowResult &result : rowResults) {
    pairOut << result.mutationID << "\t-\tactivated\t" << result.propagation
            << "\t-1\t" << result.propagateRC << "\t" << result.note << "\n";
  }
  pairOut.close();

  if (rewrite.nativeGlobalFilterPrequalifyOnly) {
    outs() << "prequalify_pair_file\t" << cfg.pairFile << "\n";
    outs() << "prequalify_total_mutants\t" << prequalifyTotalMutants << "\n";
    outs() << "prequalify_not_propagated_mutants\t"
           << prequalifyNotPropagatedMutants << "\n";
    outs() << "prequalify_propagated_mutants\t" << prequalifyPropagatedMutants
           << "\n";
    outs() << "prequalify_create_mutated_error_mutants\t"
           << prequalifyCreateMutatedErrorMutants << "\n";
    outs() << "prequalify_probe_error_mutants\t" << prequalifyProbeErrorMutants
           << "\n";
    outs() << "prequalify_cmd_token_not_propagated_mutants\t"
           << prequalifyCmdTokenNotPropagatedMutants << "\n";
    outs() << "prequalify_cmd_token_propagated_mutants\t"
           << prequalifyCmdTokenPropagatedMutants << "\n";
    outs() << "prequalify_cmd_rc_not_propagated_mutants\t"
           << prequalifyCmdRCNotPropagatedMutants << "\n";
    outs() << "prequalify_cmd_rc_propagated_mutants\t"
           << prequalifyCmdRCPropagatedMutants << "\n";
    outs() << "prequalify_cmd_timeout_propagated_mutants\t"
           << prequalifyCmdTimeoutPropagatedMutants << "\n";
    outs() << "prequalify_cmd_error_mutants\t" << prequalifyCmdErrorMutants
           << "\n";
    return 0;
  }

  auto scriptPath = resolveScriptPath(argv0, "run_mutation_cover.sh");
  if (!scriptPath) {
    errs() << "circt-mut: unable to locate script 'run_mutation_cover.sh'.\n";
    errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree with"
              " utils scripts.\n";
    return 1;
  }

  SmallVector<std::string, 64> rewrittenArgs =
      rewriteCoverArgsForPrequalifyDispatch(rewrite, cfg);
  rewrittenArgs.push_back("--reuse-pair-file");
  rewrittenArgs.push_back(cfg.pairFile);

  SmallVector<StringRef, 64> rewrittenArgsRef;
  for (const std::string &arg : rewrittenArgs)
    rewrittenArgsRef.push_back(arg);
  return dispatchToScript(*scriptPath, rewrittenArgsRef);
}

struct MatrixRewriteResult {
  bool ok = false;
  std::string error;
  SmallVector<std::string, 32> rewrittenArgs;
  bool nativeGlobalFilterPrequalify = false;
  bool nativeMatrixDispatch = false;
};

struct MatrixLanePreflightDefaults {
  std::string mutationsModes;
  std::string mutationsSeed;
  std::string mutationsYosys;
  std::string mutationsProfiles;
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
    StringRef laneMutationsModes = getCol(13);
    StringRef laneMutationsSeed = getCol(9);
    StringRef laneMutationsYosys = getCol(10);
    StringRef laneMutationsProfiles = getCol(32);
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
                         "global_propagate_z3", "z3", /*allowAuto=*/true))
      return false;
    if (useBMCFilter &&
        !resolveLaneTool(laneGlobalFilterBMCZ3, defaults.globalFilterBMCZ3,
                         "global_propagate_bmc_z3", "z3", /*allowAuto=*/true))
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

    bool hasMutationsFile = !mutationsFile.empty() && mutationsFile != "-";
    bool hasGenerateCount = !generateCount.empty() && generateCount != "-";
    if (hasMutationsFile && hasGenerateCount) {
      error =
          (Twine("Lane mutation source conflict in --lanes-tsv at line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel +
           "): provide either mutations_file or generate_count, not both.")
              .str();
      return false;
    }
    if (!hasMutationsFile && !hasGenerateCount) {
      error = (Twine("Lane mutation source missing in --lanes-tsv at line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel +
               "): when mutations_file is '-', generate_count is required.")
                  .str();
      return false;
    }

    bool autoGenerateLane = !hasMutationsFile && hasGenerateCount;
    if (!autoGenerateLane)
      continue;

    StringRef effectiveModes =
        withDefault(laneMutationsModes, defaults.mutationsModes);
    if (auto unknown = firstUnknownMutationModeCSV(effectiveModes)) {
      error = (Twine("Unknown lane mutations_modes value in --lanes-tsv at "
                     "line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + *unknown +
               " (expected inv|const0|const1|cnot0|cnot1|"
               "arith|control|balanced|all|stuck|invert|connect).")
                  .str();
      return false;
    }

    StringRef effectiveSeed = withDefault(laneMutationsSeed, defaults.mutationsSeed);
    if (effectiveSeed.empty())
      effectiveSeed = "1";
    if (!Regex("^[0-9]+$").match(effectiveSeed)) {
      error = (Twine("Invalid lane mutations_seed value in --lanes-tsv at "
                     "line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + effectiveSeed +
               " (expected 0-9 integer).")
                  .str();
      return false;
    }

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

    StringRef effectiveProfiles =
        withDefault(laneMutationsProfiles, defaults.mutationsProfiles);
    if (auto unknown = firstUnknownMutationProfile(effectiveProfiles)) {
      error = (Twine("Unknown lane mutations_profiles value in --lanes-tsv at "
                     "line ") +
               Twine(static_cast<unsigned long long>(lineIdx + 1)) +
               " (lane " + laneLabel + "): " + *unknown +
               " (expected arith-depth|control-depth|balanced-depth|"
               "fault-basic|fault-stuck|fault-connect|cover|none).")
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
    if (auto unknown =
            firstUnknownMutationModeInAllocationCSV(effectiveModeCounts)) {
      error =
          (Twine("Unknown lane mutations_mode_counts mode in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + *unknown +
           " (expected inv|const0|const1|cnot0|cnot1|"
           "arith|control|balanced|all|stuck|invert|connect).")
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
    if (auto unknown =
            firstUnknownMutationModeInAllocationCSV(effectiveModeWeights)) {
      error =
          (Twine("Unknown lane mutations_mode_weights mode in --lanes-tsv at "
                 "line ") +
           Twine(static_cast<unsigned long long>(lineIdx + 1)) + " (lane " +
           laneLabel + "): " + *unknown +
           " (expected inv|const0|const1|cnot0|cnot1|"
           "arith|control|balanced|all|stuck|invert|connect).")
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
    if (arg == "--native-global-filter-prequalify") {
      result.nativeGlobalFilterPrequalify = true;
      continue;
    }
    if (arg == "--native-matrix-dispatch") {
      result.nativeMatrixDispatch = true;
      continue;
    }
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
    auto resolveWithRequiredValue =
        [&](StringRef flag, std::string *resolvedOut,
            std::optional<StringRef> autoToolName = std::nullopt) -> bool {
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
      std::optional<std::string> resolved;
      if (requested == "auto") {
        if (!autoToolName) {
          result.error =
              (Twine("circt-mut matrix: invalid value for ") + flag + ": auto")
                  .str();
          return false;
        }
        resolved = resolveToolPathFromEnvPath(*autoToolName);
        if (!resolved) {
          result.error = (Twine("circt-mut matrix: unable to resolve ") + flag +
                          " executable: auto (searched PATH).")
                             .str();
          return false;
        }
      } else {
        resolved = resolveToolPath(requested);
      }
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
                                    &defaults.globalFilterZ3, "z3"))
        return result;
      continue;
    }
    if (arg == "--default-formal-global-propagate-bmc-z3" ||
        arg.starts_with("--default-formal-global-propagate-bmc-z3=")) {
      if (!resolveWithRequiredValue("--default-formal-global-propagate-bmc-z3",
                                    &defaults.globalFilterBMCZ3, "z3"))
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
    if (arg == "--default-mutations-seed" ||
        arg.starts_with("--default-mutations-seed=")) {
      defaults.mutationsSeed = valueFromArg().str();
    }
    if (arg == "--default-mutations-modes" ||
        arg.starts_with("--default-mutations-modes=")) {
      defaults.mutationsModes = valueFromArg().str();
    }
    if (arg == "--default-mutations-mode-counts" ||
        arg.starts_with("--default-mutations-mode-counts=")) {
      defaults.mutationsModeCounts = valueFromArg().str();
    }
    if (arg == "--default-mutations-mode-weights" ||
        arg.starts_with("--default-mutations-mode-weights=")) {
      defaults.mutationsModeWeights = valueFromArg().str();
    }
    if (arg == "--default-mutations-profiles" ||
        arg.starts_with("--default-mutations-profiles=")) {
      defaults.mutationsProfiles = valueFromArg().str();
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
  if (auto unknown = firstUnknownMutationModeCSV(defaults.mutationsModes)) {
    result.error =
        (Twine("circt-mut matrix: unknown --default-mutations-modes value: ") +
         *unknown +
         " (expected inv|const0|const1|cnot0|cnot1|"
         "arith|control|balanced|all|stuck|invert|connect).")
            .str();
    return result;
  }
  if (auto unknown =
          firstUnknownMutationModeInAllocationCSV(defaults.mutationsModeCounts)) {
    result.error =
        (Twine("circt-mut matrix: unknown --default-mutations-mode-counts "
               "mode: ") +
         *unknown +
         " (expected inv|const0|const1|cnot0|cnot1|"
         "arith|control|balanced|all|stuck|invert|connect).")
            .str();
    return result;
  }
  if (auto unknown = firstUnknownMutationModeInAllocationCSV(
          defaults.mutationsModeWeights)) {
    result.error =
        (Twine("circt-mut matrix: unknown --default-mutations-mode-weights "
               "mode: ") +
         *unknown +
         " (expected inv|const0|const1|cnot0|cnot1|"
         "arith|control|balanced|all|stuck|invert|connect).")
            .str();
    return result;
  }
  if (!defaults.mutationsSeed.empty() &&
      !Regex("^[0-9]+$").match(defaults.mutationsSeed)) {
    result.error = (Twine("circt-mut matrix: invalid --default-mutations-seed "
                          "value: ") +
                    defaults.mutationsSeed + " (expected 0-9 integer).")
                       .str();
    return result;
  }
  if (auto unknown = firstUnknownMutationProfile(defaults.mutationsProfiles)) {
    result.error =
        (Twine("circt-mut matrix: unknown --default-mutations-profiles value: ") +
         *unknown +
         " (expected arith-depth|control-depth|balanced-depth|"
         "fault-basic|fault-stuck|fault-connect|cover|none).")
            .str();
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
  if (rewrite.nativeGlobalFilterProbe)
    return runNativeCoverGlobalFilterProbe(rewrite);
  if (rewrite.nativeGlobalFilterPrequalify)
    return runNativeCoverGlobalFilterPrequalifyAndDispatch(argv0, rewrite);

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

namespace {
enum MatrixLaneColumn : size_t {
  ColLaneID = 0,
  ColDesign = 1,
  ColMutationsFile = 2,
  ColTestsManifest = 3,
  ColGenerateCount = 7,
  ColMutationsTop = 8,
  ColMutationsSeed = 9,
  ColMutationsYosys = 10,
  ColReusePairFile = 11,
  ColMutationsModes = 13,
  ColGlobalPropagateCmd = 14,
  ColGlobalPropagateCirctLEC = 15,
  ColGlobalPropagateCirctBMC = 16,
  ColGlobalPropagateBMCArgs = 17,
  ColGlobalPropagateBMCBound = 18,
  ColGlobalPropagateBMCModule = 19,
  ColGlobalPropagateBMCRunSMTLib = 20,
  ColGlobalPropagateBMCZ3 = 21,
  ColGlobalPropagateBMCAssumeKnownInputs = 22,
  ColGlobalPropagateBMCIgnoreAssertsUntil = 23,
  ColGlobalPropagateCirctLECArgs = 24,
  ColGlobalPropagateC1 = 25,
  ColGlobalPropagateC2 = 26,
  ColGlobalPropagateZ3 = 27,
  ColGlobalPropagateAssumeKnownInputs = 28,
  ColGlobalPropagateAcceptXpropOnly = 29,
  ColMutationsCfg = 30,
  ColMutationsSelect = 31,
  ColMutationsProfiles = 32,
  ColMutationsModeCounts = 33,
  ColGlobalPropagateCirctChain = 34,
  ColSkipBaseline = 39,
  ColFailOnUndetected = 40,
  ColFailOnErrors = 41,
  ColGlobalPropagateTimeoutSeconds = 42,
  ColGlobalPropagateLECTimeoutSeconds = 43,
  ColGlobalPropagateBMCTimeoutSeconds = 44,
  ColMutationsModeWeights = 45
};

struct MatrixNativePrequalifyConfig {
  std::string lanesTSVPath;
  std::string outDir = "mutation-matrix-results";
  std::string createMutatedScript;
  std::string jobsPerLane = "1";
  std::string matrixJobs = "1";
  std::string laneSchedulePolicy = "fifo";
  std::string reuseCacheDir;
  std::string reuseCacheMode;
  std::string defaultReusePairFile;
  std::string defaultMutationsModes;
  std::string defaultMutationsModeCounts;
  std::string defaultMutationsModeWeights;
  std::string defaultMutationsProfiles;
  std::string defaultMutationsCfg;
  std::string defaultMutationsSelect;
  std::string defaultMutationsSeed;
  std::string defaultMutationsYosys;
  std::string defaultGlobalFilterCmd;
  std::string defaultGlobalFilterCirctLEC;
  std::string defaultGlobalFilterCirctBMC;
  std::string defaultGlobalFilterCirctChain;
  std::string defaultGlobalFilterCirctLECArgs;
  std::string defaultGlobalFilterC1;
  std::string defaultGlobalFilterC2;
  std::string defaultGlobalFilterZ3;
  std::string defaultGlobalFilterBMCArgs;
  std::string defaultGlobalFilterBMCBound;
  std::string defaultGlobalFilterBMCModule;
  std::string defaultGlobalFilterBMCZ3;
  std::string defaultGlobalFilterBMCIgnoreAssertsUntil;
  std::string defaultGlobalFilterTimeoutSeconds;
  std::string defaultGlobalFilterLECTimeoutSeconds;
  std::string defaultGlobalFilterBMCTimeoutSeconds;
  bool defaultGlobalFilterAssumeKnownInputs = false;
  bool defaultGlobalFilterAcceptXpropOnly = false;
  bool defaultGlobalFilterBMCRunSMTLib = false;
  bool defaultGlobalFilterBMCAssumeKnownInputs = false;
  bool defaultSkipBaseline = false;
  bool defaultFailOnUndetected = false;
  bool defaultFailOnErrors = false;
};
} // namespace

static std::optional<std::string>
getLastOptionValue(ArrayRef<std::string> args, StringRef flag) {
  std::optional<std::string> value;
  std::string prefix = (flag + "=").str();
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == flag) {
      if (i + 1 < args.size())
        value = args[++i];
      continue;
    }
    if (arg.starts_with(prefix))
      value = arg.substr(prefix.size()).str();
  }
  return value;
}

static std::vector<std::string> getOptionValues(ArrayRef<std::string> args,
                                                StringRef flag) {
  std::vector<std::string> values;
  std::string prefix = (flag + "=").str();
  for (size_t i = 0; i < args.size(); ++i) {
    StringRef arg = args[i];
    if (arg == flag) {
      if (i + 1 < args.size() && !StringRef(args[i + 1]).starts_with("--"))
        values.push_back(args[i + 1]);
    }
    if (arg.starts_with(prefix))
      values.push_back(arg.substr(prefix.size()).str());
  }
  return values;
}

static bool hasOptionFlag(ArrayRef<std::string> args, StringRef flag) {
  for (StringRef arg : args)
    if (arg == flag)
      return true;
  return false;
}

static bool parseMatrixNativePrequalifyConfig(ArrayRef<std::string> args,
                                              MatrixNativePrequalifyConfig &cfg,
                                              std::string &error) {
  auto readValue = [&](StringRef flag, std::string &out) {
    if (auto value = getLastOptionValue(args, flag))
      out = *value;
  };
  readValue("--lanes-tsv", cfg.lanesTSVPath);
  readValue("--out-dir", cfg.outDir);
  readValue("--create-mutated-script", cfg.createMutatedScript);
  readValue("--jobs-per-lane", cfg.jobsPerLane);
  readValue("--jobs", cfg.matrixJobs);
  readValue("--lane-schedule-policy", cfg.laneSchedulePolicy);
  readValue("--reuse-cache-dir", cfg.reuseCacheDir);
  readValue("--reuse-cache-mode", cfg.reuseCacheMode);
  readValue("--default-reuse-pair-file", cfg.defaultReusePairFile);
  readValue("--default-mutations-modes", cfg.defaultMutationsModes);
  readValue("--default-mutations-mode-counts", cfg.defaultMutationsModeCounts);
  readValue("--default-mutations-mode-weights",
            cfg.defaultMutationsModeWeights);
  readValue("--default-mutations-profiles", cfg.defaultMutationsProfiles);
  readValue("--default-mutations-cfg", cfg.defaultMutationsCfg);
  readValue("--default-mutations-select", cfg.defaultMutationsSelect);
  readValue("--default-mutations-seed", cfg.defaultMutationsSeed);
  readValue("--default-mutations-yosys", cfg.defaultMutationsYosys);
  readValue("--default-formal-global-propagate-cmd",
            cfg.defaultGlobalFilterCmd);
  readValue("--default-formal-global-propagate-circt-lec",
            cfg.defaultGlobalFilterCirctLEC);
  readValue("--default-formal-global-propagate-circt-bmc",
            cfg.defaultGlobalFilterCirctBMC);
  readValue("--default-formal-global-propagate-circt-chain",
            cfg.defaultGlobalFilterCirctChain);
  readValue("--default-formal-global-propagate-circt-lec-args",
            cfg.defaultGlobalFilterCirctLECArgs);
  readValue("--default-formal-global-propagate-c1", cfg.defaultGlobalFilterC1);
  readValue("--default-formal-global-propagate-c2", cfg.defaultGlobalFilterC2);
  readValue("--default-formal-global-propagate-z3", cfg.defaultGlobalFilterZ3);
  readValue("--default-formal-global-propagate-circt-bmc-args",
            cfg.defaultGlobalFilterBMCArgs);
  readValue("--default-formal-global-propagate-bmc-bound",
            cfg.defaultGlobalFilterBMCBound);
  readValue("--default-formal-global-propagate-bmc-module",
            cfg.defaultGlobalFilterBMCModule);
  readValue("--default-formal-global-propagate-bmc-z3",
            cfg.defaultGlobalFilterBMCZ3);
  readValue("--default-formal-global-propagate-bmc-ignore-asserts-until",
            cfg.defaultGlobalFilterBMCIgnoreAssertsUntil);
  readValue("--default-formal-global-propagate-timeout-seconds",
            cfg.defaultGlobalFilterTimeoutSeconds);
  readValue("--default-formal-global-propagate-lec-timeout-seconds",
            cfg.defaultGlobalFilterLECTimeoutSeconds);
  readValue("--default-formal-global-propagate-bmc-timeout-seconds",
            cfg.defaultGlobalFilterBMCTimeoutSeconds);

  cfg.defaultGlobalFilterAssumeKnownInputs = hasOptionFlag(
      args, "--default-formal-global-propagate-assume-known-inputs");
  cfg.defaultGlobalFilterAcceptXpropOnly = hasOptionFlag(
      args, "--default-formal-global-propagate-accept-xprop-only");
  cfg.defaultGlobalFilterBMCRunSMTLib = hasOptionFlag(
      args, "--default-formal-global-propagate-bmc-run-smtlib");
  cfg.defaultGlobalFilterBMCAssumeKnownInputs = hasOptionFlag(
      args, "--default-formal-global-propagate-bmc-assume-known-inputs");
  cfg.defaultSkipBaseline = hasOptionFlag(args, "--skip-baseline");
  cfg.defaultFailOnUndetected = hasOptionFlag(args, "--fail-on-undetected");
  cfg.defaultFailOnErrors = hasOptionFlag(args, "--fail-on-errors");

  if (cfg.lanesTSVPath.empty()) {
    error = "circt-mut matrix: missing --lanes-tsv for native prequalification.";
    return false;
  }
  if (cfg.laneSchedulePolicy != "fifo" &&
      cfg.laneSchedulePolicy != "cache-aware") {
    error = (Twine("circt-mut matrix: invalid --lane-schedule-policy for "
                   "native dispatch: ") +
             cfg.laneSchedulePolicy + " (expected fifo|cache-aware)")
                .str();
    return false;
  }
  return true;
}

static bool parseLaneBoolWithDefault(StringRef rawValue, bool defaultValue,
                                     bool &out, std::string &error,
                                     StringRef fieldName, StringRef laneID) {
  StringRef value = rawValue.trim().lower();
  if (value.empty() || value == "-") {
    out = defaultValue;
    return true;
  }
  if (value == "1" || value == "true" || value == "yes") {
    out = true;
    return true;
  }
  if (value == "0" || value == "false" || value == "no") {
    out = false;
    return true;
  }
  error = (Twine("circt-mut matrix: invalid lane boolean override for ") +
           fieldName + " in lane " + laneID +
           " (expected 1|0|true|false|yes|no|-).")
              .str();
  return false;
}

static std::string computeMatrixLaneScheduleKey(
    const MatrixNativePrequalifyConfig &cfg, StringRef laneID,
    StringRef laneDesign, StringRef laneGenerateCount, StringRef laneMutationsTop,
    StringRef laneMutationsSeed, StringRef laneMutationsYosys,
    StringRef laneMutationsModes, StringRef laneMutationsModeCounts,
    StringRef laneMutationsModeWeights, StringRef laneMutationsProfiles,
    StringRef laneMutationsCfg, StringRef laneMutationsSelect) {
  if (cfg.reuseCacheDir.empty() || laneGenerateCount.empty() ||
      laneGenerateCount == "-")
    return (Twine("lane:") + laneID).str();

  StringRef seed = laneMutationsSeed;
  if (seed.empty() || seed == "-")
    seed = cfg.defaultMutationsSeed;
  if (seed.empty() || seed == "-")
    seed = "1";

  StringRef yosys = laneMutationsYosys;
  if (yosys.empty() || yosys == "-")
    yosys = cfg.defaultMutationsYosys;
  if (yosys.empty() || yosys == "-")
    yosys = "yosys";

  SmallString<512> key;
  raw_svector_ostream os(key);
  os << "cache:v1\n";
  os << "design=" << laneDesign << "\n";
  os << "count=" << laneGenerateCount << "\n";
  os << "top=" << laneMutationsTop << "\n";
  os << "seed=" << seed << "\n";
  os << "yosys=" << yosys << "\n";
  os << "modes=" << laneMutationsModes << "\n";
  os << "mode_counts=" << laneMutationsModeCounts << "\n";
  os << "mode_weights=" << laneMutationsModeWeights << "\n";
  os << "profiles=" << laneMutationsProfiles << "\n";
  os << "cfg=" << laneMutationsCfg << "\n";
  os << "select=" << laneMutationsSelect << "\n";
  return std::string(key.str());
}

static int runNativeMatrixGlobalFilterPrequalify(
    const char *argv0, const MatrixRewriteResult &rewrite,
    SmallVectorImpl<std::string> &dispatchArgs) {
  MatrixNativePrequalifyConfig cfg;
  std::string error;
  if (!parseMatrixNativePrequalifyConfig(rewrite.rewrittenArgs, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (mainExec.empty()) {
    errs() << "circt-mut matrix: unable to locate circt-mut executable for "
              "native prequalification.\n";
    return 1;
  }

  auto lanesBufferOrErr = MemoryBuffer::getFile(cfg.lanesTSVPath);
  if (!lanesBufferOrErr) {
    errs() << "circt-mut matrix: unable to read --lanes-tsv: "
           << cfg.lanesTSVPath << "\n";
    return 1;
  }

  std::error_code mkdirEC = sys::fs::create_directories(cfg.outDir);
  if (mkdirEC) {
    errs() << "circt-mut matrix: failed to create --out-dir: " << cfg.outDir
           << ": " << mkdirEC.message() << "\n";
    return 1;
  }

  auto trimmedColumn = [](const std::vector<std::string> &cols,
                          size_t index) -> StringRef {
    if (index >= cols.size())
      return StringRef();
    return StringRef(cols[index]).trim();
  };
  auto effectiveColumn = [&](const std::vector<std::string> &cols, size_t index,
                             StringRef defaultValue) -> std::string {
    StringRef v = trimmedColumn(cols, index);
    if (!v.empty() && v != "-")
      return v.str();
    return defaultValue.str();
  };

  SmallVector<StringRef, 256> rawLines;
  lanesBufferOrErr.get()->getBuffer().split(rawLines, '\n', /*MaxSplit=*/-1,
                                            /*KeepEmpty=*/true);
  std::vector<std::string> rewrittenLines;
  rewrittenLines.reserve(rawLines.size());
  struct LanePrequalifySummaryRow {
    std::string laneID;
    std::string pairFile;
    std::string logFile;
    bool hasSummary = false;
    uint64_t totalMutants = 0;
    uint64_t notPropagatedMutants = 0;
    uint64_t propagatedMutants = 0;
    uint64_t createMutatedErrorMutants = 0;
    uint64_t probeErrorMutants = 0;
    uint64_t cmdTokenNotPropagatedMutants = 0;
    uint64_t cmdTokenPropagatedMutants = 0;
    uint64_t cmdRCNotPropagatedMutants = 0;
    uint64_t cmdRCPropagatedMutants = 0;
    uint64_t cmdTimeoutPropagatedMutants = 0;
    uint64_t cmdErrorMutants = 0;
  };
  std::vector<LanePrequalifySummaryRow> laneSummaryRows;
  laneSummaryRows.reserve(rawLines.size());
  uint64_t prequalifiedLaneCount = 0;
  uint64_t prequalifySummaryLaneCount = 0;
  uint64_t prequalifySummaryMissingLaneCount = 0;
  uint64_t prequalifyTotalMutants = 0;
  uint64_t prequalifyNotPropagatedMutants = 0;
  uint64_t prequalifyPropagatedMutants = 0;
  uint64_t prequalifyCreateMutatedErrorMutants = 0;
  uint64_t prequalifyProbeErrorMutants = 0;
  uint64_t prequalifyCmdTokenNotPropagatedMutants = 0;
  uint64_t prequalifyCmdTokenPropagatedMutants = 0;
  uint64_t prequalifyCmdRCNotPropagatedMutants = 0;
  uint64_t prequalifyCmdRCPropagatedMutants = 0;
  uint64_t prequalifyCmdTimeoutPropagatedMutants = 0;
  uint64_t prequalifyCmdErrorMutants = 0;

  auto parsePrequalifySummaryMetrics =
      [&](StringRef logPath, StringMap<uint64_t> &metrics,
          std::string &parseError) -> bool {
    std::string logText = readTextFileOrEmpty(logPath);
    SmallVector<StringRef, 128> lines;
    StringRef(logText).split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef line : lines) {
      line = line.trim();
      if (line.empty() || line.starts_with("#"))
        continue;
      if (!line.starts_with("prequalify_"))
        continue;
      size_t tab = line.find('\t');
      if (tab == StringRef::npos)
        continue;
      StringRef key = line.take_front(tab).trim();
      StringRef value = line.drop_front(tab + 1).trim();
      bool isNumericMetric =
          key == "prequalify_total_mutants" || key.ends_with("_mutants");
      if (!isNumericMetric)
        continue;
      uint64_t parsed = 0;
      if (value.getAsInteger(10, parsed)) {
        parseError =
            (Twine("circt-mut matrix: invalid native prequalification metric in ") +
             logPath + ": " + key + "=" + value)
                .str();
        return false;
      }
      metrics[key] = parsed;
    }
    return true;
  };

  auto addMetric = [](const StringMap<uint64_t> &metrics, StringRef key,
                      uint64_t &dst) {
    if (auto it = metrics.find(key); it != metrics.end())
      dst += it->second;
  };
  auto getMetric = [](const StringMap<uint64_t> &metrics,
                      StringRef key) -> uint64_t {
    if (auto it = metrics.find(key); it != metrics.end())
      return it->second;
    return 0;
  };

  for (size_t lineNo = 0; lineNo < rawLines.size(); ++lineNo) {
    StringRef rawLine = rawLines[lineNo];
    StringRef line = rawLine;
    if (line.ends_with("\r"))
      line = line.drop_back();
    if (line.empty() || line.starts_with("#")) {
      rewrittenLines.push_back(line.str());
      continue;
    }

    SmallVector<StringRef, 64> splitCols;
    line.split(splitCols, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    std::vector<std::string> cols;
    cols.reserve(splitCols.size());
    for (StringRef col : splitCols)
      cols.push_back(col.str());
    if (cols.size() < 4) {
      errs() << "circt-mut matrix: malformed lane in --lanes-tsv at line "
             << (lineNo + 1) << " (expected at least 4 columns).\n";
      return 1;
    }

    std::string laneID = trimmedColumn(cols, ColLaneID).str();
    if (laneID.empty()) {
      errs() << "circt-mut matrix: missing lane_id in --lanes-tsv at line "
             << (lineNo + 1) << ".\n";
      return 1;
    }

    std::string laneDesign = trimmedColumn(cols, ColDesign).str();
    std::string laneTestsManifest = trimmedColumn(cols, ColTestsManifest).str();
    std::string laneGlobalCmd = effectiveColumn(cols, ColGlobalPropagateCmd,
                                                cfg.defaultGlobalFilterCmd);
    std::string laneGlobalLEC =
        effectiveColumn(cols, ColGlobalPropagateCirctLEC,
                        cfg.defaultGlobalFilterCirctLEC);
    std::string laneGlobalBMC =
        effectiveColumn(cols, ColGlobalPropagateCirctBMC,
                        cfg.defaultGlobalFilterCirctBMC);
    std::string laneGlobalChain =
        effectiveColumn(cols, ColGlobalPropagateCirctChain,
                        cfg.defaultGlobalFilterCirctChain);
    bool hasGlobalFilterMode = !laneGlobalCmd.empty() || !laneGlobalLEC.empty() ||
                               !laneGlobalBMC.empty() ||
                               !laneGlobalChain.empty();
    if (!hasGlobalFilterMode) {
      rewrittenLines.push_back(line.str());
      continue;
    }

    std::string existingReusePair =
        effectiveColumn(cols, ColReusePairFile, cfg.defaultReusePairFile);
    if (!existingReusePair.empty()) {
      errs() << "circt-mut matrix: --native-global-filter-prequalify does not "
                "support pre-existing reuse pair input for lane "
             << laneID << ".\n";
      return 1;
    }

    std::string laneWorkDir = joinPath2(cfg.outDir, laneID);
    std::string lanePairFile =
        joinPath2(laneWorkDir, "native_global_filter_prequalify.tsv");
    std::string lanePrequalifyLog =
        joinPath2(laneWorkDir, "native_global_filter_prequalify.log");
    std::error_code laneDirEC = sys::fs::create_directories(laneWorkDir);
    if (laneDirEC) {
      errs() << "circt-mut matrix: failed to create lane work dir for native "
                "prequalification: "
             << laneWorkDir << ": " << laneDirEC.message() << "\n";
      return 1;
    }

    SmallVector<std::string, 64> coverCmd;
    coverCmd.push_back(mainExec);
    coverCmd.push_back("cover");
    coverCmd.push_back("--design");
    coverCmd.push_back(laneDesign);
    coverCmd.push_back("--tests-manifest");
    coverCmd.push_back(laneTestsManifest);
    coverCmd.push_back("--native-global-filter-prequalify-only");
    coverCmd.push_back("--native-global-filter-prequalify-pair-file");
    coverCmd.push_back(lanePairFile);
    coverCmd.push_back("--work-dir");
    coverCmd.push_back(laneWorkDir);

    if (!cfg.createMutatedScript.empty()) {
      coverCmd.push_back("--create-mutated-script");
      coverCmd.push_back(cfg.createMutatedScript);
    }
    if (!cfg.jobsPerLane.empty()) {
      coverCmd.push_back("--jobs");
      coverCmd.push_back(cfg.jobsPerLane);
    }
    if (!cfg.reuseCacheDir.empty()) {
      coverCmd.push_back("--reuse-cache-dir");
      coverCmd.push_back(cfg.reuseCacheDir);
      if (!cfg.reuseCacheMode.empty()) {
        coverCmd.push_back("--reuse-cache-mode");
        coverCmd.push_back(cfg.reuseCacheMode);
      }
    }

    std::string laneMutationsFile = trimmedColumn(cols, ColMutationsFile).str();
    std::string laneGenerateCount =
        effectiveColumn(cols, ColGenerateCount, "");
    bool generatedLane =
        laneMutationsFile.empty() || laneMutationsFile == "-";
    if (!generatedLane) {
      coverCmd.push_back("--mutations-file");
      coverCmd.push_back(laneMutationsFile);
    } else {
      if (laneGenerateCount.empty() || laneGenerateCount == "-") {
        errs() << "circt-mut matrix: --native-global-filter-prequalify lane "
                "requires mutation source for lane " << laneID << ".\n";
        return 1;
      }
      coverCmd.push_back("--generate-mutations");
      coverCmd.push_back(laneGenerateCount);

      auto maybeAddArg = [&](StringRef flag, std::string value) {
        if (value.empty())
          return;
        coverCmd.push_back(flag.str());
        coverCmd.push_back(value);
      };
      maybeAddArg("--mutations-top",
                  effectiveColumn(cols, ColMutationsTop, ""));
      maybeAddArg("--mutations-seed",
                  effectiveColumn(cols, ColMutationsSeed,
                                  cfg.defaultMutationsSeed));
      maybeAddArg("--mutations-yosys",
                  effectiveColumn(cols, ColMutationsYosys,
                                  cfg.defaultMutationsYosys));
      maybeAddArg("--mutations-modes",
                  effectiveColumn(cols, ColMutationsModes,
                                  cfg.defaultMutationsModes));
      maybeAddArg("--mutations-mode-counts",
                  effectiveColumn(cols, ColMutationsModeCounts,
                                  cfg.defaultMutationsModeCounts));
      maybeAddArg("--mutations-mode-weights",
                  effectiveColumn(cols, ColMutationsModeWeights,
                                  cfg.defaultMutationsModeWeights));
      maybeAddArg("--mutations-profiles",
                  effectiveColumn(cols, ColMutationsProfiles,
                                  cfg.defaultMutationsProfiles));
      maybeAddArg("--mutations-cfg",
                  effectiveColumn(cols, ColMutationsCfg, cfg.defaultMutationsCfg));
      maybeAddArg("--mutations-select",
                  effectiveColumn(cols, ColMutationsSelect,
                                  cfg.defaultMutationsSelect));
    }

    auto maybeAddArg = [&](StringRef flag, std::string value) {
      if (value.empty())
        return;
      coverCmd.push_back(flag.str());
      coverCmd.push_back(value);
    };
    maybeAddArg("--formal-global-propagate-cmd", laneGlobalCmd);
    maybeAddArg("--formal-global-propagate-circt-chain", laneGlobalChain);
    maybeAddArg("--formal-global-propagate-circt-lec", laneGlobalLEC);
    maybeAddArg("--formal-global-propagate-circt-bmc", laneGlobalBMC);
    maybeAddArg("--formal-global-propagate-circt-lec-args",
                effectiveColumn(cols, ColGlobalPropagateCirctLECArgs,
                                cfg.defaultGlobalFilterCirctLECArgs));
    maybeAddArg("--formal-global-propagate-c1",
                effectiveColumn(cols, ColGlobalPropagateC1,
                                cfg.defaultGlobalFilterC1));
    maybeAddArg("--formal-global-propagate-c2",
                effectiveColumn(cols, ColGlobalPropagateC2,
                                cfg.defaultGlobalFilterC2));
    maybeAddArg("--formal-global-propagate-z3",
                effectiveColumn(cols, ColGlobalPropagateZ3,
                                cfg.defaultGlobalFilterZ3));
    maybeAddArg("--formal-global-propagate-circt-bmc-args",
                effectiveColumn(cols, ColGlobalPropagateBMCArgs,
                                cfg.defaultGlobalFilterBMCArgs));
    maybeAddArg("--formal-global-propagate-bmc-bound",
                effectiveColumn(cols, ColGlobalPropagateBMCBound,
                                cfg.defaultGlobalFilterBMCBound));
    maybeAddArg("--formal-global-propagate-bmc-module",
                effectiveColumn(cols, ColGlobalPropagateBMCModule,
                                cfg.defaultGlobalFilterBMCModule));
    maybeAddArg("--formal-global-propagate-bmc-z3",
                effectiveColumn(cols, ColGlobalPropagateBMCZ3,
                                cfg.defaultGlobalFilterBMCZ3));
    maybeAddArg("--formal-global-propagate-bmc-ignore-asserts-until",
                effectiveColumn(cols, ColGlobalPropagateBMCIgnoreAssertsUntil,
                                cfg.defaultGlobalFilterBMCIgnoreAssertsUntil));
    maybeAddArg("--formal-global-propagate-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateTimeoutSeconds,
                                cfg.defaultGlobalFilterTimeoutSeconds));
    maybeAddArg("--formal-global-propagate-lec-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateLECTimeoutSeconds,
                                cfg.defaultGlobalFilterLECTimeoutSeconds));
    maybeAddArg("--formal-global-propagate-bmc-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateBMCTimeoutSeconds,
                                cfg.defaultGlobalFilterBMCTimeoutSeconds));

    bool assumeKnown = false;
    bool acceptXpropOnly = false;
    bool bmcRunSMTLib = false;
    bool bmcAssumeKnown = false;
    if (!parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateAssumeKnownInputs),
            cfg.defaultGlobalFilterAssumeKnownInputs, assumeKnown, error,
            "global_propagate_assume_known_inputs", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateAcceptXpropOnly),
            cfg.defaultGlobalFilterAcceptXpropOnly, acceptXpropOnly, error,
            "global_propagate_accept_xprop_only", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateBMCRunSMTLib),
            cfg.defaultGlobalFilterBMCRunSMTLib, bmcRunSMTLib, error,
            "global_propagate_bmc_run_smtlib", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateBMCAssumeKnownInputs),
            cfg.defaultGlobalFilterBMCAssumeKnownInputs, bmcAssumeKnown, error,
            "global_propagate_bmc_assume_known_inputs", laneID)) {
      errs() << error << "\n";
      return 1;
    }
    if (assumeKnown)
      coverCmd.push_back("--formal-global-propagate-assume-known-inputs");
    if (acceptXpropOnly)
      coverCmd.push_back("--formal-global-propagate-accept-xprop-only");
    if (bmcRunSMTLib)
      coverCmd.push_back("--formal-global-propagate-bmc-run-smtlib");
    if (bmcAssumeKnown)
      coverCmd.push_back("--formal-global-propagate-bmc-assume-known-inputs");

    int prequalifyRC = -1;
    std::string runError;
    if (!runArgvToLog(coverCmd, lanePrequalifyLog, /*timeoutSeconds=*/0,
                      prequalifyRC, runError)) {
      errs() << "circt-mut matrix: failed native prequalification execution for "
                "lane "
             << laneID << ": " << runError << "\n";
      return 1;
    }
    if (prequalifyRC != 0) {
      errs() << "circt-mut matrix: native prequalification failed for lane "
             << laneID << " (exit=" << prequalifyRC
             << "), see log: " << lanePrequalifyLog << "\n";
      return 1;
    }
    StringMap<uint64_t> laneMetrics;
    if (!parsePrequalifySummaryMetrics(lanePrequalifyLog, laneMetrics, error)) {
      errs() << error << "\n";
      return 1;
    }
    if (laneMetrics.find("prequalify_total_mutants") != laneMetrics.end())
      ++prequalifySummaryLaneCount;
    else
      ++prequalifySummaryMissingLaneCount;
    addMetric(laneMetrics, "prequalify_total_mutants", prequalifyTotalMutants);
    addMetric(laneMetrics, "prequalify_not_propagated_mutants",
              prequalifyNotPropagatedMutants);
    addMetric(laneMetrics, "prequalify_propagated_mutants",
              prequalifyPropagatedMutants);
    addMetric(laneMetrics, "prequalify_create_mutated_error_mutants",
              prequalifyCreateMutatedErrorMutants);
    addMetric(laneMetrics, "prequalify_probe_error_mutants",
              prequalifyProbeErrorMutants);
    addMetric(laneMetrics, "prequalify_cmd_token_not_propagated_mutants",
              prequalifyCmdTokenNotPropagatedMutants);
    addMetric(laneMetrics, "prequalify_cmd_token_propagated_mutants",
              prequalifyCmdTokenPropagatedMutants);
    addMetric(laneMetrics, "prequalify_cmd_rc_not_propagated_mutants",
              prequalifyCmdRCNotPropagatedMutants);
    addMetric(laneMetrics, "prequalify_cmd_rc_propagated_mutants",
              prequalifyCmdRCPropagatedMutants);
    addMetric(laneMetrics, "prequalify_cmd_timeout_propagated_mutants",
              prequalifyCmdTimeoutPropagatedMutants);
    addMetric(laneMetrics, "prequalify_cmd_error_mutants",
              prequalifyCmdErrorMutants);
    laneSummaryRows.push_back(LanePrequalifySummaryRow{
        laneID,
        lanePairFile,
        lanePrequalifyLog,
        laneMetrics.find("prequalify_total_mutants") != laneMetrics.end(),
        getMetric(laneMetrics, "prequalify_total_mutants"),
        getMetric(laneMetrics, "prequalify_not_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_create_mutated_error_mutants"),
        getMetric(laneMetrics, "prequalify_probe_error_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_token_not_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_token_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_rc_not_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_rc_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_timeout_propagated_mutants"),
        getMetric(laneMetrics, "prequalify_cmd_error_mutants")});

    if (cols.size() <= ColReusePairFile)
      cols.resize(ColReusePairFile + 1);
    cols[ColReusePairFile] = lanePairFile;

    std::string outLine;
    raw_string_ostream lineOS(outLine);
    for (size_t i = 0; i < cols.size(); ++i) {
      if (i > 0)
        lineOS << '\t';
      lineOS << cols[i];
    }
    lineOS.flush();
    rewrittenLines.push_back(std::move(outLine));
    ++prequalifiedLaneCount;
  }

  std::string rewrittenLanesPath =
      joinPath2(cfg.outDir, "native_global_filter_prequalify_lanes.tsv");
  if (!ensureParentDirForFile(rewrittenLanesPath, error)) {
    errs() << error << "\n";
    return 1;
  }
  std::error_code lanesOutEC;
  raw_fd_ostream lanesOut(rewrittenLanesPath, lanesOutEC, sys::fs::OF_Text);
  if (lanesOutEC) {
    errs() << "circt-mut matrix: failed to write native prequalified lanes "
              "manifest: "
           << rewrittenLanesPath << ": " << lanesOutEC.message() << "\n";
    return 1;
  }
  for (size_t i = 0; i < rewrittenLines.size(); ++i) {
    lanesOut << rewrittenLines[i];
    if (i + 1 < rewrittenLines.size())
      lanesOut << '\n';
  }
  lanesOut.close();

  std::string summaryPath =
      joinPath2(cfg.outDir, "native_matrix_prequalify_summary.tsv");
  std::error_code summaryEC;
  raw_fd_ostream summaryOut(summaryPath, summaryEC, sys::fs::OF_Text);
  if (summaryEC) {
    errs() << "circt-mut matrix: failed to write native prequalify summary: "
           << summaryPath << ": " << summaryEC.message() << "\n";
    return 1;
  }
  summaryOut << "lane_id\tpair_file\tlog_file\thas_summary\t"
             << "prequalify_total_mutants\t"
             << "prequalify_not_propagated_mutants\t"
             << "prequalify_propagated_mutants\t"
             << "prequalify_create_mutated_error_mutants\t"
             << "prequalify_probe_error_mutants\t"
             << "prequalify_cmd_token_not_propagated_mutants\t"
             << "prequalify_cmd_token_propagated_mutants\t"
             << "prequalify_cmd_rc_not_propagated_mutants\t"
             << "prequalify_cmd_rc_propagated_mutants\t"
             << "prequalify_cmd_timeout_propagated_mutants\t"
             << "prequalify_cmd_error_mutants\n";
  for (const auto &row : laneSummaryRows) {
    summaryOut << row.laneID << '\t' << row.pairFile << '\t' << row.logFile
               << '\t' << (row.hasSummary ? "1" : "0") << '\t';
    if (row.hasSummary)
      summaryOut << row.totalMutants;
    else
      summaryOut << '-';
    summaryOut << '\t' << row.notPropagatedMutants << '\t'
               << row.propagatedMutants << '\t'
               << row.createMutatedErrorMutants << '\t' << row.probeErrorMutants
               << '\t' << row.cmdTokenNotPropagatedMutants << '\t'
               << row.cmdTokenPropagatedMutants << '\t'
               << row.cmdRCNotPropagatedMutants << '\t'
               << row.cmdRCPropagatedMutants << '\t'
               << row.cmdTimeoutPropagatedMutants << '\t' << row.cmdErrorMutants
               << '\n';
  }
  summaryOut.close();

  dispatchArgs.clear();
  bool replacedLanesArg = false;
  for (size_t i = 0; i < rewrite.rewrittenArgs.size(); ++i) {
    StringRef arg = rewrite.rewrittenArgs[i];
    if (arg == "--lanes-tsv") {
      dispatchArgs.push_back("--lanes-tsv");
      dispatchArgs.push_back(rewrittenLanesPath);
      if (i + 1 < rewrite.rewrittenArgs.size())
        ++i;
      replacedLanesArg = true;
      continue;
    }
    if (arg.starts_with("--lanes-tsv=")) {
      dispatchArgs.push_back("--lanes-tsv");
      dispatchArgs.push_back(rewrittenLanesPath);
      replacedLanesArg = true;
      continue;
    }
    dispatchArgs.push_back(arg.str());
  }
  if (!replacedLanesArg) {
    dispatchArgs.push_back("--lanes-tsv");
    dispatchArgs.push_back(rewrittenLanesPath);
  }

  outs() << "native_matrix_prequalify_lanes_tsv\t" << rewrittenLanesPath
         << "\n";
  outs() << "native_matrix_prequalify_summary_tsv\t" << summaryPath << "\n";
  outs() << "native_matrix_prequalify_lanes\t" << prequalifiedLaneCount << "\n";
  outs() << "native_matrix_prequalify_summary_lanes\t"
         << prequalifySummaryLaneCount << "\n";
  outs() << "native_matrix_prequalify_summary_missing_lanes\t"
         << prequalifySummaryMissingLaneCount << "\n";
  outs() << "native_matrix_prequalify_total_mutants\t" << prequalifyTotalMutants
         << "\n";
  outs() << "native_matrix_prequalify_not_propagated_mutants\t"
         << prequalifyNotPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_propagated_mutants\t"
         << prequalifyPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_create_mutated_error_mutants\t"
         << prequalifyCreateMutatedErrorMutants << "\n";
  outs() << "native_matrix_prequalify_probe_error_mutants\t"
         << prequalifyProbeErrorMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_token_not_propagated_mutants\t"
         << prequalifyCmdTokenNotPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_token_propagated_mutants\t"
         << prequalifyCmdTokenPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_rc_not_propagated_mutants\t"
         << prequalifyCmdRCNotPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_rc_propagated_mutants\t"
         << prequalifyCmdRCPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_timeout_propagated_mutants\t"
         << prequalifyCmdTimeoutPropagatedMutants << "\n";
  outs() << "native_matrix_prequalify_cmd_error_mutants\t"
         << prequalifyCmdErrorMutants << "\n";
  return 0;
}

struct MatrixPrequalifyLaneMetrics {
  bool hasSummary = false;
  std::string pairFile;
  std::string logFile;
  uint64_t totalMutants = 0;
  uint64_t notPropagatedMutants = 0;
  uint64_t propagatedMutants = 0;
  uint64_t createMutatedErrorMutants = 0;
  uint64_t probeErrorMutants = 0;
  uint64_t cmdTokenNotPropagatedMutants = 0;
  uint64_t cmdTokenPropagatedMutants = 0;
  uint64_t cmdRCNotPropagatedMutants = 0;
  uint64_t cmdRCPropagatedMutants = 0;
  uint64_t cmdTimeoutPropagatedMutants = 0;
  uint64_t cmdErrorMutants = 0;
};

static bool loadMatrixPrequalifyLaneMetrics(
    StringRef summaryPath, StringMap<MatrixPrequalifyLaneMetrics> &laneMetrics,
    std::string &error) {
  auto splitTSV = [](StringRef line, SmallVectorImpl<StringRef> &out) {
    out.clear();
    line.split(out, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  };
  auto bufferOrErr = MemoryBuffer::getFile(summaryPath);
  if (!bufferOrErr) {
    error = (Twine("circt-mut matrix: unable to read native prequalify summary: ") +
             summaryPath)
                .str();
    return false;
  }
  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  if (lines.empty())
    return true;

  SmallVector<StringRef, 64> fields;
  splitTSV(lines.front().rtrim("\r"), fields);
  StringMap<size_t> columns;
  for (size_t i = 0; i < fields.size(); ++i)
    columns[fields[i].trim()] = i;
  auto requireCol = [&](StringRef name, size_t &out) -> bool {
    auto it = columns.find(name);
    if (it == columns.end()) {
      error = (Twine("circt-mut matrix: missing native prequalify summary "
                     "column: ") +
               name + " in " + summaryPath)
                  .str();
      return false;
    }
    out = it->second;
    return true;
  };
  size_t laneIDCol = 0, hasSummaryCol = 0, totalCol = 0, notPropCol = 0,
         propCol = 0, createErrCol = 0, probeErrCol = 0, pairFileCol = 0,
         logFileCol = 0,
         cmdTokenNotPropCol = 0, cmdTokenPropCol = 0, cmdRCNotPropCol = 0,
         cmdRCPropCol = 0, cmdTimeoutPropCol = 0, cmdErrCol = 0;
  if (!requireCol("lane_id", laneIDCol) || !requireCol("has_summary", hasSummaryCol) ||
      !requireCol("pair_file", pairFileCol) || !requireCol("log_file", logFileCol) ||
      !requireCol("prequalify_total_mutants", totalCol) ||
      !requireCol("prequalify_not_propagated_mutants", notPropCol) ||
      !requireCol("prequalify_propagated_mutants", propCol) ||
      !requireCol("prequalify_create_mutated_error_mutants", createErrCol) ||
      !requireCol("prequalify_probe_error_mutants", probeErrCol) ||
      !requireCol("prequalify_cmd_token_not_propagated_mutants",
                  cmdTokenNotPropCol) ||
      !requireCol("prequalify_cmd_token_propagated_mutants", cmdTokenPropCol) ||
      !requireCol("prequalify_cmd_rc_not_propagated_mutants", cmdRCNotPropCol) ||
      !requireCol("prequalify_cmd_rc_propagated_mutants", cmdRCPropCol) ||
      !requireCol("prequalify_cmd_timeout_propagated_mutants",
                  cmdTimeoutPropCol) ||
      !requireCol("prequalify_cmd_error_mutants", cmdErrCol))
    return false;

  auto getField = [&](ArrayRef<StringRef> row, size_t idx) -> StringRef {
    if (idx >= row.size())
      return StringRef();
    return row[idx].trim();
  };
  auto parseUInt = [&](StringRef value, uint64_t &out, StringRef key,
                       size_t lineNo) -> bool {
    if (value.empty() || value == "-") {
      out = 0;
      return true;
    }
    if (value.getAsInteger(10, out)) {
      error = (Twine("circt-mut matrix: invalid native prequalify metric in ") +
               summaryPath + ":" + Twine(lineNo) + " key=" + key + " value=" +
               value)
                  .str();
      return false;
    }
    return true;
  };

  for (size_t lineNo = 1; lineNo < lines.size(); ++lineNo) {
    StringRef line = lines[lineNo].rtrim("\r");
    if (line.trim().empty())
      continue;
    splitTSV(line, fields);
    StringRef laneID = getField(fields, laneIDCol);
    if (laneID.empty())
      continue;

    MatrixPrequalifyLaneMetrics row;
    StringRef hasSummaryValue = getField(fields, hasSummaryCol);
    if (hasSummaryValue == "1")
      row.hasSummary = true;
    else if (hasSummaryValue.empty() || hasSummaryValue == "0" ||
             hasSummaryValue == "-")
      row.hasSummary = false;
    else {
      error = (Twine("circt-mut matrix: invalid has_summary value in ") +
               summaryPath + ":" + Twine(lineNo) + " for lane " + laneID + ": " +
               hasSummaryValue)
                  .str();
      return false;
    }
    StringRef pairFileValue = getField(fields, pairFileCol);
    StringRef logFileValue = getField(fields, logFileCol);
    row.pairFile = pairFileValue.empty() ? "-" : pairFileValue.str();
    row.logFile = logFileValue.empty() ? "-" : logFileValue.str();
    if (!parseUInt(getField(fields, totalCol), row.totalMutants,
                   "prequalify_total_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, notPropCol), row.notPropagatedMutants,
                   "prequalify_not_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, propCol), row.propagatedMutants,
                   "prequalify_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, createErrCol), row.createMutatedErrorMutants,
                   "prequalify_create_mutated_error_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, probeErrCol), row.probeErrorMutants,
                   "prequalify_probe_error_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdTokenNotPropCol),
                   row.cmdTokenNotPropagatedMutants,
                   "prequalify_cmd_token_not_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdTokenPropCol),
                   row.cmdTokenPropagatedMutants,
                   "prequalify_cmd_token_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdRCNotPropCol),
                   row.cmdRCNotPropagatedMutants,
                   "prequalify_cmd_rc_not_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdRCPropCol), row.cmdRCPropagatedMutants,
                   "prequalify_cmd_rc_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdTimeoutPropCol),
                   row.cmdTimeoutPropagatedMutants,
                   "prequalify_cmd_timeout_propagated_mutants", lineNo + 1) ||
        !parseUInt(getField(fields, cmdErrCol), row.cmdErrorMutants,
                   "prequalify_cmd_error_mutants", lineNo + 1))
      return false;

    laneMetrics[laneID] = row;
  }
  return true;
}

static bool annotateMatrixResultsWithPrequalifyMetrics(
    StringRef resultsPath,
    const StringMap<MatrixPrequalifyLaneMetrics> &laneMetricsByID,
    uint64_t &annotatedLaneRows, uint64_t &missingSummaryLaneRows,
    std::string &error) {
  auto splitTSV = [](StringRef line, SmallVectorImpl<StringRef> &out) {
    out.clear();
    line.split(out, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  };
  annotatedLaneRows = 0;
  missingSummaryLaneRows = 0;
  auto bufferOrErr = MemoryBuffer::getFile(resultsPath);
  if (!bufferOrErr) {
    error = (Twine("circt-mut matrix: unable to read matrix results file: ") +
             resultsPath)
                .str();
    return false;
  }
  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  if (lines.empty())
    return true;

  SmallVector<StringRef, 64> headerFields;
  splitTSV(lines.front().rtrim("\r"), headerFields);
  StringMap<size_t> columns;
  for (size_t i = 0; i < headerFields.size(); ++i)
    columns[headerFields[i].trim()] = i;

  auto ensureCol = [&](StringRef name) -> size_t {
    if (auto it = columns.find(name); it != columns.end())
      return it->second;
    size_t idx = headerFields.size();
    headerFields.push_back(name);
    columns[name] = idx;
    return idx;
  };

  auto laneIDIt = columns.find("lane_id");
  if (laneIDIt == columns.end()) {
    error = (Twine("circt-mut matrix: missing lane_id column in matrix results: ") +
             resultsPath)
                .str();
    return false;
  }
  size_t laneIDCol = laneIDIt->second;
  size_t prequalifyPresentCol = ensureCol("prequalify_summary_present");
  size_t prequalifyPairFileCol = ensureCol("prequalify_pair_file");
  size_t prequalifyLogFileCol = ensureCol("prequalify_log_file");
  size_t totalCol = ensureCol("prequalify_total_mutants");
  size_t notPropCol = ensureCol("prequalify_not_propagated_mutants");
  size_t propCol = ensureCol("prequalify_propagated_mutants");
  size_t createErrCol = ensureCol("prequalify_create_mutated_error_mutants");
  size_t probeErrCol = ensureCol("prequalify_probe_error_mutants");
  size_t cmdTokenNotPropCol =
      ensureCol("prequalify_cmd_token_not_propagated_mutants");
  size_t cmdTokenPropCol = ensureCol("prequalify_cmd_token_propagated_mutants");
  size_t cmdRCNotPropCol = ensureCol("prequalify_cmd_rc_not_propagated_mutants");
  size_t cmdRCPropCol = ensureCol("prequalify_cmd_rc_propagated_mutants");
  size_t cmdTimeoutPropCol =
      ensureCol("prequalify_cmd_timeout_propagated_mutants");
  size_t cmdErrCol = ensureCol("prequalify_cmd_error_mutants");

  auto toString = [](uint64_t v) { return std::to_string(v); };
  auto assignPrequalify = [&](SmallVectorImpl<std::string> &row,
                              const MatrixPrequalifyLaneMetrics *metrics) {
    if (metrics) {
      row[prequalifyPresentCol] = metrics->hasSummary ? "1" : "0";
      row[prequalifyPairFileCol] = metrics->pairFile;
      row[prequalifyLogFileCol] = metrics->logFile;
      row[totalCol] =
          metrics->hasSummary ? toString(metrics->totalMutants) : std::string("-");
      row[notPropCol] = toString(metrics->notPropagatedMutants);
      row[propCol] = toString(metrics->propagatedMutants);
      row[createErrCol] = toString(metrics->createMutatedErrorMutants);
      row[probeErrCol] = toString(metrics->probeErrorMutants);
      row[cmdTokenNotPropCol] = toString(metrics->cmdTokenNotPropagatedMutants);
      row[cmdTokenPropCol] = toString(metrics->cmdTokenPropagatedMutants);
      row[cmdRCNotPropCol] = toString(metrics->cmdRCNotPropagatedMutants);
      row[cmdRCPropCol] = toString(metrics->cmdRCPropagatedMutants);
      row[cmdTimeoutPropCol] = toString(metrics->cmdTimeoutPropagatedMutants);
      row[cmdErrCol] = toString(metrics->cmdErrorMutants);
      ++annotatedLaneRows;
      if (!metrics->hasSummary)
        ++missingSummaryLaneRows;
      return;
    }
    row[prequalifyPresentCol] = "0";
    row[prequalifyPairFileCol] = "-";
    row[prequalifyLogFileCol] = "-";
    row[totalCol] = "-";
    row[notPropCol] = "0";
    row[propCol] = "0";
    row[createErrCol] = "0";
    row[probeErrCol] = "0";
    row[cmdTokenNotPropCol] = "0";
    row[cmdTokenPropCol] = "0";
    row[cmdRCNotPropCol] = "0";
    row[cmdRCPropCol] = "0";
    row[cmdTimeoutPropCol] = "0";
    row[cmdErrCol] = "0";
    ++missingSummaryLaneRows;
  };

  std::vector<std::string> outLines;
  outLines.reserve(lines.size());
  {
    std::string headerLine;
    raw_string_ostream os(headerLine);
    for (size_t i = 0; i < headerFields.size(); ++i) {
      if (i)
        os << '\t';
      os << headerFields[i];
    }
    os.flush();
    outLines.push_back(std::move(headerLine));
  }

  SmallVector<StringRef, 64> rowFields;
  for (size_t lineNo = 1; lineNo < lines.size(); ++lineNo) {
    StringRef line = lines[lineNo].rtrim("\r");
    if (line.trim().empty())
      continue;
    splitTSV(line, rowFields);
    SmallVector<std::string, 64> row;
    row.reserve(headerFields.size());
    for (StringRef f : rowFields)
      row.push_back(f.str());
    if (row.size() < headerFields.size())
      row.resize(headerFields.size());
    StringRef laneID =
        laneIDCol < row.size() ? StringRef(row[laneIDCol]).trim() : StringRef();
    const MatrixPrequalifyLaneMetrics *metrics = nullptr;
    if (!laneID.empty()) {
      if (auto it = laneMetricsByID.find(laneID); it != laneMetricsByID.end())
        metrics = &it->second;
    }
    assignPrequalify(row, metrics);

    std::string out;
    raw_string_ostream os(out);
    for (size_t i = 0; i < row.size(); ++i) {
      if (i)
        os << '\t';
      os << row[i];
    }
    os.flush();
    outLines.push_back(std::move(out));
  }

  std::error_code ec;
  raw_fd_ostream out(resultsPath, ec, sys::fs::OF_Text);
  if (ec) {
    error = (Twine("circt-mut matrix: failed to rewrite matrix results file: ") +
             resultsPath + ": " + ec.message())
                .str();
    return false;
  }
  for (size_t i = 0; i < outLines.size(); ++i) {
    out << outLines[i];
    if (i + 1 < outLines.size())
      out << '\n';
  }
  return true;
}

static int runNativeMatrixDispatch(const char *argv0,
                                   ArrayRef<std::string> args) {
  MatrixNativePrequalifyConfig cfg;
  std::string error;
  if (!parseMatrixNativePrequalifyConfig(args, cfg, error)) {
    errs() << error << "\n";
    return 1;
  }

  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (mainExec.empty()) {
    errs() << "circt-mut matrix: unable to locate circt-mut executable for "
              "native matrix dispatch.\n";
    return 1;
  }

  auto lanesBufferOrErr = MemoryBuffer::getFile(cfg.lanesTSVPath);
  if (!lanesBufferOrErr) {
    errs() << "circt-mut matrix: unable to read --lanes-tsv: "
           << cfg.lanesTSVPath << "\n";
    return 1;
  }

  std::error_code mkdirEC = sys::fs::create_directories(cfg.outDir);
  if (mkdirEC) {
    errs() << "circt-mut matrix: failed to create --out-dir: " << cfg.outDir
           << ": " << mkdirEC.message() << "\n";
    return 1;
  }

  std::string resultsPath = joinPath2(cfg.outDir, "results.tsv");
  if (auto v = getLastOptionValue(args, "--results-file"))
    resultsPath = *v;
  std::string gateSummaryPath = joinPath2(cfg.outDir, "gate_summary.tsv");
  if (auto v = getLastOptionValue(args, "--gate-summary-file"))
    gateSummaryPath = *v;
  std::string runtimeSummaryPath =
      joinPath2(cfg.outDir, "native_matrix_dispatch_runtime.tsv");
  if (!ensureParentDirForFile(resultsPath, error)) {
    errs() << error << "\n";
    return 1;
  }
  if (!ensureParentDirForFile(gateSummaryPath, error)) {
    errs() << error << "\n";
    return 1;
  }

  auto trimmedColumn = [](const std::vector<std::string> &cols,
                          size_t index) -> StringRef {
    if (index >= cols.size())
      return StringRef();
    return StringRef(cols[index]).trim();
  };
  auto effectiveColumn = [&](const std::vector<std::string> &cols, size_t index,
                             StringRef defaultValue) -> std::string {
    StringRef v = trimmedColumn(cols, index);
    if (!v.empty() && v != "-")
      return v.str();
    return defaultValue.str();
  };

  std::error_code outEC;
  raw_fd_ostream out(resultsPath, outEC, sys::fs::OF_Text);
  if (outEC) {
    errs() << "circt-mut matrix: failed to open results file: " << resultsPath
           << ": " << outEC.message() << "\n";
    return 1;
  }
  out << "lane_id\tstatus\texit_code\tcoverage_percent\truntime_ns\tgate_status\t"
         "lane_dir\tmetrics_file\tsummary_json\tconfig_error_code\t"
         "config_error_reason\n";

  SmallVector<StringRef, 256> rawLines;
  lanesBufferOrErr.get()->getBuffer().split(rawLines, '\n', /*MaxSplit=*/-1,
                                            /*KeepEmpty=*/true);
  uint64_t laneTotal = 0;
  uint64_t lanePass = 0;
  uint64_t laneFail = 0;
  uint64_t laneSkip = 0;
  uint64_t laneExecuted = 0;
  uint64_t laneFilteredInclude = 0;
  uint64_t laneFilteredExclude = 0;
  uint64_t laneRuntimeNanos = 0;
  std::vector<std::pair<std::string, uint64_t>> runtimeRows;
  runtimeRows.reserve(rawLines.size());
  bool stopOnFail = hasOptionFlag(args, "--stop-on-fail");
  uint64_t matrixJobs = 1;
  if (!cfg.matrixJobs.empty()) {
    if (StringRef(cfg.matrixJobs).getAsInteger(10, matrixJobs) ||
        matrixJobs == 0) {
      errs() << "circt-mut matrix: invalid --jobs for native dispatch: "
             << cfg.matrixJobs << " (expected positive integer)\n";
      return 1;
    }
  }
  bool bufferedMode = matrixJobs > 1 || cfg.laneSchedulePolicy == "cache-aware";
  std::vector<Regex> includeLaneRegexes;
  std::vector<Regex> excludeLaneRegexes;
  for (const auto &v : getOptionValues(args, "--include-lane-regex")) {
    Regex r(v);
    std::string regexError;
    if (!r.isValid(regexError)) {
      errs() << "circt-mut matrix: invalid --include-lane-regex: " << v
             << " (" << regexError << ")\n";
      return 1;
    }
    includeLaneRegexes.push_back(std::move(r));
  }
  for (const auto &v : getOptionValues(args, "--exclude-lane-regex")) {
    Regex r(v);
    std::string regexError;
    if (!r.isValid(regexError)) {
      errs() << "circt-mut matrix: invalid --exclude-lane-regex: " << v
             << " (" << regexError << ")\n";
      return 1;
    }
    excludeLaneRegexes.push_back(std::move(r));
  }
  StringMap<uint64_t> gateCounts;
  std::vector<std::string> bufferedRows;
  bufferedRows.reserve(rawLines.size());
  struct PendingLaneJob {
    size_t rowIndex = 0;
    std::string laneID;
    std::string laneWorkDir;
    std::string laneLog;
    std::string scheduleKey;
    SmallVector<std::string, 96> coverCmd;
  };
  std::vector<PendingLaneJob> pendingJobs;
  uint64_t scheduleUniqueKeys = 0;
  uint64_t scheduleDeferredFollowers = 0;
  auto formatLaneRow = [](StringRef laneID, StringRef status, int exitCode,
                          StringRef coveragePercent, StringRef runtimeNanos,
                          StringRef gateStatus, StringRef laneDir,
                          StringRef metricsFile, StringRef summaryJSON,
                          StringRef configErrorCode,
                          StringRef configErrorReason) {
    std::string row;
    raw_string_ostream os(row);
    os << laneID << '\t' << status << '\t' << exitCode << '\t'
       << coveragePercent << '\t' << runtimeNanos << '\t' << gateStatus << '\t'
       << laneDir << '\t' << metricsFile << '\t' << summaryJSON << '\t'
       << configErrorCode << '\t' << configErrorReason;
    os.flush();
    return row;
  };
  auto emitLaneRow = [&](StringRef laneID, StringRef status, int exitCode,
                         StringRef coveragePercent, StringRef runtimeNanos,
                         StringRef gateStatus, StringRef laneDir,
                         StringRef metricsFile, StringRef summaryJSON,
                         StringRef configErrorCode,
                         StringRef configErrorReason) {
    if (bufferedMode) {
      bufferedRows.push_back(formatLaneRow(
          laneID, status, exitCode, coveragePercent, runtimeNanos, gateStatus,
          laneDir, metricsFile, summaryJSON, configErrorCode,
          configErrorReason));
      return;
    }
    out << laneID << '\t' << status << '\t' << exitCode << '\t'
        << coveragePercent << '\t' << runtimeNanos << '\t' << gateStatus
        << '\t' << laneDir << '\t' << metricsFile << '\t' << summaryJSON
        << '\t' << configErrorCode << '\t' << configErrorReason << '\n';
  };

  auto maybeAddArg = [](SmallVectorImpl<std::string> &cmd, StringRef flag,
                        const std::string &value) {
    if (value.empty())
      return;
    cmd.push_back(flag.str());
    cmd.push_back(value);
  };
  for (size_t lineNo = 0; lineNo < rawLines.size(); ++lineNo) {
    StringRef rawLine = rawLines[lineNo];
    StringRef line = rawLine;
    if (line.ends_with("\r"))
      line = line.drop_back();
    if (line.empty() || line.starts_with("#"))
      continue;

    SmallVector<StringRef, 64> splitCols;
    line.split(splitCols, '\t', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    std::vector<std::string> cols;
    cols.reserve(splitCols.size());
    for (StringRef col : splitCols)
      cols.push_back(col.str());
    if (cols.size() < 4) {
      errs() << "circt-mut matrix: malformed lane in --lanes-tsv at line "
             << (lineNo + 1) << " (expected at least 4 columns).\n";
      return 1;
    }

    std::string laneID = trimmedColumn(cols, ColLaneID).str();
    if (laneID.empty()) {
      errs() << "circt-mut matrix: missing lane_id in --lanes-tsv at line "
             << (lineNo + 1) << ".\n";
      return 1;
    }
    if (!includeLaneRegexes.empty() &&
        llvm::none_of(includeLaneRegexes,
                      [&](const Regex &regex) { return regex.match(laneID); })) {
      ++laneFilteredInclude;
      continue;
    }
    if (llvm::any_of(excludeLaneRegexes,
                     [&](const Regex &regex) { return regex.match(laneID); })) {
      ++laneFilteredExclude;
      continue;
    }

    ++laneTotal;
    std::string laneDesign = trimmedColumn(cols, ColDesign).str();
    std::string laneTestsManifest = trimmedColumn(cols, ColTestsManifest).str();
    std::string laneMutationsFile = trimmedColumn(cols, ColMutationsFile).str();
    std::string laneGenerateCount = effectiveColumn(cols, ColGenerateCount, "");
    std::string laneWorkDir = joinPath2(cfg.outDir, laneID);
    std::error_code laneDirEC = sys::fs::create_directories(laneWorkDir);
    if (laneDirEC) {
      emitLaneRow(laneID, "FAIL", 1, "-", "-", "FAIL", laneWorkDir, "-", "-",
                  "DIR_ERROR", "lane_work_dir_create_failed");
      ++laneFail;
      gateCounts["FAIL"]++;
      if (stopOnFail)
        break;
      continue;
    }
    std::string laneLog = joinPath2(laneWorkDir, "native_matrix_lane.log");

    if (laneDesign.empty() || laneTestsManifest.empty() ||
        ((laneMutationsFile.empty() || laneMutationsFile == "-") &&
         laneGenerateCount.empty())) {
      emitLaneRow(laneID, "FAIL", 1, "-", "-", "FAIL", laneWorkDir, "-", "-",
                  "CONFIG_ERROR", "missing_required_lane_fields");
      ++laneFail;
      gateCounts["FAIL"]++;
      if (stopOnFail)
        break;
      continue;
    }

    SmallVector<std::string, 96> coverCmd;
    coverCmd.push_back(mainExec);
    coverCmd.push_back("cover");
    coverCmd.push_back("--design");
    coverCmd.push_back(laneDesign);
    coverCmd.push_back("--tests-manifest");
    coverCmd.push_back(laneTestsManifest);
    coverCmd.push_back("--work-dir");
    coverCmd.push_back(laneWorkDir);
    if (!cfg.createMutatedScript.empty()) {
      coverCmd.push_back("--create-mutated-script");
      coverCmd.push_back(cfg.createMutatedScript);
    }
    if (!cfg.jobsPerLane.empty()) {
      coverCmd.push_back("--jobs");
      coverCmd.push_back(cfg.jobsPerLane);
    }
    if (!cfg.reuseCacheDir.empty()) {
      coverCmd.push_back("--reuse-cache-dir");
      coverCmd.push_back(cfg.reuseCacheDir);
      if (!cfg.reuseCacheMode.empty()) {
        coverCmd.push_back("--reuse-cache-mode");
        coverCmd.push_back(cfg.reuseCacheMode);
      }
    }
    std::string laneReusePair =
        effectiveColumn(cols, ColReusePairFile, cfg.defaultReusePairFile);
    maybeAddArg(coverCmd, "--reuse-pair-file", laneReusePair);

    bool generatedLane =
        laneMutationsFile.empty() || laneMutationsFile == "-";
    std::string laneMutationsTop;
    std::string laneMutationsSeed;
    std::string laneMutationsYosys;
    std::string laneMutationsModes;
    std::string laneMutationsModeCounts;
    std::string laneMutationsModeWeights;
    std::string laneMutationsProfiles;
    std::string laneMutationsCfg;
    std::string laneMutationsSelect;
    if (!generatedLane) {
      coverCmd.push_back("--mutations-file");
      coverCmd.push_back(laneMutationsFile);
    } else {
      coverCmd.push_back("--generate-mutations");
      coverCmd.push_back(laneGenerateCount);
      laneMutationsTop = effectiveColumn(cols, ColMutationsTop, "");
      laneMutationsSeed =
          effectiveColumn(cols, ColMutationsSeed, cfg.defaultMutationsSeed);
      laneMutationsYosys =
          effectiveColumn(cols, ColMutationsYosys, cfg.defaultMutationsYosys);
      laneMutationsModes =
          effectiveColumn(cols, ColMutationsModes, cfg.defaultMutationsModes);
      laneMutationsModeCounts = effectiveColumn(cols, ColMutationsModeCounts,
                                                cfg.defaultMutationsModeCounts);
      laneMutationsModeWeights = effectiveColumn(
          cols, ColMutationsModeWeights, cfg.defaultMutationsModeWeights);
      laneMutationsProfiles = effectiveColumn(cols, ColMutationsProfiles,
                                              cfg.defaultMutationsProfiles);
      laneMutationsCfg =
          effectiveColumn(cols, ColMutationsCfg, cfg.defaultMutationsCfg);
      laneMutationsSelect =
          effectiveColumn(cols, ColMutationsSelect, cfg.defaultMutationsSelect);
      maybeAddArg(coverCmd, "--mutations-top", laneMutationsTop);
      maybeAddArg(coverCmd, "--mutations-seed", laneMutationsSeed);
      maybeAddArg(coverCmd, "--mutations-yosys", laneMutationsYosys);
      maybeAddArg(coverCmd, "--mutations-modes", laneMutationsModes);
      maybeAddArg(coverCmd, "--mutations-mode-counts", laneMutationsModeCounts);
      maybeAddArg(coverCmd, "--mutations-mode-weights",
                  laneMutationsModeWeights);
      maybeAddArg(coverCmd, "--mutations-profiles", laneMutationsProfiles);
      maybeAddArg(coverCmd, "--mutations-cfg", laneMutationsCfg);
      maybeAddArg(coverCmd, "--mutations-select", laneMutationsSelect);
    }

    std::string laneGlobalCmd = effectiveColumn(cols, ColGlobalPropagateCmd,
                                                cfg.defaultGlobalFilterCmd);
    std::string laneGlobalLEC =
        effectiveColumn(cols, ColGlobalPropagateCirctLEC,
                        cfg.defaultGlobalFilterCirctLEC);
    std::string laneGlobalBMC =
        effectiveColumn(cols, ColGlobalPropagateCirctBMC,
                        cfg.defaultGlobalFilterCirctBMC);
    std::string laneGlobalChain =
        effectiveColumn(cols, ColGlobalPropagateCirctChain,
                        cfg.defaultGlobalFilterCirctChain);
    maybeAddArg(coverCmd, "--formal-global-propagate-cmd", laneGlobalCmd);
    maybeAddArg(coverCmd, "--formal-global-propagate-circt-chain",
                laneGlobalChain);
    maybeAddArg(coverCmd, "--formal-global-propagate-circt-lec",
                laneGlobalLEC);
    maybeAddArg(coverCmd, "--formal-global-propagate-circt-bmc",
                laneGlobalBMC);
    maybeAddArg(coverCmd, "--formal-global-propagate-circt-lec-args",
                effectiveColumn(cols, ColGlobalPropagateCirctLECArgs,
                                cfg.defaultGlobalFilterCirctLECArgs));
    maybeAddArg(coverCmd, "--formal-global-propagate-c1",
                effectiveColumn(cols, ColGlobalPropagateC1,
                                cfg.defaultGlobalFilterC1));
    maybeAddArg(coverCmd, "--formal-global-propagate-c2",
                effectiveColumn(cols, ColGlobalPropagateC2,
                                cfg.defaultGlobalFilterC2));
    maybeAddArg(coverCmd, "--formal-global-propagate-z3",
                effectiveColumn(cols, ColGlobalPropagateZ3,
                                cfg.defaultGlobalFilterZ3));
    maybeAddArg(coverCmd, "--formal-global-propagate-circt-bmc-args",
                effectiveColumn(cols, ColGlobalPropagateBMCArgs,
                                cfg.defaultGlobalFilterBMCArgs));
    maybeAddArg(coverCmd, "--formal-global-propagate-bmc-bound",
                effectiveColumn(cols, ColGlobalPropagateBMCBound,
                                cfg.defaultGlobalFilterBMCBound));
    maybeAddArg(coverCmd, "--formal-global-propagate-bmc-module",
                effectiveColumn(cols, ColGlobalPropagateBMCModule,
                                cfg.defaultGlobalFilterBMCModule));
    maybeAddArg(coverCmd, "--formal-global-propagate-bmc-z3",
                effectiveColumn(cols, ColGlobalPropagateBMCZ3,
                                cfg.defaultGlobalFilterBMCZ3));
    maybeAddArg(coverCmd, "--formal-global-propagate-bmc-ignore-asserts-until",
                effectiveColumn(cols, ColGlobalPropagateBMCIgnoreAssertsUntil,
                                cfg.defaultGlobalFilterBMCIgnoreAssertsUntil));
    maybeAddArg(coverCmd, "--formal-global-propagate-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateTimeoutSeconds,
                                cfg.defaultGlobalFilterTimeoutSeconds));
    maybeAddArg(coverCmd, "--formal-global-propagate-lec-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateLECTimeoutSeconds,
                                cfg.defaultGlobalFilterLECTimeoutSeconds));
    maybeAddArg(coverCmd, "--formal-global-propagate-bmc-timeout-seconds",
                effectiveColumn(cols, ColGlobalPropagateBMCTimeoutSeconds,
                                cfg.defaultGlobalFilterBMCTimeoutSeconds));

    bool assumeKnown = false;
    bool acceptXpropOnly = false;
    bool bmcRunSMTLib = false;
    bool bmcAssumeKnown = false;
    bool skipBaseline = false;
    bool failOnUndetected = false;
    bool failOnErrors = false;
    if (!parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateAssumeKnownInputs),
            cfg.defaultGlobalFilterAssumeKnownInputs, assumeKnown, error,
            "global_propagate_assume_known_inputs", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateAcceptXpropOnly),
            cfg.defaultGlobalFilterAcceptXpropOnly, acceptXpropOnly, error,
            "global_propagate_accept_xprop_only", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateBMCRunSMTLib),
            cfg.defaultGlobalFilterBMCRunSMTLib, bmcRunSMTLib, error,
            "global_propagate_bmc_run_smtlib", laneID) ||
        !parseLaneBoolWithDefault(
            trimmedColumn(cols, ColGlobalPropagateBMCAssumeKnownInputs),
            cfg.defaultGlobalFilterBMCAssumeKnownInputs, bmcAssumeKnown, error,
            "global_propagate_bmc_assume_known_inputs", laneID) ||
        !parseLaneBoolWithDefault(trimmedColumn(cols, ColSkipBaseline),
                                  cfg.defaultSkipBaseline, skipBaseline, error,
                                  "skip_baseline", laneID) ||
        !parseLaneBoolWithDefault(trimmedColumn(cols, ColFailOnUndetected),
                                  cfg.defaultFailOnUndetected,
                                  failOnUndetected, error,
                                  "fail_on_undetected", laneID) ||
        !parseLaneBoolWithDefault(trimmedColumn(cols, ColFailOnErrors),
                                  cfg.defaultFailOnErrors, failOnErrors, error,
                                  "fail_on_errors", laneID)) {
      emitLaneRow(laneID, "FAIL", 1, "-", "-", "FAIL", laneWorkDir, "-", "-",
                  "CONFIG_ERROR", "invalid_lane_boolean_override");
      ++laneFail;
      gateCounts["FAIL"]++;
      if (stopOnFail)
        break;
      continue;
    }
    if (assumeKnown)
      coverCmd.push_back("--formal-global-propagate-assume-known-inputs");
    if (acceptXpropOnly)
      coverCmd.push_back("--formal-global-propagate-accept-xprop-only");
    if (bmcRunSMTLib)
      coverCmd.push_back("--formal-global-propagate-bmc-run-smtlib");
    if (bmcAssumeKnown)
      coverCmd.push_back("--formal-global-propagate-bmc-assume-known-inputs");
    if (skipBaseline)
      coverCmd.push_back("--skip-baseline");
    if (failOnUndetected)
      coverCmd.push_back("--fail-on-undetected");
    if (failOnErrors)
      coverCmd.push_back("--fail-on-errors");

    if (bufferedMode) {
      PendingLaneJob job;
      job.rowIndex = bufferedRows.size();
      job.laneID = laneID;
      job.laneWorkDir = laneWorkDir;
      job.laneLog = joinPath2(laneWorkDir, "native_matrix_lane.log");
      job.scheduleKey = computeMatrixLaneScheduleKey(
          cfg, laneID, laneDesign, laneGenerateCount, laneMutationsTop,
          laneMutationsSeed, laneMutationsYosys, laneMutationsModes,
          laneMutationsModeCounts, laneMutationsModeWeights,
          laneMutationsProfiles, laneMutationsCfg, laneMutationsSelect);
      job.coverCmd = std::move(coverCmd);
      pendingJobs.push_back(std::move(job));
      bufferedRows.emplace_back();
      continue;
    }

    int coverRC = -1;
    std::string runError;
    auto laneStart = std::chrono::steady_clock::now();
    if (!runArgvToLog(coverCmd, laneLog, /*timeoutSeconds=*/0, coverRC,
                      runError)) {
      auto laneEnd = std::chrono::steady_clock::now();
      uint64_t runtimeNanos = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(laneEnd -
                                                               laneStart)
              .count());
      laneRuntimeNanos += runtimeNanos;
      ++laneExecuted;
      runtimeRows.emplace_back(laneID, runtimeNanos);
      emitLaneRow(laneID, "FAIL", 1, "-", std::to_string(runtimeNanos), "FAIL",
                  laneWorkDir, "-", "-",
                  "DISPATCH_ERROR", "cover_invocation_failed");
      ++laneFail;
      gateCounts["FAIL"]++;
      if (stopOnFail)
        break;
      continue;
    }

    std::string metricsPath = joinPath2(laneWorkDir, "metrics.tsv");
    std::string summaryPath = joinPath2(laneWorkDir, "summary.json");
    std::string coveragePercent = "-";
    auto laneEnd = std::chrono::steady_clock::now();
    uint64_t runtimeNanos = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(laneEnd - laneStart)
            .count());
    laneRuntimeNanos += runtimeNanos;
    ++laneExecuted;
    runtimeRows.emplace_back(laneID, runtimeNanos);
    if (sys::fs::exists(metricsPath)) {
      std::string metricsText = readTextFileOrEmpty(metricsPath);
      SmallVector<StringRef, 128> metricLines;
      StringRef(metricsText).split(metricLines, '\n', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
      for (StringRef metricLine : metricLines) {
        metricLine = metricLine.trim();
        if (metricLine.empty() || metricLine.starts_with("#"))
          continue;
        size_t tabPos = metricLine.find('\t');
        if (tabPos == StringRef::npos)
          continue;
        StringRef key = metricLine.take_front(tabPos).trim();
        StringRef value = metricLine.drop_front(tabPos + 1).trim();
        if (key == "mutation_coverage_percent" && !value.empty()) {
          coveragePercent = value.str();
          break;
        }
      }
    }

    bool pass = coverRC == 0;
    emitLaneRow(laneID, pass ? "PASS" : "FAIL", coverRC, coveragePercent,
                std::to_string(runtimeNanos), pass ? "PASS" : "FAIL", laneWorkDir,
                sys::fs::exists(metricsPath) ? metricsPath : "-",
                sys::fs::exists(summaryPath) ? summaryPath : "-", "-", "-");
    gateCounts[pass ? "PASS" : "FAIL"]++;
    if (pass)
      ++lanePass;
    else
      ++laneFail;
    if (!pass && stopOnFail)
      break;
  }

  if (bufferedMode && cfg.laneSchedulePolicy == "cache-aware") {
    StringSet<> seenKeys;
    std::vector<PendingLaneJob> leaders;
    std::vector<PendingLaneJob> followers;
    leaders.reserve(pendingJobs.size());
    followers.reserve(pendingJobs.size());
    for (auto &job : pendingJobs) {
      StringRef key = job.scheduleKey;
      if (key.empty())
        key = job.laneID;
      if (seenKeys.insert(key).second) {
        leaders.push_back(std::move(job));
        ++scheduleUniqueKeys;
      } else {
        followers.push_back(std::move(job));
        ++scheduleDeferredFollowers;
      }
    }
    pendingJobs.clear();
    pendingJobs.reserve(leaders.size() + followers.size());
    for (auto &job : leaders)
      pendingJobs.push_back(std::move(job));
    for (auto &job : followers)
      pendingJobs.push_back(std::move(job));
  } else if (bufferedMode) {
    scheduleUniqueKeys = pendingJobs.size();
  }

  if (bufferedMode) {
    struct LaneJobOutcome {
      std::string row;
      bool pass = false;
      bool skip = false;
      uint64_t runtimeNanos = 0;
    };
    std::vector<LaneJobOutcome> outcomes(pendingJobs.size());
    std::atomic<size_t> nextJob{0};
    std::atomic<size_t> firstFailIndex{pendingJobs.size()};
    std::atomic<bool> stopRequested{false};
    uint64_t workerCount = std::min<uint64_t>(matrixJobs, pendingJobs.size());
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    auto updateFirstFail = [&](size_t idx) {
      size_t current = firstFailIndex.load();
      while (idx < current &&
             !firstFailIndex.compare_exchange_weak(current, idx)) {
      }
    };
    auto formatSkipRow = [&](const PendingLaneJob &job) {
      return formatLaneRow(job.laneID, "SKIP", 0, "-", "-", "SKIP",
                           job.laneWorkDir, "-", "-", "STOP_ON_FAIL",
                           "skipped_after_fail");
    };
    auto runOneJob = [&](size_t idx) {
      const auto &job = pendingJobs[idx];
      if (stopOnFail && stopRequested.load()) {
        outcomes[idx].row = formatSkipRow(job);
        outcomes[idx].pass = false;
        outcomes[idx].skip = true;
        return;
      }
      int coverRC = -1;
      std::string runError;
      auto laneStart = std::chrono::steady_clock::now();
      if (!runArgvToLog(job.coverCmd, job.laneLog, /*timeoutSeconds=*/0, coverRC,
                        runError)) {
        auto laneEnd = std::chrono::steady_clock::now();
        outcomes[idx].runtimeNanos = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(laneEnd -
                                                                 laneStart)
                .count());
        outcomes[idx].row = formatLaneRow(
            job.laneID, "FAIL", 1, "-", std::to_string(outcomes[idx].runtimeNanos),
            "FAIL", job.laneWorkDir, "-", "-", "DISPATCH_ERROR",
            "cover_invocation_failed");
        outcomes[idx].pass = false;
        outcomes[idx].skip = false;
        if (stopOnFail) {
          updateFirstFail(idx);
          stopRequested.store(true);
        }
        return;
      }
      std::string metricsPath = joinPath2(job.laneWorkDir, "metrics.tsv");
      std::string summaryPath = joinPath2(job.laneWorkDir, "summary.json");
      std::string coveragePercent = "-";
      auto laneEnd = std::chrono::steady_clock::now();
      outcomes[idx].runtimeNanos = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(laneEnd -
                                                               laneStart)
              .count());
      if (sys::fs::exists(metricsPath)) {
        std::string metricsText = readTextFileOrEmpty(metricsPath);
        SmallVector<StringRef, 128> metricLines;
        StringRef(metricsText).split(metricLines, '\n', /*MaxSplit=*/-1,
                                     /*KeepEmpty=*/false);
        for (StringRef metricLine : metricLines) {
          metricLine = metricLine.trim();
          if (metricLine.empty() || metricLine.starts_with("#"))
            continue;
          size_t tabPos = metricLine.find('\t');
          if (tabPos == StringRef::npos)
            continue;
          StringRef key = metricLine.take_front(tabPos).trim();
          StringRef value = metricLine.drop_front(tabPos + 1).trim();
          if (key == "mutation_coverage_percent" && !value.empty()) {
            coveragePercent = value.str();
            break;
          }
        }
      }
      bool pass = coverRC == 0;
      outcomes[idx].row = formatLaneRow(
          job.laneID, pass ? "PASS" : "FAIL", coverRC, coveragePercent,
          std::to_string(outcomes[idx].runtimeNanos), pass ? "PASS" : "FAIL",
          job.laneWorkDir,
          sys::fs::exists(metricsPath) ? metricsPath : "-",
          sys::fs::exists(summaryPath) ? summaryPath : "-", "-", "-");
      outcomes[idx].pass = pass;
      outcomes[idx].skip = false;
      if (!pass && stopOnFail) {
        updateFirstFail(idx);
        stopRequested.store(true);
      }
    };
    auto workerFn = [&]() {
      while (true) {
        size_t idx = nextJob.fetch_add(1);
        if (idx >= pendingJobs.size())
          return;
        runOneJob(idx);
      }
    };
    if (workerCount > 0) {
      for (uint64_t w = 0; w < workerCount; ++w)
        workers.emplace_back(workerFn);
      for (auto &w : workers)
        w.join();
    }
    size_t cutIndex = firstFailIndex.load();
    for (size_t i = 0; i < pendingJobs.size(); ++i) {
      const auto &job = pendingJobs[i];
      if (stopOnFail && cutIndex < pendingJobs.size() && i > cutIndex) {
        outcomes[i].row = formatSkipRow(job);
        outcomes[i].pass = false;
        outcomes[i].skip = true;
      }
      bufferedRows[job.rowIndex] = outcomes[i].row;
      if (outcomes[i].skip) {
        ++laneSkip;
        gateCounts["SKIP"]++;
      } else if (outcomes[i].pass) {
        ++laneExecuted;
        laneRuntimeNanos += outcomes[i].runtimeNanos;
        runtimeRows.emplace_back(job.laneID, outcomes[i].runtimeNanos);
        ++lanePass;
        gateCounts["PASS"]++;
      } else {
        ++laneExecuted;
        laneRuntimeNanos += outcomes[i].runtimeNanos;
        runtimeRows.emplace_back(job.laneID, outcomes[i].runtimeNanos);
        ++laneFail;
        gateCounts["FAIL"]++;
      }
    }
    for (const auto &row : bufferedRows)
      out << row << '\n';
  }
  if (!bufferedMode)
    scheduleUniqueKeys = laneTotal;

  out.close();
  std::error_code gateEC;
  raw_fd_ostream gateOut(gateSummaryPath, gateEC, sys::fs::OF_Text);
  if (gateEC) {
    errs() << "circt-mut matrix: failed to open gate summary file: "
           << gateSummaryPath << ": " << gateEC.message() << "\n";
    return 1;
  }
  gateOut << "gate_status\tcount\n";
  SmallVector<std::string, 8> gates;
  gates.reserve(gateCounts.size());
  for (const auto &it : gateCounts)
    gates.push_back(it.getKey().str());
  llvm::sort(gates);
  for (const auto &gate : gates)
    gateOut << gate << '\t' << gateCounts[gate] << '\n';
  gateOut.close();

  std::error_code runtimeEC;
  raw_fd_ostream runtimeOut(runtimeSummaryPath, runtimeEC, sys::fs::OF_Text);
  if (runtimeEC) {
    errs() << "circt-mut matrix: failed to open runtime summary file: "
           << runtimeSummaryPath << ": " << runtimeEC.message() << "\n";
    return 1;
  }
  runtimeOut << "lane_id\truntime_ns\n";
  for (const auto &it : runtimeRows)
    runtimeOut << it.first << '\t' << it.second << '\n';
  runtimeOut.close();

  outs() << "native_matrix_dispatch_results_tsv\t" << resultsPath << "\n";
  outs() << "native_matrix_dispatch_gate_summary_tsv\t" << gateSummaryPath
         << "\n";
  outs() << "native_matrix_dispatch_runtime_tsv\t" << runtimeSummaryPath
         << "\n";
  outs() << "native_matrix_dispatch_lanes\t" << laneTotal << "\n";
  outs() << "native_matrix_dispatch_lane_jobs\t" << matrixJobs << "\n";
  outs() << "native_matrix_dispatch_lane_schedule_policy\t"
         << cfg.laneSchedulePolicy << "\n";
  outs() << "native_matrix_dispatch_schedule_unique_keys\t"
         << scheduleUniqueKeys << "\n";
  outs() << "native_matrix_dispatch_schedule_deferred_followers\t"
         << scheduleDeferredFollowers << "\n";
  outs() << "native_matrix_dispatch_filtered_include\t" << laneFilteredInclude
         << "\n";
  outs() << "native_matrix_dispatch_filtered_exclude\t" << laneFilteredExclude
         << "\n";
  outs() << "native_matrix_dispatch_pass\t" << lanePass << "\n";
  outs() << "native_matrix_dispatch_fail\t" << laneFail << "\n";
  outs() << "native_matrix_dispatch_skip\t" << laneSkip << "\n";
  outs() << "native_matrix_dispatch_executed_lanes\t" << laneExecuted << "\n";
  outs() << "native_matrix_dispatch_runtime_ns\t" << laneRuntimeNanos << "\n";
  uint64_t avgRuntime = laneExecuted ? (laneRuntimeNanos / laneExecuted) : 0;
  outs() << "native_matrix_dispatch_avg_lane_runtime_ns\t" << avgRuntime
         << "\n";
  return laneFail == 0 ? 0 : 1;
}

static int
annotateMatrixResultsFromPrequalifySummary(ArrayRef<std::string> dispatchArgs) {
  std::string outDir = "mutation-matrix-results";
  if (auto v = getLastOptionValue(dispatchArgs, "--out-dir"))
    outDir = *v;
  std::string resultsPath = joinPath2(outDir, "results.tsv");
  if (auto v = getLastOptionValue(dispatchArgs, "--results-file"))
    resultsPath = *v;
  std::string summaryPath =
      joinPath2(outDir, "native_matrix_prequalify_summary.tsv");
  if (!sys::fs::exists(summaryPath) || !sys::fs::exists(resultsPath))
    return 0;

  StringMap<MatrixPrequalifyLaneMetrics> laneMetricsByID;
  std::string error;
  if (!loadMatrixPrequalifyLaneMetrics(summaryPath, laneMetricsByID, error)) {
    errs() << error << "\n";
    return 1;
  }
  uint64_t annotatedLaneRows = 0;
  uint64_t missingSummaryLaneRows = 0;
  if (!annotateMatrixResultsWithPrequalifyMetrics(
          resultsPath, laneMetricsByID, annotatedLaneRows,
          missingSummaryLaneRows, error)) {
    errs() << error << "\n";
    return 1;
  }
  outs() << "native_matrix_prequalify_results_tsv\t" << resultsPath << "\n";
  outs() << "native_matrix_prequalify_results_annotated_lanes\t"
         << annotatedLaneRows << "\n";
  outs() << "native_matrix_prequalify_results_missing_lanes\t"
         << missingSummaryLaneRows << "\n";
  return 0;
}

static int runMatrixFlow(const char *argv0, ArrayRef<StringRef> forwardedArgs) {
  MatrixRewriteResult rewrite = rewriteMatrixArgs(argv0, forwardedArgs);
  if (!rewrite.ok) {
    errs() << rewrite.error << "\n";
    return 1;
  }

  SmallVector<std::string, 32> dispatchArgsStorage;
  ArrayRef<std::string> dispatchArgs = rewrite.rewrittenArgs;
  if (rewrite.nativeGlobalFilterPrequalify) {
    if (runNativeMatrixGlobalFilterPrequalify(argv0, rewrite,
                                              dispatchArgsStorage) != 0)
      return 1;
    dispatchArgs = dispatchArgsStorage;
  }

  if (rewrite.nativeMatrixDispatch) {
    int rc = runNativeMatrixDispatch(argv0, dispatchArgs);
    if (rc != 0)
      return rc;
    if (!rewrite.nativeGlobalFilterPrequalify)
      return 0;
    return annotateMatrixResultsFromPrequalifySummary(dispatchArgs);
  }

  auto scriptPath = resolveScriptPath(argv0, "run_mutation_matrix.sh");
  if (!scriptPath) {
    errs() << "circt-mut: unable to locate script 'run_mutation_matrix.sh'.\n";
    errs() << "Set CIRCT_MUT_SCRIPTS_DIR or run from a build/install tree with"
              " utils scripts.\n";
    return 1;
  }

  SmallVector<StringRef, 32> rewrittenArgsRef;
  for (const std::string &arg : dispatchArgs)
    rewrittenArgsRef.push_back(arg);
  int rc = dispatchToScript(*scriptPath, rewrittenArgsRef);
  if (rc != 0)
    return rc;

  if (!rewrite.nativeGlobalFilterPrequalify)
    return 0;
  return annotateMatrixResultsFromPrequalifySummary(dispatchArgs);
}

struct InitOptions {
  std::string projectDir = ".";
  std::string design = "design.il";
  std::string mutationsFile = "mutations.txt";
  std::string testsManifest = "tests.tsv";
  std::string lanesTSV = "lanes.tsv";
  std::string coverWorkDir = "out/cover";
  std::string matrixOutDir = "out/matrix";
  bool matrixNativeDispatch = false;
  bool matrixNativeGlobalFilterPrequalify = false;
  std::string reportPolicyMode = "smoke";
  bool reportPolicyStopOnFail = true;
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

static constexpr StringLiteral kMatrixPolicyModeList =
    "smoke|nightly|strict|trend-nightly|trend-strict|native-trend-nightly|"
    "native-trend-strict|provenance-guard|provenance-strict|"
    "native-lifecycle-strict|native-smoke|native-nightly|native-strict|"
    "native-strict-formal|strict-formal|native-strict-formal-summary|"
    "strict-formal-summary|native-strict-formal-summary-v1|"
    "strict-formal-summary-v1";

static bool isMatrixPolicyMode(StringRef mode);

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
    if (arg == "--matrix-native-dispatch" ||
        arg.starts_with("--matrix-native-dispatch=")) {
      auto v = consumeValue(i, arg, "--matrix-native-dispatch");
      if (!v)
        return result;
      std::string lowered = v->trim().lower();
      if (lowered == "1" || lowered == "true" || lowered == "yes" ||
          lowered == "on") {
        result.opts.matrixNativeDispatch = true;
        continue;
      }
      if (lowered == "0" || lowered == "false" || lowered == "no" ||
          lowered == "off") {
        result.opts.matrixNativeDispatch = false;
        continue;
      }
      result.error = (Twine("circt-mut init: invalid --matrix-native-dispatch "
                            "value: ") +
                      *v + " (expected 1|0|true|false|yes|no|on|off)")
                         .str();
      return result;
    }
    if (arg == "--matrix-native-global-filter-prequalify" ||
        arg.starts_with("--matrix-native-global-filter-prequalify=")) {
      auto v = consumeValue(i, arg, "--matrix-native-global-filter-prequalify");
      if (!v)
        return result;
      std::string lowered = v->trim().lower();
      if (lowered == "1" || lowered == "true" || lowered == "yes" ||
          lowered == "on") {
        result.opts.matrixNativeGlobalFilterPrequalify = true;
        continue;
      }
      if (lowered == "0" || lowered == "false" || lowered == "no" ||
          lowered == "off") {
        result.opts.matrixNativeGlobalFilterPrequalify = false;
        continue;
      }
      result.error =
          (Twine("circt-mut init: invalid --matrix-native-global-filter-prequalify "
                 "value: ") +
           *v + " (expected 1|0|true|false|yes|no|on|off)")
              .str();
      return result;
    }
    if (arg == "--report-policy-mode" ||
        arg.starts_with("--report-policy-mode=")) {
      auto v = consumeValue(i, arg, "--report-policy-mode");
      if (!v)
        return result;
      std::string mode = v->trim().lower();
      if (!isMatrixPolicyMode(mode)) {
        result.error =
            (Twine("circt-mut init: invalid --report-policy-mode value: ") +
             *v +
             (Twine(" (expected ") + kMatrixPolicyModeList + ")"))
                .str();
        return result;
      }
      result.opts.reportPolicyMode = std::move(mode);
      continue;
    }
    if (arg == "--report-policy-stop-on-fail" ||
        arg.starts_with("--report-policy-stop-on-fail=")) {
      auto v = consumeValue(i, arg, "--report-policy-stop-on-fail");
      if (!v)
        return result;
      std::string lowered = v->trim().lower();
      if (lowered == "1" || lowered == "true" || lowered == "yes" ||
          lowered == "on") {
        result.opts.reportPolicyStopOnFail = true;
        continue;
      }
      if (lowered == "0" || lowered == "false" || lowered == "no" ||
          lowered == "off") {
        result.opts.reportPolicyStopOnFail = false;
        continue;
      }
      result.error = (Twine("circt-mut init: invalid --report-policy-stop-on-fail "
                            "value: ") +
                      *v + " (expected 1|0|true|false|yes|no|on|off)")
                         .str();
      return result;
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
  cfg << "native_matrix_dispatch = "
      << (opts.matrixNativeDispatch ? "true" : "false") << "\n";
  cfg << "native_global_filter_prequalify = "
      << (opts.matrixNativeGlobalFilterPrequalify ? "true" : "false") << "\n";
  cfg << "default_formal_global_propagate_circt_chain = \"auto\"\n";
  cfg << "default_formal_global_propagate_timeout_seconds = 60\n\n";
  cfg << "[report]\n";
  cfg << "policy_mode = \"" << escapeTomlBasicString(opts.reportPolicyMode)
      << "\"\n";
  cfg << "policy_stop_on_fail = "
      << (opts.reportPolicyStopOnFail ? "true" : "false") << "\n";
  cfg << "append_history = \"out/report-history.tsv\"\n";
  cfg << "history_max_runs = 200\n";
  cfg << "history_bootstrap = true\n";
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
  bool withReport = false;
  bool withReportOnFail = false;
  std::string reportMode;
  std::string reportCompare;
  std::string reportCompareHistoryLatest;
  std::string reportHistory;
  std::string reportAppendHistory;
  std::string reportTrendHistory;
  std::string reportTrendWindow;
  std::string reportHistoryMaxRuns;
  std::string reportOut;
  SmallVector<std::string, 4> reportFailIfValueGt;
  SmallVector<std::string, 4> reportFailIfValueLt;
  SmallVector<std::string, 4> reportFailIfDeltaGt;
  SmallVector<std::string, 4> reportFailIfDeltaLt;
  SmallVector<std::string, 4> reportFailIfTrendDeltaGt;
  SmallVector<std::string, 4> reportFailIfTrendDeltaLt;
  std::optional<bool> reportHistoryBootstrap;
  SmallVector<std::string, 4> reportPolicyProfiles;
  std::string reportPolicyMode;
  std::optional<bool> reportPolicyStopOnFail;
  SmallVector<std::string, 4> reportExternalFormalResultsFiles;
  std::string reportExternalFormalOutDir;
  std::optional<bool> reportFailOnPrequalifyDrift;
};

struct RunParseResult {
  bool ok = false;
  bool showHelp = false;
  std::string error;
  RunOptions opts;
};

struct RunConfigValues {
  StringMap<std::string> run;
  StringMap<std::string> cover;
  StringMap<std::string> matrix;
  StringMap<std::string> report;
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

    if (section == "run")
      cfg.run[key] = parsedValue;
    else if (section == "cover")
      cfg.cover[key] = parsedValue;
    else if (section == "matrix")
      cfg.matrix[key] = parsedValue;
    else if (section == "report")
      cfg.report[key] = parsedValue;
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

static bool appendMatrixPolicyModeProfiles(StringRef mode, bool stopOnFail,
                                           SmallVectorImpl<std::string> &out,
                                           std::string &error,
                                           StringRef errorPrefix);

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

static bool appendOptionalConfigBoolOptionArg(
    SmallVectorImpl<std::string> &args, const StringMap<std::string> &sectionMap,
    StringRef key, StringRef trueOptionFlag, StringRef falseOptionFlag,
    StringRef sectionName, std::string &error) {
  auto it = sectionMap.find(key);
  if (it == sectionMap.end() || it->second.empty())
    return true;
  std::string lowered = StringRef(it->second).trim().lower();
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    args.push_back(trueOptionFlag.str());
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    args.push_back(falseOptionFlag.str());
    return true;
  }
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
  auto parseBoolValue = [&](StringRef value, StringRef optName)
      -> std::optional<bool> {
    StringRef lowered = value.trim().lower();
    if (lowered == "1" || lowered == "true" || lowered == "yes" ||
        lowered == "on")
      return true;
    if (lowered == "0" || lowered == "false" || lowered == "no" ||
        lowered == "off")
      return false;
    result.error =
        (Twine("circt-mut run: invalid ") + optName + " value: " + value +
         " (expected 1|0|true|false|yes|no|on|off)")
            .str();
    return std::nullopt;
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
    if (arg == "--with-report") {
      result.opts.withReport = true;
      continue;
    }
    if (arg == "--with-report-on-fail") {
      result.opts.withReportOnFail = true;
      continue;
    }
    if (arg == "--report-mode" || arg.starts_with("--report-mode=")) {
      auto v = consumeValue(i, arg, "--report-mode");
      if (!v)
        return result;
      result.opts.reportMode = v->str();
      continue;
    }
    if (arg == "--report-compare" || arg.starts_with("--report-compare=")) {
      auto v = consumeValue(i, arg, "--report-compare");
      if (!v)
        return result;
      result.opts.reportCompare = v->str();
      continue;
    }
    if (arg == "--report-compare-history-latest" ||
        arg.starts_with("--report-compare-history-latest=")) {
      auto v = consumeValue(i, arg, "--report-compare-history-latest");
      if (!v)
        return result;
      result.opts.reportCompareHistoryLatest = v->str();
      continue;
    }
    if (arg == "--report-history" || arg.starts_with("--report-history=")) {
      auto v = consumeValue(i, arg, "--report-history");
      if (!v)
        return result;
      result.opts.reportHistory = v->str();
      continue;
    }
    if (arg == "--report-append-history" ||
        arg.starts_with("--report-append-history=")) {
      auto v = consumeValue(i, arg, "--report-append-history");
      if (!v)
        return result;
      result.opts.reportAppendHistory = v->str();
      continue;
    }
    if (arg == "--report-trend-history" ||
        arg.starts_with("--report-trend-history=")) {
      auto v = consumeValue(i, arg, "--report-trend-history");
      if (!v)
        return result;
      result.opts.reportTrendHistory = v->str();
      continue;
    }
    if (arg == "--report-trend-window" ||
        arg.starts_with("--report-trend-window=")) {
      auto v = consumeValue(i, arg, "--report-trend-window");
      if (!v)
        return result;
      result.opts.reportTrendWindow = v->str();
      continue;
    }
    if (arg == "--report-history-max-runs" ||
        arg.starts_with("--report-history-max-runs=")) {
      auto v = consumeValue(i, arg, "--report-history-max-runs");
      if (!v)
        return result;
      result.opts.reportHistoryMaxRuns = v->str();
      continue;
    }
    if (arg == "--report-out" || arg.starts_with("--report-out=")) {
      auto v = consumeValue(i, arg, "--report-out");
      if (!v)
        return result;
      result.opts.reportOut = v->str();
      continue;
    }
    if (arg == "--report-fail-if-value-gt" ||
        arg.starts_with("--report-fail-if-value-gt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-value-gt");
      if (!v)
        return result;
      result.opts.reportFailIfValueGt.push_back(v->str());
      continue;
    }
    if (arg == "--report-fail-if-value-lt" ||
        arg.starts_with("--report-fail-if-value-lt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-value-lt");
      if (!v)
        return result;
      result.opts.reportFailIfValueLt.push_back(v->str());
      continue;
    }
    if (arg == "--report-fail-if-delta-gt" ||
        arg.starts_with("--report-fail-if-delta-gt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-delta-gt");
      if (!v)
        return result;
      result.opts.reportFailIfDeltaGt.push_back(v->str());
      continue;
    }
    if (arg == "--report-fail-if-delta-lt" ||
        arg.starts_with("--report-fail-if-delta-lt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-delta-lt");
      if (!v)
        return result;
      result.opts.reportFailIfDeltaLt.push_back(v->str());
      continue;
    }
    if (arg == "--report-fail-if-trend-delta-gt" ||
        arg.starts_with("--report-fail-if-trend-delta-gt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-trend-delta-gt");
      if (!v)
        return result;
      result.opts.reportFailIfTrendDeltaGt.push_back(v->str());
      continue;
    }
    if (arg == "--report-fail-if-trend-delta-lt" ||
        arg.starts_with("--report-fail-if-trend-delta-lt=")) {
      auto v = consumeValue(i, arg, "--report-fail-if-trend-delta-lt");
      if (!v)
        return result;
      result.opts.reportFailIfTrendDeltaLt.push_back(v->str());
      continue;
    }
    if (arg == "--report-history-bootstrap") {
      result.opts.reportHistoryBootstrap = true;
      continue;
    }
    if (arg == "--report-no-history-bootstrap") {
      result.opts.reportHistoryBootstrap = false;
      continue;
    }
    if (arg == "--report-policy-profile" ||
        arg.starts_with("--report-policy-profile=")) {
      auto v = consumeValue(i, arg, "--report-policy-profile");
      if (!v)
        return result;
      StringRef profile = StringRef(*v).trim();
      if (profile.empty()) {
        result.error =
            "circt-mut run: --report-policy-profile requires non-empty value";
        return result;
      }
      result.opts.reportPolicyProfiles.push_back(profile.str());
      continue;
    }
    if (arg == "--report-policy-mode" ||
        arg.starts_with("--report-policy-mode=")) {
      auto v = consumeValue(i, arg, "--report-policy-mode");
      if (!v)
        return result;
      std::string mode = StringRef(*v).trim().lower();
      if (!isMatrixPolicyMode(mode)) {
        result.error = (Twine("circt-mut run: invalid --report-policy-mode "
                              "value: ") +
                        *v +
                        (Twine(" (expected ") + kMatrixPolicyModeList + ")"))
                            .str();
        return result;
      }
      result.opts.reportPolicyMode = mode;
      continue;
    }
    if (arg == "--report-policy-stop-on-fail" ||
        arg.starts_with("--report-policy-stop-on-fail=")) {
      auto v = consumeValue(i, arg, "--report-policy-stop-on-fail");
      if (!v)
        return result;
      auto parsed = parseBoolValue(*v, "--report-policy-stop-on-fail");
      if (!parsed)
        return result;
      result.opts.reportPolicyStopOnFail = *parsed;
      continue;
    }
    if (arg == "--report-external-formal-results" ||
        arg.starts_with("--report-external-formal-results=")) {
      auto v = consumeValue(i, arg, "--report-external-formal-results");
      if (!v)
        return result;
      StringRef path = v->trim();
      if (path.empty()) {
        result.error = "circt-mut run: --report-external-formal-results "
                       "requires non-empty value";
        return result;
      }
      result.opts.reportExternalFormalResultsFiles.push_back(path.str());
      continue;
    }
    if (arg == "--report-external-formal-out-dir" ||
        arg.starts_with("--report-external-formal-out-dir=")) {
      auto v = consumeValue(i, arg, "--report-external-formal-out-dir");
      if (!v)
        return result;
      StringRef path = v->trim();
      if (path.empty()) {
        result.error = "circt-mut run: --report-external-formal-out-dir "
                       "requires non-empty value";
        return result;
      }
      result.opts.reportExternalFormalOutDir = path.str();
      continue;
    }
    if (arg == "--report-fail-on-prequalify-drift") {
      result.opts.reportFailOnPrequalifyDrift = true;
      continue;
    }
    if (arg == "--report-no-fail-on-prequalify-drift") {
      result.opts.reportFailOnPrequalifyDrift = false;
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
  if (!result.opts.reportMode.empty() && result.opts.reportMode != "cover" &&
      result.opts.reportMode != "matrix" && result.opts.reportMode != "all") {
    result.error =
        (Twine("circt-mut run: invalid --report-mode value: ") +
         result.opts.reportMode + " (expected cover|matrix|all)")
            .str();
    return result;
  }
  std::string effectiveReportMode =
      result.opts.reportMode.empty() ? result.opts.mode : result.opts.reportMode;
  if (!result.opts.reportCompare.empty() &&
      !result.opts.reportCompareHistoryLatest.empty()) {
    result.error = "circt-mut run: --report-compare and "
                   "--report-compare-history-latest are mutually exclusive";
    return result;
  }
  if (!result.opts.reportHistory.empty() &&
      (!result.opts.reportCompare.empty() ||
       !result.opts.reportCompareHistoryLatest.empty() ||
       !result.opts.reportTrendHistory.empty() ||
       !result.opts.reportAppendHistory.empty())) {
    result.error = "circt-mut run: --report-history is mutually exclusive with "
                   "--report-compare, --report-compare-history-latest, "
                   "--report-trend-history, and --report-append-history";
    return result;
  }
  if (!result.opts.reportPolicyMode.empty() &&
      !result.opts.reportPolicyProfiles.empty()) {
    result.error = "circt-mut run: --report-policy-mode and "
                   "--report-policy-profile are mutually exclusive";
    return result;
  }
  if (!result.opts.reportPolicyMode.empty() &&
      effectiveReportMode != "matrix" && effectiveReportMode != "all") {
    result.error = "circt-mut run: --report-policy-mode requires "
                   "--report-mode matrix|all (or --mode matrix|all when "
                   "--report-mode is unset)";
    return result;
  }
  if (result.opts.reportPolicyStopOnFail.has_value() &&
      result.opts.reportPolicyMode.empty()) {
    result.error = "circt-mut run: --report-policy-stop-on-fail requires "
                   "--report-policy-mode";
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

  bool withReport = opts.withReport;
  if (!withReport) {
    auto it = cfg.run.find("with_report");
    if (it != cfg.run.end() && !it->second.empty()) {
      std::string lowered = StringRef(it->second).trim().lower();
      if (lowered == "1" || lowered == "true" || lowered == "yes" ||
          lowered == "on")
        withReport = true;
      else if (lowered == "0" || lowered == "false" || lowered == "no" ||
               lowered == "off")
        withReport = false;
      else {
        errs() << "circt-mut run: invalid boolean [run] key 'with_report' "
                  "value '"
               << it->second
               << "' (expected 1|0|true|false|yes|no|on|off)\n";
        return 1;
      }
    }
  }
  bool withReportOnFail = opts.withReportOnFail;
  if (!withReportOnFail) {
    auto it = cfg.run.find("with_report_on_fail");
    if (it != cfg.run.end() && !it->second.empty()) {
      std::string lowered = StringRef(it->second).trim().lower();
      if (lowered == "1" || lowered == "true" || lowered == "yes" ||
          lowered == "on")
        withReportOnFail = true;
      else if (lowered == "0" || lowered == "false" || lowered == "no" ||
               lowered == "off")
        withReportOnFail = false;
      else {
        errs() << "circt-mut run: invalid boolean [run] key "
                  "'with_report_on_fail' value '"
               << it->second
               << "' (expected 1|0|true|false|yes|no|on|off)\n";
        return 1;
      }
    }
  }
  std::string reportMode = opts.reportMode;
  if (reportMode.empty()) {
    auto it = cfg.run.find("report_mode");
    if (it != cfg.run.end() && !it->second.empty())
      reportMode = StringRef(it->second).trim().str();
  }
  if (reportMode.empty())
    reportMode = opts.mode;
  if (reportMode != "cover" && reportMode != "matrix" && reportMode != "all") {
    errs() << "circt-mut run: invalid [run] key 'report_mode' value '"
           << reportMode << "' (expected cover|matrix|all)\n";
    return 1;
  }
  if (!withReport) {
    if (!opts.reportMode.empty() || !opts.reportCompare.empty() ||
        !opts.reportCompareHistoryLatest.empty() ||
        !opts.reportHistory.empty() || !opts.reportAppendHistory.empty() ||
        !opts.reportTrendHistory.empty() || !opts.reportTrendWindow.empty() ||
        !opts.reportHistoryMaxRuns.empty() || !opts.reportOut.empty() ||
        !opts.reportFailIfValueGt.empty() || !opts.reportFailIfValueLt.empty() ||
        !opts.reportFailIfDeltaGt.empty() || !opts.reportFailIfDeltaLt.empty() ||
        !opts.reportFailIfTrendDeltaGt.empty() ||
        !opts.reportFailIfTrendDeltaLt.empty() ||
        opts.reportHistoryBootstrap.has_value() || !opts.reportPolicyProfiles.empty() ||
        !opts.reportPolicyMode.empty() ||
        opts.reportPolicyStopOnFail.has_value() ||
        !opts.reportExternalFormalResultsFiles.empty() ||
        !opts.reportExternalFormalOutDir.empty() ||
        opts.reportFailOnPrequalifyDrift.has_value()) {
      errs() << "circt-mut run: report override options require "
                "--with-report or [run] with_report = true\n";
      return 1;
    }
    if (opts.withReportOnFail) {
      errs() << "circt-mut run: --with-report-on-fail requires "
                "--with-report or [run] with_report = true\n";
      return 1;
    }
  }
  SmallVector<std::string, 48> reportArgsOwned;
  if (withReport) {
    bool hasCLIReportCompare = !opts.reportCompare.empty();
    bool hasCLIReportCompareHistoryLatest =
        !opts.reportCompareHistoryLatest.empty();
    bool hasCLIReportHistory = !opts.reportHistory.empty();
    bool hasCLIReportAppendHistory = !opts.reportAppendHistory.empty();
    bool hasCLIReportTrendHistory = !opts.reportTrendHistory.empty();
    if (hasCLIReportCompare && hasCLIReportCompareHistoryLatest) {
      errs() << "circt-mut run: --report-compare and "
                "--report-compare-history-latest are mutually exclusive\n";
      return 1;
    }
    auto runCompareIt = cfg.run.find("report_compare");
    auto runCompareHistoryIt = cfg.run.find("report_compare_history_latest");
    bool hasConfigReportCompare =
        runCompareIt != cfg.run.end() && !runCompareIt->second.empty();
    bool hasConfigReportCompareHistory =
        runCompareHistoryIt != cfg.run.end() && !runCompareHistoryIt->second.empty();
    auto runHistoryIt = cfg.run.find("report_history");
    auto runAppendHistoryIt = cfg.run.find("report_append_history");
    auto runTrendHistoryIt = cfg.run.find("report_trend_history");
    bool hasConfigReportHistory =
        runHistoryIt != cfg.run.end() && !runHistoryIt->second.empty();
    bool hasConfigReportAppendHistory =
        runAppendHistoryIt != cfg.run.end() && !runAppendHistoryIt->second.empty();
    bool hasConfigReportTrendHistory =
        runTrendHistoryIt != cfg.run.end() && !runTrendHistoryIt->second.empty();
    if (!hasCLIReportCompare && !hasCLIReportCompareHistoryLatest &&
        hasConfigReportCompare && hasConfigReportCompareHistory) {
      errs() << "circt-mut run: [run] keys 'report_compare' and "
                "'report_compare_history_latest' are mutually exclusive\n";
      return 1;
    }
    bool hasAnyReportHistoryShorthand =
        hasCLIReportHistory || hasConfigReportHistory;
    bool hasAnyReportCompareSelector =
        hasCLIReportCompare || hasCLIReportCompareHistoryLatest ||
        hasConfigReportCompare || hasConfigReportCompareHistory;
    bool hasAnyReportAppendSelector =
        hasCLIReportAppendHistory || hasConfigReportAppendHistory;
    bool hasAnyReportTrendSelector =
        hasCLIReportTrendHistory || hasConfigReportTrendHistory;
    if (hasAnyReportHistoryShorthand &&
        (hasAnyReportCompareSelector || hasAnyReportAppendSelector ||
         hasAnyReportTrendSelector)) {
      errs() << "circt-mut run: report_history shorthand is mutually "
                "exclusive with report compare/trend/append selectors "
                "(--report-history/[run] report_history vs "
                "--report-compare/--report-compare-history-latest/"
                "--report-trend-history/--report-append-history)\n";
      return 1;
    }

    reportArgsOwned.push_back("report");
    reportArgsOwned.push_back("--project-dir");
    reportArgsOwned.push_back(opts.projectDir);
    reportArgsOwned.push_back("--config");
    reportArgsOwned.push_back(std::string(configPath));
    reportArgsOwned.push_back("--mode");
    reportArgsOwned.push_back(reportMode);

    if (!opts.reportExternalFormalResultsFiles.empty()) {
      for (const auto &path : opts.reportExternalFormalResultsFiles) {
        reportArgsOwned.push_back("--external-formal-results");
        reportArgsOwned.push_back(path);
      }
    } else {
      auto extFormalIt = cfg.run.find("report_external_formal_results");
      if (extFormalIt != cfg.run.end() && !extFormalIt->second.empty()) {
        SmallVector<StringRef, 8> tokens;
        StringRef(extFormalIt->second)
            .split(tokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (StringRef raw : tokens) {
          StringRef token = raw.trim();
          if (token.empty())
            continue;
          std::string value = token.str();
          if (!sys::path::is_absolute(value) && StringRef(value).contains('/')) {
            SmallString<256> joined(opts.projectDir);
            sys::path::append(joined, value);
            value = std::string(joined.str());
          }
          reportArgsOwned.push_back("--external-formal-results");
          reportArgsOwned.push_back(value);
        }
      }
    }
    if (!opts.reportExternalFormalOutDir.empty()) {
      reportArgsOwned.push_back("--external-formal-out-dir");
      reportArgsOwned.push_back(opts.reportExternalFormalOutDir);
    } else {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run,
                                  "report_external_formal_out_dir",
                                  "--external-formal-out-dir",
                                  opts.projectDir);
    }

    auto appendRunReportCSV = [&](StringRef key) -> bool {
      auto it = cfg.run.find(key);
      if (it == cfg.run.end() || it->second.empty())
        return false;
      bool appended = false;
      SmallVector<StringRef, 8> entries;
      StringRef(it->second).split(entries, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
      for (StringRef raw : entries) {
        StringRef token = raw.trim();
        if (token.empty())
          continue;
        reportArgsOwned.push_back("--policy-profile");
        reportArgsOwned.push_back(token.str());
        appended = true;
      }
      return appended;
    };
    bool hasCLIReportPolicyMode = !opts.reportPolicyMode.empty();
    bool hasCLIReportPolicyProfile = !opts.reportPolicyProfiles.empty();
    auto runPolicyStopOnFailIt = cfg.run.find("report_policy_stop_on_fail");
    bool hasConfigReportPolicyStopOnFail =
        runPolicyStopOnFailIt != cfg.run.end() &&
        !runPolicyStopOnFailIt->second.empty();
    auto runPolicyModeIt = cfg.run.find("report_policy_mode");
    bool hasConfigReportPolicyMode =
        runPolicyModeIt != cfg.run.end() && !runPolicyModeIt->second.empty();
    auto runPolicyProfileIt = cfg.run.find("report_policy_profile");
    bool hasConfigReportPolicyProfile =
        runPolicyProfileIt != cfg.run.end() && !runPolicyProfileIt->second.empty();
    auto runPolicyProfilesIt = cfg.run.find("report_policy_profiles");
    bool hasConfigReportPolicyProfiles =
        runPolicyProfilesIt != cfg.run.end() &&
        !runPolicyProfilesIt->second.empty();
    if (!hasCLIReportPolicyMode && !hasCLIReportPolicyProfile &&
        hasConfigReportPolicyMode &&
        (hasConfigReportPolicyProfile || hasConfigReportPolicyProfiles)) {
      errs() << "circt-mut run: [run] keys 'report_policy_mode' and "
                "'report_policy_profile(s)' are mutually exclusive\n";
      return 1;
    }
    if (!hasCLIReportPolicyMode && hasConfigReportPolicyStopOnFail &&
        !hasConfigReportPolicyMode) {
      errs() << "circt-mut run: [run] key 'report_policy_stop_on_fail' "
                "requires 'report_policy_mode'\n";
      return 1;
    }
    bool hasExplicitPolicyProfile = false;
    if (hasCLIReportPolicyProfile) {
      for (const auto &profile : opts.reportPolicyProfiles) {
        reportArgsOwned.push_back("--policy-profile");
        reportArgsOwned.push_back(profile);
      }
      hasExplicitPolicyProfile = true;
    } else if (!hasCLIReportPolicyMode) {
      hasExplicitPolicyProfile = appendRunReportCSV("report_policy_profile");
      hasExplicitPolicyProfile |= appendRunReportCSV("report_policy_profiles");
    }
    if (!hasExplicitPolicyProfile) {
      bool hasPolicyMode = false;
      std::string mode;
      std::optional<bool> stopOnFail;
      if (hasCLIReportPolicyMode) {
        hasPolicyMode = true;
        mode = opts.reportPolicyMode;
        if (opts.reportPolicyStopOnFail.has_value())
          stopOnFail = opts.reportPolicyStopOnFail;
      } else {
        auto policyModeIt = cfg.run.find("report_policy_mode");
        auto policyStopOnFailIt = cfg.run.find("report_policy_stop_on_fail");
        hasPolicyMode =
            policyModeIt != cfg.run.end() && !policyModeIt->second.empty();
        bool hasPolicyStopOnFail = policyStopOnFailIt != cfg.run.end() &&
                                   !policyStopOnFailIt->second.empty();
        if (hasPolicyStopOnFail && !hasPolicyMode) {
          errs() << "circt-mut run: [run] key 'report_policy_stop_on_fail' "
                    "requires 'report_policy_mode'\n";
          return 1;
        }
        if (hasPolicyMode)
          mode = StringRef(policyModeIt->second).trim().lower();
        if (hasPolicyStopOnFail) {
          StringRef raw = StringRef(policyStopOnFailIt->second).trim().lower();
          if (raw == "1" || raw == "true" || raw == "yes" || raw == "on")
            stopOnFail = true;
          else if (raw == "0" || raw == "false" || raw == "no" ||
                   raw == "off")
            stopOnFail = false;
          else {
            errs() << "circt-mut run: invalid boolean [run] key "
                      "'report_policy_stop_on_fail' value '"
                   << policyStopOnFailIt->second
                   << "' (expected 1|0|true|false|yes|no|on|off)\n";
            return 1;
          }
        }
      }
      if (hasPolicyMode) {
        if (reportMode != "matrix" && reportMode != "all") {
          errs() << "circt-mut run: report policy mode requires "
                    "'report_mode = matrix|all'\n";
          return 1;
        }
        if (!isMatrixPolicyMode(mode)) {
          errs() << "circt-mut run: invalid report policy mode value '"
                 << mode << "' (expected " << kMatrixPolicyModeList << ")\n";
          return 1;
        }
        bool stop = stopOnFail.value_or(false);
        reportArgsOwned.push_back("--policy-mode");
        reportArgsOwned.push_back(mode);
        reportArgsOwned.push_back("--policy-stop-on-fail");
        reportArgsOwned.push_back(stop ? "true" : "false");
      }
    }

    if (hasCLIReportCompare) {
      reportArgsOwned.push_back("--compare");
      reportArgsOwned.push_back(opts.reportCompare);
    } else if (!hasCLIReportCompareHistoryLatest) {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_compare",
                                  "--compare", opts.projectDir);
    }
    if (hasCLIReportCompareHistoryLatest) {
      reportArgsOwned.push_back("--compare-history-latest");
      reportArgsOwned.push_back(opts.reportCompareHistoryLatest);
    } else if (!hasCLIReportCompare) {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run,
                                  "report_compare_history_latest",
                                  "--compare-history-latest", opts.projectDir);
    }
    if (!opts.reportHistory.empty()) {
      reportArgsOwned.push_back("--history");
      reportArgsOwned.push_back(opts.reportHistory);
    } else {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_history",
                                  "--history", opts.projectDir);
    }
    if (!opts.reportAppendHistory.empty()) {
      reportArgsOwned.push_back("--append-history");
      reportArgsOwned.push_back(opts.reportAppendHistory);
    } else {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run,
                                  "report_append_history", "--append-history",
                                  opts.projectDir);
    }
    if (!opts.reportTrendHistory.empty()) {
      reportArgsOwned.push_back("--trend-history");
      reportArgsOwned.push_back(opts.reportTrendHistory);
    } else {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run,
                                  "report_trend_history", "--trend-history",
                                  opts.projectDir);
    }
    if (!opts.reportHistoryMaxRuns.empty()) {
      reportArgsOwned.push_back("--history-max-runs");
      reportArgsOwned.push_back(opts.reportHistoryMaxRuns);
    } else {
      appendOptionalConfigArg(reportArgsOwned, cfg.run, "report_history_max_runs",
                              "--history-max-runs");
    }
    if (!opts.reportTrendWindow.empty()) {
      reportArgsOwned.push_back("--trend-window");
      reportArgsOwned.push_back(opts.reportTrendWindow);
    } else {
      appendOptionalConfigArg(reportArgsOwned, cfg.run, "report_trend_window",
                              "--trend-window");
    }
    appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_cover_work_dir",
                                "--cover-work-dir", opts.projectDir);
    appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_matrix_out_dir",
                                "--matrix-out-dir", opts.projectDir);
    appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_lane_budget_out",
                                "--lane-budget-out", opts.projectDir);
    appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_skip_budget_out",
                                "--skip-budget-out", opts.projectDir);
    if (!opts.reportOut.empty()) {
      reportArgsOwned.push_back("--out");
      reportArgsOwned.push_back(opts.reportOut);
    } else {
      appendOptionalConfigPathArg(reportArgsOwned, cfg.run, "report_out",
                                  "--out", opts.projectDir);
    }
    auto appendRulesFromCSV = [&](StringRef key, StringRef optionFlag) {
      auto it = cfg.run.find(key);
      if (it == cfg.run.end() || it->second.empty())
        return;
      SmallVector<StringRef, 8> entries;
      StringRef(it->second).split(entries, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
      for (StringRef raw : entries) {
        StringRef token = raw.trim();
        if (token.empty())
          continue;
        reportArgsOwned.push_back(optionFlag.str());
        reportArgsOwned.push_back(token.str());
      }
    };
    auto appendRuleOverridesOrConfig = [&](const SmallVectorImpl<std::string> &cliRules,
                                           StringRef configKey,
                                           StringRef optionFlag) {
      if (!cliRules.empty()) {
        for (const auto &rule : cliRules) {
          reportArgsOwned.push_back(optionFlag.str());
          reportArgsOwned.push_back(rule);
        }
        return;
      }
      appendRulesFromCSV(configKey, optionFlag);
    };
    appendRuleOverridesOrConfig(opts.reportFailIfValueGt,
                                "report_fail_if_value_gt",
                                "--fail-if-value-gt");
    appendRuleOverridesOrConfig(opts.reportFailIfValueLt,
                                "report_fail_if_value_lt",
                                "--fail-if-value-lt");
    appendRuleOverridesOrConfig(opts.reportFailIfDeltaGt,
                                "report_fail_if_delta_gt",
                                "--fail-if-delta-gt");
    appendRuleOverridesOrConfig(opts.reportFailIfDeltaLt,
                                "report_fail_if_delta_lt",
                                "--fail-if-delta-lt");
    appendRuleOverridesOrConfig(opts.reportFailIfTrendDeltaGt,
                                "report_fail_if_trend_delta_gt",
                                "--fail-if-trend-delta-gt");
    appendRuleOverridesOrConfig(opts.reportFailIfTrendDeltaLt,
                                "report_fail_if_trend_delta_lt",
                                "--fail-if-trend-delta-lt");

    if (opts.reportHistoryBootstrap.has_value()) {
      if (*opts.reportHistoryBootstrap)
        reportArgsOwned.push_back("--history-bootstrap");
    } else if (!appendOptionalConfigBoolFlagArg(reportArgsOwned, cfg.run,
                                                "report_history_bootstrap",
                                                "--history-bootstrap", "run",
                                                error)) {
      errs() << error << "\n";
      return 1;
    }
    if (opts.reportFailOnPrequalifyDrift.has_value()) {
      reportArgsOwned.push_back(*opts.reportFailOnPrequalifyDrift
                                    ? "--fail-on-prequalify-drift"
                                    : "--no-fail-on-prequalify-drift");
    } else if (!appendOptionalConfigBoolOptionArg(
                   reportArgsOwned, cfg.run, "report_fail_on_prequalify_drift",
                   "--fail-on-prequalify-drift",
                   "--no-fail-on-prequalify-drift", "run", error)) {
      errs() << error << "\n";
      return 1;
    }
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
    if (!appendOptionalConfigBoolFlagArg(args, cfg.cover,
                                         "native_global_filter_prequalify",
                                         "--native-global-filter-prequalify",
                                         "cover", error)) {
      errs() << error << "\n";
      return 1;
    }
    if (!appendOptionalConfigBoolFlagArg(
            args, cfg.cover, "native_global_filter_prequalify_only",
            "--native-global-filter-prequalify-only", "cover", error)) {
      errs() << error << "\n";
      return 1;
    }
    appendOptionalConfigPathArg(args, cfg.cover,
                                "native_global_filter_prequalify_pair_file",
                                "--native-global-filter-prequalify-pair-file",
                                opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.cover,
                                "native_global_filter_probe_mutant",
                                "--native-global-filter-probe-mutant",
                                opts.projectDir);
    appendOptionalConfigPathArg(args, cfg.cover,
                                "native_global_filter_probe_log",
                                "--native-global-filter-probe-log",
                                opts.projectDir);

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
                                         "--stop-on-fail", "matrix", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.matrix,
                                         "native_global_filter_prequalify",
                                         "--native-global-filter-prequalify",
                                         "matrix", error) ||
        !appendOptionalConfigBoolFlagArg(args, cfg.matrix,
                                         "native_matrix_dispatch",
                                         "--native-matrix-dispatch", "matrix",
                                         error)) {
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
    appendOptionalConfigArg(args, cfg.matrix, "default_mutations_seed",
                            "--default-mutations-seed");
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

  int rc = 0;
  if (opts.mode == "cover") {
    rc = runCoverFromConfig();
  } else if (opts.mode == "matrix") {
    rc = runMatrixFromConfig();
  } else {
    rc = runCoverFromConfig();
    if (rc == 0)
      rc = runMatrixFromConfig();
  }
  if (!withReport)
    return rc;
  if (rc != 0 && !withReportOnFail)
    return rc;

  std::string mainExec =
      sys::fs::getMainExecutable(argv0, reinterpret_cast<void *>(&printHelp));
  if (mainExec.empty()) {
    errs() << "circt-mut run: unable to locate circt-mut executable for "
              "post-run report.\n";
    return 1;
  }
  SmallVector<std::string, 48> reportExecArgsOwned;
  reportExecArgsOwned.push_back(mainExec);
  reportExecArgsOwned.append(reportArgsOwned.begin(), reportArgsOwned.end());
  SmallVector<StringRef, 48> reportArgs;
  for (const auto &arg : reportExecArgsOwned)
    reportArgs.push_back(arg);

  std::string errMsg;
  int reportRC = sys::ExecuteAndWait(mainExec, reportArgs, /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                     &errMsg);
  if (!errMsg.empty())
    errs() << "circt-mut run: post-run report execution error: " << errMsg
           << "\n";
  if (rc != 0)
    return rc;
  return reportRC;
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
  std::string historyFile;
  bool historyBootstrap = false;
  uint64_t historyMaxRuns = 0;
  std::string trendHistoryFile;
  uint64_t trendWindowRuns = 0;
  SmallVector<std::string, 4> policyProfiles;
  std::string policyMode;
  std::optional<bool> policyStopOnFail;
  SmallVector<std::string, 4> externalFormalResultsFiles;
  std::string externalFormalOutDir;
  std::string appendHistoryFile;
  SmallVector<DeltaGateRule, 8> failIfValueGtRules;
  SmallVector<DeltaGateRule, 8> failIfValueLtRules;
  SmallVector<DeltaGateRule, 8> failIfDeltaGtRules;
  SmallVector<DeltaGateRule, 8> failIfDeltaLtRules;
  SmallVector<DeltaGateRule, 8> failIfTrendDeltaGtRules;
  SmallVector<DeltaGateRule, 8> failIfTrendDeltaLtRules;
  bool failOnPrequalifyDrift = false;
  bool failOnPrequalifyDriftOverrideSet = false;
  std::string laneBudgetOutFile;
  std::string skipBudgetOutFile;
  std::string outFile;
};

struct ExternalFormalSummary {
  uint64_t files = 0;
  uint64_t lines = 0;
  uint64_t parsedStatusLines = 0;
  uint64_t parsedSummaryLines = 0;
  uint64_t unparsedLines = 0;
  uint64_t pass = 0;
  uint64_t fail = 0;
  uint64_t error = 0;
  uint64_t skip = 0;
  uint64_t xfail = 0;
  uint64_t xpass = 0;
  uint64_t summaryTotal = 0;
  uint64_t summaryPass = 0;
  uint64_t summaryFail = 0;
  uint64_t summaryError = 0;
  uint64_t summarySkip = 0;
  uint64_t summaryXFail = 0;
  uint64_t summaryXPass = 0;
  uint64_t summaryTSVFiles = 0;
  uint64_t summaryTSVRows = 0;
  uint64_t summaryTSVSchemaValidFiles = 0;
  uint64_t summaryTSVSchemaInvalidFiles = 0;
  uint64_t summaryTSVParseErrors = 0;
  uint64_t summaryTSVConsistentRows = 0;
  uint64_t summaryTSVInconsistentRows = 0;
  uint64_t summaryTSVSchemaVersionRows = 0;
  uint64_t summaryTSVSchemaVersionInvalidRows = 0;
  uint64_t summaryTSVSchemaVersionMin = 0;
  uint64_t summaryTSVSchemaVersionMax = 0;
  uint64_t summaryTSVDuplicateRows = 0;
  uint64_t summaryTSVUniqueRows = 0;
  StringMap<uint64_t> summaryCounterSums;
  std::map<std::pair<std::string, std::string>, StringMap<uint64_t>>
      summaryCounterSumsBySuiteMode;
};

struct MatrixLaneBudgetRow {
  std::string laneID;
  std::string status;
  std::string gateStatus;
  bool hasMetrics = false;
  bool hasPrequalifySummary = false;
  std::string prequalifyPairFile = "-";
  std::string prequalifyLogFile = "-";
  uint64_t prequalifyTotalMutants = 0;
  uint64_t prequalifyNotPropagatedMutants = 0;
  uint64_t prequalifyPropagatedMutants = 0;
  uint64_t prequalifyCreateMutatedErrorMutants = 0;
  uint64_t prequalifyProbeErrorMutants = 0;
  uint64_t prequalifyCmdTokenNotPropagatedMutants = 0;
  uint64_t prequalifyCmdTokenPropagatedMutants = 0;
  uint64_t prequalifyCmdRCNotPropagatedMutants = 0;
  uint64_t prequalifyCmdRCPropagatedMutants = 0;
  uint64_t prequalifyCmdTimeoutPropagatedMutants = 0;
  uint64_t prequalifyCmdErrorMutants = 0;
  uint64_t detectedMutants = 0;
  uint64_t errors = 0;
  uint64_t timeoutMutants = 0;
  uint64_t lecUnknownMutants = 0;
  uint64_t bmcUnknownMutants = 0;
  std::string configErrorCode;
  std::string configErrorReason;
  bool hasRuntimeNanos = false;
  uint64_t runtimeNanos = 0;
};

struct SkipBudgetSummary {
  uint64_t totalRows = 0;
  uint64_t skipRows = 0;
  uint64_t nonSkipRows = 0;
  uint64_t stopOnFailRows = 0;
  uint64_t nonStopOnFailSkipRows = 0;
  uint64_t rowsWithReason = 0;
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

static void splitWhitespace(StringRef line, SmallVectorImpl<StringRef> &tokens) {
  tokens.clear();
  size_t idx = 0;
  while (idx < line.size()) {
    while (idx < line.size() &&
           std::isspace(static_cast<unsigned char>(line[idx])))
      ++idx;
    if (idx >= line.size())
      break;
    size_t start = idx;
    while (idx < line.size() &&
           !std::isspace(static_cast<unsigned char>(line[idx])))
      ++idx;
    tokens.push_back(line.slice(start, idx));
  }
}

static std::optional<uint64_t> parseUnsignedTokenValue(StringRef token) {
  token = token.trim();
  while (!token.empty() &&
         !std::isdigit(static_cast<unsigned char>(token.back())))
    token = token.drop_back();
  while (!token.empty() &&
         !std::isdigit(static_cast<unsigned char>(token.front())))
    token = token.drop_front();
  if (token.empty())
    return std::nullopt;
  uint64_t value = 0;
  if (token.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static bool parseSummaryTokenCount(StringRef token, StringRef key,
                                   uint64_t &valueOut) {
  if (!token.consume_front(key))
    return false;
  if (!token.consume_front("="))
    return false;
  auto parsed = parseUnsignedTokenValue(token);
  if (!parsed)
    return false;
  valueOut = *parsed;
  return true;
}

static std::string sanitizeReportKeySegment(StringRef raw);

static bool collectExternalFormalSummary(
    ArrayRef<std::string> files, std::vector<std::pair<std::string, std::string>> &rows,
    std::string &error) {
  ExternalFormalSummary summary;
  SmallVector<StringRef, 32> tokens;
  SmallVector<StringRef, 32> fields;
  for (const auto &path : files) {
    auto bufferOrErr = MemoryBuffer::getFile(path);
    if (!bufferOrErr) {
      error = (Twine("circt-mut report: unable to read external formal results file: ") +
               path)
                  .str();
      return false;
    }
    ++summary.files;
    SmallVector<StringRef, 256> lines;
    bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                         /*KeepEmpty=*/false);
    StringRef baseName = sys::path::filename(path);
    if (baseName == "summary.tsv") {
      ++summary.summaryTSVFiles;
      bool headerSeen = false;
      bool schemaValid = false;
      bool schemaInvalid = false;
      size_t totalCol = static_cast<size_t>(-1);
      size_t passCol = static_cast<size_t>(-1);
      size_t failCol = static_cast<size_t>(-1);
      size_t xfailCol = static_cast<size_t>(-1);
      size_t xpassCol = static_cast<size_t>(-1);
      size_t errorCol = static_cast<size_t>(-1);
      size_t skipCol = static_cast<size_t>(-1);
      size_t schemaVersionCol = static_cast<size_t>(-1);
      size_t suiteCol = static_cast<size_t>(-1);
      size_t modeCol = static_cast<size_t>(-1);
      size_t summaryCol = static_cast<size_t>(-1);
      StringSet<> seenSummaryRows;
      auto parseCountField = [&](StringRef field,
                                 uint64_t &out) -> bool {
        StringRef token = field.trim();
        if (token.empty())
          return false;
        return !token.getAsInteger(10, out);
      };
      for (StringRef rawLine : lines) {
        StringRef line = rawLine.rtrim("\r").trim();
        if (line.empty() || line.starts_with("#"))
          continue;
        ++summary.lines;
        splitTSVLine(line, fields);
        if (!headerSeen) {
          headerSeen = true;
          auto findCol = [&](StringRef name) -> size_t {
            for (size_t i = 0, e = fields.size(); i < e; ++i)
              if (fields[i].trim().lower() == name)
                return i;
            return static_cast<size_t>(-1);
          };
          totalCol = findCol("total");
          passCol = findCol("pass");
          failCol = findCol("fail");
          xfailCol = findCol("xfail");
          xpassCol = findCol("xpass");
          errorCol = findCol("error");
          skipCol = findCol("skip");
          schemaVersionCol = findCol("schema_version");
          suiteCol = findCol("suite");
          modeCol = findCol("mode");
          summaryCol = findCol("summary");
          schemaValid = totalCol != static_cast<size_t>(-1) &&
                        passCol != static_cast<size_t>(-1) &&
                        failCol != static_cast<size_t>(-1) &&
                        xfailCol != static_cast<size_t>(-1) &&
                        xpassCol != static_cast<size_t>(-1) &&
                        errorCol != static_cast<size_t>(-1) &&
                        skipCol != static_cast<size_t>(-1);
          if (!schemaValid) {
            schemaInvalid = true;
            ++summary.summaryTSVParseErrors;
          }
          continue;
        }
        if (!schemaValid) {
          ++summary.unparsedLines;
          continue;
        }
        size_t maxCol = std::max({totalCol, passCol, failCol, xfailCol, xpassCol,
                                  errorCol, skipCol});
        if (fields.size() <= maxCol) {
          ++summary.summaryTSVParseErrors;
          ++summary.unparsedLines;
          continue;
        }
        uint64_t total = 0, pass = 0, fail = 0, xfail = 0;
        uint64_t xpass = 0, errorCount = 0, skip = 0;
        uint64_t schemaVersion = 1;
        if (!parseCountField(fields[totalCol], total) ||
            !parseCountField(fields[passCol], pass) ||
            !parseCountField(fields[failCol], fail) ||
            !parseCountField(fields[xfailCol], xfail) ||
            !parseCountField(fields[xpassCol], xpass) ||
            !parseCountField(fields[errorCol], errorCount) ||
            !parseCountField(fields[skipCol], skip)) {
          ++summary.summaryTSVParseErrors;
          ++summary.unparsedLines;
          continue;
        }
        if (schemaVersionCol != static_cast<size_t>(-1)) {
          if (fields.size() <= schemaVersionCol ||
              !parseCountField(fields[schemaVersionCol], schemaVersion)) {
            ++summary.summaryTSVParseErrors;
            ++summary.summaryTSVSchemaVersionInvalidRows;
            ++summary.unparsedLines;
            continue;
          }
        }
        ++summary.summaryTSVRows;
        ++summary.parsedSummaryLines;
        bool duplicateRow = false;
        if (suiteCol != static_cast<size_t>(-1) &&
            modeCol != static_cast<size_t>(-1) && fields.size() > suiteCol &&
            fields.size() > modeCol) {
          std::string rowKey =
              (fields[suiteCol].trim().str() + "\t" + fields[modeCol].trim().str());
          duplicateRow = !seenSummaryRows.insert(rowKey).second;
        }
        if (duplicateRow)
          ++summary.summaryTSVDuplicateRows;
        else
          ++summary.summaryTSVUniqueRows;
        ++summary.summaryTSVSchemaVersionRows;
        if (summary.summaryTSVSchemaVersionMin == 0 ||
            schemaVersion < summary.summaryTSVSchemaVersionMin)
          summary.summaryTSVSchemaVersionMin = schemaVersion;
        if (schemaVersion > summary.summaryTSVSchemaVersionMax)
          summary.summaryTSVSchemaVersionMax = schemaVersion;
        uint64_t statusSum =
            pass + fail + xfail + xpass + errorCount + skip;
        if (total == statusSum)
          ++summary.summaryTSVConsistentRows;
        else
          ++summary.summaryTSVInconsistentRows;
        summary.summaryTotal += total;
        summary.summaryPass += pass;
        summary.summaryFail += fail;
        summary.summaryError += errorCount;
        summary.summarySkip += skip;
        summary.summaryXFail += xfail;
        summary.summaryXPass += xpass;
        SmallVector<std::pair<std::string, uint64_t>, 16>
            rowSummaryCounterPairs;
        if (summaryCol != static_cast<size_t>(-1) &&
            fields.size() > summaryCol) {
          splitWhitespace(fields[summaryCol].trim(), tokens);
          for (StringRef token : tokens) {
            size_t eqPos = token.find('=');
            if (eqPos == StringRef::npos || eqPos == 0)
              continue;
            StringRef key = token.take_front(eqPos).trim();
            if (key.empty())
              continue;
            auto value = parseUnsignedTokenValue(token.drop_front(eqPos + 1));
            if (!value)
              continue;
            summary.summaryCounterSums[key] += *value;
            rowSummaryCounterPairs.emplace_back(key.str(), *value);
          }
        }
        if (suiteCol != static_cast<size_t>(-1) &&
            modeCol != static_cast<size_t>(-1) && fields.size() > suiteCol &&
            fields.size() > modeCol && !rowSummaryCounterPairs.empty()) {
          StringRef suiteName = fields[suiteCol].trim();
          StringRef modeName = fields[modeCol].trim();
          if (!suiteName.empty() && !modeName.empty()) {
            auto key = std::make_pair(suiteName.str(), modeName.str());
            auto &counterMap = summary.summaryCounterSumsBySuiteMode[key];
            for (const auto &entry : rowSummaryCounterPairs)
              counterMap[entry.first] += entry.second;
          }
        }
      }
      if (!headerSeen || !schemaValid || schemaInvalid)
        ++summary.summaryTSVSchemaInvalidFiles;
      else
        ++summary.summaryTSVSchemaValidFiles;
      continue;
    }
    for (StringRef rawLine : lines) {
      StringRef line = rawLine.rtrim("\r").trim();
      if (line.empty() || line.starts_with("#"))
        continue;
      ++summary.lines;
      splitWhitespace(line, tokens);
      if (tokens.empty())
        continue;

      bool hasSummaryCounts = false;
      uint64_t total = 0, pass = 0, fail = 0, errorCount = 0, skip = 0;
      uint64_t xfail = 0, xpass = 0;
      for (StringRef token : tokens) {
        uint64_t value = 0;
        if (parseSummaryTokenCount(token, "total", value)) {
          total += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "pass", value)) {
          pass += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "fail", value)) {
          fail += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "error", value)) {
          errorCount += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "skip", value)) {
          skip += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "xfail", value)) {
          xfail += value;
          hasSummaryCounts = true;
          continue;
        }
        if (parseSummaryTokenCount(token, "xpass", value)) {
          xpass += value;
          hasSummaryCounts = true;
          continue;
        }
      }
      if (hasSummaryCounts) {
        ++summary.parsedSummaryLines;
        summary.summaryTotal += total;
        summary.summaryPass += pass;
        summary.summaryFail += fail;
        summary.summaryError += errorCount;
        summary.summarySkip += skip;
        summary.summaryXFail += xfail;
        summary.summaryXPass += xpass;
        continue;
      }

      StringRef status;
      for (StringRef token : tokens) {
        if (token == "PASS" || token == "FAIL" || token == "ERROR" ||
            token == "SKIP" || token == "XFAIL" || token == "XPASS") {
          status = token;
          break;
        }
      }
      if (status.empty()) {
        ++summary.unparsedLines;
        continue;
      }

      ++summary.parsedStatusLines;
      if (status == "PASS")
        ++summary.pass;
      else if (status == "FAIL")
        ++summary.fail;
      else if (status == "ERROR")
        ++summary.error;
      else if (status == "SKIP")
        ++summary.skip;
      else if (status == "XFAIL")
        ++summary.xfail;
      else if (status == "XPASS")
        ++summary.xpass;
    }
  }

  rows.emplace_back("external_formal.files", std::to_string(summary.files));
  rows.emplace_back("external_formal.lines", std::to_string(summary.lines));
  rows.emplace_back("external_formal.parsed_status_lines",
                    std::to_string(summary.parsedStatusLines));
  rows.emplace_back("external_formal.parsed_summary_lines",
                    std::to_string(summary.parsedSummaryLines));
  rows.emplace_back("external_formal.unparsed_lines",
                    std::to_string(summary.unparsedLines));
  rows.emplace_back("external_formal.pass", std::to_string(summary.pass));
  rows.emplace_back("external_formal.fail", std::to_string(summary.fail));
  rows.emplace_back("external_formal.error", std::to_string(summary.error));
  rows.emplace_back("external_formal.skip", std::to_string(summary.skip));
  rows.emplace_back("external_formal.xfail", std::to_string(summary.xfail));
  rows.emplace_back("external_formal.xpass", std::to_string(summary.xpass));
  rows.emplace_back("external_formal.summary_total",
                    std::to_string(summary.summaryTotal));
  rows.emplace_back("external_formal.summary_pass",
                    std::to_string(summary.summaryPass));
  rows.emplace_back("external_formal.summary_fail",
                    std::to_string(summary.summaryFail));
  rows.emplace_back("external_formal.summary_error",
                    std::to_string(summary.summaryError));
  rows.emplace_back("external_formal.summary_skip",
                    std::to_string(summary.summarySkip));
  rows.emplace_back("external_formal.summary_xfail",
                    std::to_string(summary.summaryXFail));
  rows.emplace_back("external_formal.summary_xpass",
                    std::to_string(summary.summaryXPass));
  rows.emplace_back("external_formal.summary_tsv_files",
                    std::to_string(summary.summaryTSVFiles));
  rows.emplace_back("external_formal.summary_tsv_rows",
                    std::to_string(summary.summaryTSVRows));
  rows.emplace_back("external_formal.summary_tsv_schema_valid_files",
                    std::to_string(summary.summaryTSVSchemaValidFiles));
  rows.emplace_back("external_formal.summary_tsv_schema_invalid_files",
                    std::to_string(summary.summaryTSVSchemaInvalidFiles));
  rows.emplace_back("external_formal.summary_tsv_parse_errors",
                    std::to_string(summary.summaryTSVParseErrors));
  rows.emplace_back("external_formal.summary_tsv_consistent_rows",
                    std::to_string(summary.summaryTSVConsistentRows));
  rows.emplace_back("external_formal.summary_tsv_inconsistent_rows",
                    std::to_string(summary.summaryTSVInconsistentRows));
  rows.emplace_back("external_formal.summary_tsv_schema_version_rows",
                    std::to_string(summary.summaryTSVSchemaVersionRows));
  rows.emplace_back("external_formal.summary_tsv_schema_version_invalid_rows",
                    std::to_string(summary.summaryTSVSchemaVersionInvalidRows));
  rows.emplace_back("external_formal.summary_tsv_schema_version_min",
                    std::to_string(summary.summaryTSVSchemaVersionMin));
  rows.emplace_back("external_formal.summary_tsv_schema_version_max",
                    std::to_string(summary.summaryTSVSchemaVersionMax));
  rows.emplace_back("external_formal.summary_tsv_duplicate_rows",
                    std::to_string(summary.summaryTSVDuplicateRows));
  rows.emplace_back("external_formal.summary_tsv_unique_rows",
                    std::to_string(summary.summaryTSVUniqueRows));

  uint64_t failLikeSum = summary.fail + summary.error + summary.xpass +
                         summary.summaryFail + summary.summaryError +
                         summary.summaryXPass;
  rows.emplace_back("external_formal.fail_like_sum", std::to_string(failLikeSum));
  SmallVector<std::pair<std::string, uint64_t>, 32> summaryCounterRows;
  summaryCounterRows.reserve(summary.summaryCounterSums.size());
  for (const auto &entry : summary.summaryCounterSums)
    summaryCounterRows.emplace_back(entry.getKey().str(), entry.getValue());
  llvm::sort(summaryCounterRows, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (const auto &entry : summaryCounterRows)
    rows.emplace_back("external_formal.summary_counter." + entry.first,
                      std::to_string(entry.second));
  for (const auto &suiteModeEntry : summary.summaryCounterSumsBySuiteMode) {
    SmallVector<std::pair<std::string, uint64_t>, 16> suiteCounterRows;
    suiteCounterRows.reserve(suiteModeEntry.second.size());
    for (const auto &counterEntry : suiteModeEntry.second)
      suiteCounterRows.emplace_back(counterEntry.getKey().str(),
                                    counterEntry.getValue());
    llvm::sort(suiteCounterRows, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    std::string safeSuite = sanitizeReportKeySegment(suiteModeEntry.first.first);
    std::string safeMode = sanitizeReportKeySegment(suiteModeEntry.first.second);
    std::string rowPrefix = ("external_formal.summary_counter_by_suite_mode." +
                             safeSuite + "." + safeMode + ".");
    for (const auto &counterEntry : suiteCounterRows)
      rows.emplace_back(rowPrefix + counterEntry.first,
                        std::to_string(counterEntry.second));
  }
  return true;
}

static bool discoverExternalFormalResultsFromOutDir(
    StringRef outDir, SmallVectorImpl<std::string> &files, std::string &error) {
  if (outDir.empty())
    return true;
  bool isDir = false;
  if (std::error_code ec = sys::fs::is_directory(outDir, isDir); ec || !isDir) {
    error = (Twine("circt-mut report: external formal out-dir is not a directory: ") +
             outDir)
                .str();
    return false;
  }

  SmallString<256> summaryPath(outDir);
  sys::path::append(summaryPath, "summary.tsv");
  if (sys::fs::exists(summaryPath)) {
    files.push_back(std::string(summaryPath.str()));
    return true;
  }

  std::error_code ec;
  for (sys::fs::directory_iterator it(outDir, ec), end; it != end && !ec;
       it.increment(ec)) {
    if (sys::fs::is_directory(it->path()))
      continue;
    StringRef name = sys::path::filename(it->path());
    if (!name.contains("results"))
      continue;
    if (!name.ends_with(".txt") && !name.ends_with(".tsv"))
      continue;
    files.push_back(it->path());
  }
  if (ec) {
    error = (Twine("circt-mut report: failed to scan external formal out-dir: ") +
             outDir + ": " + ec.message())
                .str();
    return false;
  }
  llvm::sort(files);
  return true;
}

static std::string sanitizeReportKeySegment(StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  bool prevUnderscore = false;
  for (char c : raw) {
    bool isSafe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_';
    if (isSafe) {
      out.push_back(c);
      prevUnderscore = false;
    } else if (!prevUnderscore) {
      out.push_back('_');
      prevUnderscore = true;
    }
  }
  while (!out.empty() && out.back() == '_')
    out.pop_back();
  if (out.empty())
    out = "lane";
  return out;
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

static void appendMatrixPrequalifyProvenanceDeficitZeroRules(
    ReportOptions &opts) {
  appendUniqueRule(
      opts.failIfValueGtRules,
      "matrix.prequalify_results_summary_present_missing_pair_file_lanes", 0.0);
  appendUniqueRule(
      opts.failIfValueGtRules,
      "matrix.prequalify_results_summary_present_missing_log_file_lanes", 0.0);
}

static void appendMatrixPrequalifyProvenanceColumnPresenceRules(
    ReportOptions &opts) {
  appendUniqueRule(opts.failIfValueLtRules,
                   "matrix.prequalify_results_pair_file_column_present", 1.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "matrix.prequalify_results_log_file_column_present", 1.0);
}

static void appendMatrixNativeLifecycleStrictRules(ReportOptions &opts) {
  appendUniqueRule(opts.failIfValueLtRules,
                   "matrix.results_metrics_file_column_present", 1.0);
  appendUniqueRule(opts.failIfValueGtRules,
                   "matrix.results_metrics_file_fallback_lanes", 0.0);
  appendUniqueRule(opts.failIfValueLtRules, "matrix.runtime_summary_present",
                   1.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "matrix.native_prequalify_summary_file_exists", 1.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "matrix.prequalify_results_columns_present", 1.0);
}

static void appendTrendHistoryQualityRules(ReportOptions &opts,
                                           double minSelectedRuns) {
  appendUniqueRule(opts.failIfValueLtRules, "trend.history_runs_selected",
                   minSelectedRuns);
  appendUniqueRule(opts.failIfValueLtRules, "trend.numeric_keys", 1.0);
}

static void appendTrendHistoryQualityStrictRules(ReportOptions &opts) {
  appendTrendHistoryQualityRules(opts, 3.0);
  appendUniqueRule(opts.failIfValueGtRules,
                   "trend.matrix_core_numeric_keys_missing_history", 0.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "trend.matrix_core_numeric_keys_full_history", 5.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "trend.numeric_keys_full_history_pct", 80.0);
  appendUniqueRule(opts.failIfValueLtRules,
                   "trend.matrix_core_numeric_keys_full_history_pct", 100.0);
}

static bool isMatrixPolicyMode(StringRef mode) {
  return mode == "smoke" || mode == "nightly" || mode == "strict" ||
         mode == "trend-nightly" || mode == "trend-strict" ||
         mode == "native-trend-nightly" || mode == "native-trend-strict" ||
         mode == "provenance-guard" || mode == "provenance-strict" ||
         mode == "native-lifecycle-strict" || mode == "native-smoke" ||
         mode == "native-nightly" || mode == "native-strict" ||
         mode == "native-strict-formal" || mode == "strict-formal" ||
         mode == "native-strict-formal-summary" ||
         mode == "strict-formal-summary" ||
         mode == "native-strict-formal-summary-v1" ||
         mode == "strict-formal-summary-v1";
}

static bool matrixPolicyModeUsesStopOnFail(StringRef mode) {
  return mode == "smoke" || mode == "nightly" || mode == "strict" ||
         mode == "trend-nightly" || mode == "trend-strict" ||
         mode == "native-trend-nightly" || mode == "native-trend-strict" ||
         mode == "native-strict-formal" || mode == "strict-formal" ||
         mode == "native-strict-formal-summary" ||
         mode == "strict-formal-summary" ||
         mode == "native-strict-formal-summary-v1" ||
         mode == "strict-formal-summary-v1";
}

static bool appendMatrixPolicyModeProfiles(StringRef mode, bool stopOnFail,
                                           SmallVectorImpl<std::string> &out,
                                           std::string &error,
                                           StringRef errorPrefix) {
  std::string policyProfile;
  std::string provenanceProfile;
  std::string externalFormalProfile;
  std::string externalFormalSummaryProfile;
  std::string externalFormalSummaryV1Profile;
  std::string modeContractProfile;
  if (mode == "smoke") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-smoke"
                        : "formal-regression-matrix-composite-smoke";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
  } else if (mode == "nightly") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-nightly"
                        : "formal-regression-matrix-composite-nightly";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
  } else if (mode == "strict") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-strict"
                        : "formal-regression-matrix-composite-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
  } else if (mode == "trend-nightly") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-trend-nightly"
                        : "formal-regression-matrix-composite-trend-nightly";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
  } else if (mode == "trend-strict") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-trend-strict"
                        : "formal-regression-matrix-composite-trend-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
  } else if (mode == "native-trend-nightly") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-trend-nightly"
                        : "formal-regression-matrix-composite-trend-nightly";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-family-contract";
  } else if (mode == "native-trend-strict") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-trend-strict"
                        : "formal-regression-matrix-composite-trend-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-family-contract";
  } else if (mode == "provenance-guard") {
    policyProfile = "formal-regression-matrix-provenance-guard";
  } else if (mode == "provenance-strict") {
    policyProfile = "formal-regression-matrix-provenance-strict";
  } else if (mode == "native-lifecycle-strict") {
    policyProfile = "formal-regression-matrix-native-lifecycle-strict";
  } else if (mode == "native-smoke") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-smoke"
                        : "formal-regression-matrix-composite-smoke";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-family-contract";
  } else if (mode == "native-nightly") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-nightly"
                        : "formal-regression-matrix-composite-nightly";
    provenanceProfile = "formal-regression-matrix-provenance-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-family-contract";
  } else if (mode == "native-strict") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-native-strict"
                        : "formal-regression-matrix-composite-native-strict";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-strict-contract";
  } else if (mode == "native-strict-formal") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-native-strict"
                        : "formal-regression-matrix-composite-native-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-strict-contract";
  } else if (mode == "strict-formal") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-strict"
                        : "formal-regression-matrix-composite-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
  } else if (mode == "native-strict-formal-summary") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-native-strict"
                        : "formal-regression-matrix-composite-native-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
    externalFormalSummaryProfile =
        "formal-regression-matrix-external-formal-summary-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-strict-contract";
  } else if (mode == "strict-formal-summary") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-strict"
                        : "formal-regression-matrix-composite-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
    externalFormalSummaryProfile =
        "formal-regression-matrix-external-formal-summary-guard";
  } else if (mode == "native-strict-formal-summary-v1") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-native-strict"
                        : "formal-regression-matrix-composite-native-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
    externalFormalSummaryProfile =
        "formal-regression-matrix-external-formal-summary-guard";
    externalFormalSummaryV1Profile =
        "formal-regression-matrix-external-formal-summary-v1-guard";
    modeContractProfile =
        "formal-regression-matrix-policy-mode-native-strict-contract";
  } else if (mode == "strict-formal-summary-v1") {
    policyProfile = stopOnFail
                        ? "formal-regression-matrix-composite-stop-on-fail-strict"
                        : "formal-regression-matrix-composite-strict";
    provenanceProfile = "formal-regression-matrix-provenance-strict";
    externalFormalProfile = "formal-regression-matrix-external-formal-guard";
    externalFormalSummaryProfile =
        "formal-regression-matrix-external-formal-summary-guard";
    externalFormalSummaryV1Profile =
        "formal-regression-matrix-external-formal-summary-v1-guard";
  } else {
    error = (Twine(errorPrefix) + " invalid report policy mode value '" + mode +
             (Twine("' (expected ") + kMatrixPolicyModeList + ")"))
                .str();
    return false;
  }
  out.push_back(policyProfile);
  if (!provenanceProfile.empty())
    out.push_back(provenanceProfile);
  if (!externalFormalProfile.empty())
    out.push_back(externalFormalProfile);
  if (!externalFormalSummaryProfile.empty())
    out.push_back(externalFormalSummaryProfile);
  if (!externalFormalSummaryV1Profile.empty())
    out.push_back(externalFormalSummaryV1Profile);
  if (!modeContractProfile.empty())
    out.push_back(modeContractProfile);
  return true;
}

static bool applyPolicyProfile(StringRef profile, ReportOptions &opts,
                               std::string &error) {
  auto applyComposite = [&](StringRef nested) -> bool {
    return applyPolicyProfile(nested, opts, error);
  };

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
    appendTrendHistoryQualityRules(opts, 2.0);
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
  if (profile == "formal-regression-matrix-basic") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-trend") {
    opts.failOnPrequalifyDrift = true;
    appendTrendHistoryQualityRules(opts, 2.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-guard") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-trend-guard") {
    opts.failOnPrequalifyDrift = true;
    appendTrendHistoryQualityRules(opts, 2.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-guard-smoke") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 5.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 5.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 5.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-guard-nightly") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-stop-on-fail-guard-smoke") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 5.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 5.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 5.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.skip_budget_rows_non_stop_on_fail", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-stop-on-fail-guard-nightly") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.skip_budget_rows_non_stop_on_fail", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-guard-strict") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-nightly") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-strict") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "matrix.prequalify_drift_lane_rows_compared", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-stop-on-fail-basic") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfDeltaGtRules,
                     "matrix.skip_budget_rows_non_stop_on_fail", 0.0);
    appendUniqueRule(opts.failIfDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-stop-on-fail-trend") {
    opts.failOnPrequalifyDrift = true;
    appendTrendHistoryQualityRules(opts, 2.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.skip_budget_rows_non_stop_on_fail", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-stop-on-fail-strict") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "matrix.prequalify_drift_lane_rows_compared", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.skip_budget_rows_non_stop_on_fail", 0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-full-lanes-strict") {
    opts.failOnPrequalifyDrift = true;
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_timeout_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_lec_unknown_mutants_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.global_filter_bmc_unknown_mutants_sum", 0.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.errors_sum", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.detected_mutants_sum",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "matrix.prequalify_drift_lane_rows_compared", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-lane-drift-nightly") {
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-lane-drift-strict") {
    appendUniqueRule(opts.failIfValueLtRules, "matrix.prequalify_drift_comparable",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "matrix.prequalify_drift_lane_rows_compared", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_nonzero_metrics", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_mismatch", 0.0);
    appendUniqueRule(
        opts.failIfValueGtRules,
        "matrix.prequalify_drift_lane_rows_missing_in_results", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.prequalify_drift_lane_rows_missing_in_native",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-lane-trend-nightly") {
    opts.failOnPrequalifyDrift = true;
    appendTrendHistoryQualityRules(opts, 2.0);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_timeout_mutants_value", 0.0);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_lec_unknown_mutants_value", 0.0);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_bmc_unknown_mutants_value", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.lane_budget.worst_errors_value", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules,
                     "matrix.lane_budget.lowest_detected_mutants_value", 0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-lane-trend-strict") {
    opts.failOnPrequalifyDrift = true;
    appendTrendHistoryQualityStrictRules(opts);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_timeout_mutants_value", 0.0);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_lec_unknown_mutants_value", 0.0);
    appendUniqueRule(
        opts.failIfTrendDeltaGtRules,
        "matrix.lane_budget.worst_global_filter_bmc_unknown_mutants_value", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.lane_budget.worst_errors_value", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules,
                     "matrix.lane_budget.lanes_zero_detected_mutants", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.lanes_skip", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules,
                     "matrix.lane_budget.lowest_detected_mutants_value", 0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules, "matrix.detected_mutants_sum",
                     0.0);
    appendUniqueRule(opts.failIfTrendDeltaLtRules,
                     "matrix.prequalify_drift_comparable", 0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-external-formal-guard") {
    appendUniqueRule(opts.failIfValueLtRules, "external_formal.files", 1.0);
    appendUniqueRule(opts.failIfValueGtRules, "external_formal.fail_like_sum",
                     0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-external-formal-summary-guard") {
    appendUniqueRule(opts.failIfValueLtRules, "external_formal.files", 1.0);
    appendUniqueRule(opts.failIfValueLtRules, "external_formal.summary_tsv_files",
                     1.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "external_formal.summary_tsv_schema_valid_files", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_schema_invalid_files", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_parse_errors", 0.0);
    appendUniqueRule(opts.failIfValueLtRules, "external_formal.summary_tsv_rows",
                     1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_inconsistent_rows", 0.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_schema_version_invalid_rows",
                     0.0);
    appendUniqueRule(opts.failIfValueLtRules,
                     "external_formal.summary_tsv_schema_version_min", 1.0);
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_duplicate_rows", 0.0);
    return true;
  }
  if (profile == "formal-regression-matrix-external-formal-summary-v1-guard") {
    appendUniqueRule(opts.failIfValueGtRules,
                     "external_formal.summary_tsv_schema_version_max", 1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-provenance-guard") {
    appendMatrixPrequalifyProvenanceColumnPresenceRules(opts);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    return true;
  }
  if (profile == "formal-regression-matrix-provenance-strict") {
    appendMatrixPrequalifyProvenanceColumnPresenceRules(opts);
    appendUniqueRule(opts.failIfValueLtRules,
                     "matrix.prequalify_results_summary_present_lanes", 1.0);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    return true;
  }
  if (profile == "formal-regression-matrix-native-lifecycle-strict") {
    appendMatrixNativeLifecycleStrictRules(opts);
    appendMatrixPrequalifyProvenanceColumnPresenceRules(opts);
    appendMatrixPrequalifyProvenanceDeficitZeroRules(opts);
    return true;
  }
  if (profile == "formal-regression-matrix-policy-mode-native-strict-contract") {
    appendUniqueRule(opts.failIfValueLtRules, "policy.mode_is_set", 1.0);
    appendUniqueRule(opts.failIfValueLtRules, "policy.mode_is_native_strict",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-policy-mode-native-family-contract") {
    appendUniqueRule(opts.failIfValueLtRules, "policy.mode_is_set", 1.0);
    appendUniqueRule(opts.failIfValueLtRules, "policy.mode_is_native_family",
                     1.0);
    return true;
  }
  if (profile == "formal-regression-matrix-runtime-smoke") {
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.runtime_summary_invalid_rows", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_avg",
                     300000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_max",
                     1200000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_sum",
                     3600000000000.0);
    return true;
  }
  if (profile == "formal-regression-matrix-runtime-nightly") {
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.runtime_summary_invalid_rows", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_avg",
                     120000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_max",
                     600000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_sum",
                     1800000000000.0);
    return true;
  }
  if (profile == "formal-regression-matrix-runtime-trend") {
    appendTrendHistoryQualityRules(opts, 2.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.runtime_ns_avg",
                     60000000000.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.runtime_ns_max",
                     180000000000.0);
    appendUniqueRule(opts.failIfTrendDeltaGtRules, "matrix.runtime_ns_sum",
                     600000000000.0);
    return true;
  }
  if (profile == "formal-regression-matrix-trend-history-quality") {
    appendTrendHistoryQualityRules(opts, 2.0);
    return true;
  }
  if (profile == "formal-regression-matrix-trend-history-quality-strict") {
    appendTrendHistoryQualityStrictRules(opts);
    return true;
  }
  if (profile == "formal-regression-matrix-runtime-strict") {
    appendUniqueRule(opts.failIfValueGtRules,
                     "matrix.runtime_summary_invalid_rows", 0.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_avg",
                     60000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_max",
                     240000000000.0);
    appendUniqueRule(opts.failIfValueGtRules, "matrix.runtime_ns_sum",
                     720000000000.0);
    return true;
  }
  if (profile == "formal-regression-matrix-composite-smoke") {
    return applyComposite("formal-regression-matrix-guard-smoke") &&
           applyComposite("formal-regression-matrix-runtime-smoke");
  }
  if (profile == "formal-regression-matrix-composite-nightly") {
    return applyComposite("formal-regression-matrix-guard-nightly") &&
           applyComposite("formal-regression-matrix-runtime-nightly");
  }
  if (profile == "formal-regression-matrix-composite-strict") {
    return applyComposite("formal-regression-matrix-full-lanes-strict") &&
           applyComposite("formal-regression-matrix-runtime-strict");
  }
  if (profile == "formal-regression-matrix-composite-native-strict") {
    return applyComposite("formal-regression-matrix-composite-strict") &&
           applyComposite("formal-regression-matrix-provenance-strict") &&
           applyComposite("formal-regression-matrix-native-lifecycle-strict");
  }
  if (profile == "formal-regression-matrix-composite-trend-nightly") {
    return applyComposite("formal-regression-matrix-lane-trend-nightly") &&
           applyComposite("formal-regression-matrix-runtime-trend") &&
           applyComposite("formal-regression-matrix-lane-drift-nightly") &&
           applyComposite("formal-regression-matrix-trend-history-quality");
  }
  if (profile == "formal-regression-matrix-composite-trend-strict") {
    return applyComposite("formal-regression-matrix-lane-trend-strict") &&
           applyComposite("formal-regression-matrix-runtime-trend") &&
           applyComposite("formal-regression-matrix-lane-drift-strict") &&
           applyComposite("formal-regression-matrix-trend-history-quality-strict");
  }
  if (profile == "formal-regression-matrix-composite-stop-on-fail-smoke") {
    return applyComposite("formal-regression-matrix-stop-on-fail-guard-smoke") &&
           applyComposite("formal-regression-matrix-runtime-smoke");
  }
  if (profile == "formal-regression-matrix-composite-stop-on-fail-nightly") {
    return applyComposite("formal-regression-matrix-stop-on-fail-guard-nightly") &&
           applyComposite("formal-regression-matrix-runtime-nightly");
  }
  if (profile == "formal-regression-matrix-composite-stop-on-fail-strict") {
    return applyComposite("formal-regression-matrix-stop-on-fail-strict") &&
           applyComposite("formal-regression-matrix-runtime-strict");
  }
  if (profile == "formal-regression-matrix-composite-stop-on-fail-native-strict") {
    return applyComposite("formal-regression-matrix-composite-stop-on-fail-strict") &&
           applyComposite("formal-regression-matrix-provenance-strict") &&
           applyComposite("formal-regression-matrix-native-lifecycle-strict");
  }
  if (profile ==
      "formal-regression-matrix-composite-stop-on-fail-trend-nightly") {
    return applyComposite("formal-regression-matrix-stop-on-fail-trend") &&
           applyComposite("formal-regression-matrix-lane-trend-nightly") &&
           applyComposite("formal-regression-matrix-runtime-trend") &&
           applyComposite("formal-regression-matrix-lane-drift-nightly") &&
           applyComposite("formal-regression-matrix-trend-history-quality");
  }
  if (profile ==
      "formal-regression-matrix-composite-stop-on-fail-trend-strict") {
    return applyComposite("formal-regression-matrix-stop-on-fail-trend") &&
           applyComposite("formal-regression-matrix-lane-trend-strict") &&
           applyComposite("formal-regression-matrix-runtime-trend") &&
           applyComposite("formal-regression-matrix-lane-drift-strict") &&
           applyComposite("formal-regression-matrix-trend-history-quality-strict");
  }
  error = (Twine("circt-mut report: unknown --policy-profile value: ") + profile +
           " (expected formal-regression-basic|formal-regression-trend|"
           "formal-regression-matrix-basic|formal-regression-matrix-trend|"
           "formal-regression-matrix-guard|"
           "formal-regression-matrix-trend-guard|"
           "formal-regression-matrix-guard-smoke|"
           "formal-regression-matrix-guard-nightly|"
           "formal-regression-matrix-stop-on-fail-guard-smoke|"
           "formal-regression-matrix-stop-on-fail-guard-nightly|"
           "formal-regression-matrix-guard-strict|"
           "formal-regression-matrix-nightly|"
           "formal-regression-matrix-strict|"
           "formal-regression-matrix-stop-on-fail-basic|"
           "formal-regression-matrix-stop-on-fail-trend|"
           "formal-regression-matrix-stop-on-fail-strict|"
           "formal-regression-matrix-full-lanes-strict|"
           "formal-regression-matrix-lane-drift-nightly|"
           "formal-regression-matrix-lane-drift-strict|"
           "formal-regression-matrix-lane-trend-nightly|"
           "formal-regression-matrix-lane-trend-strict|"
           "formal-regression-matrix-external-formal-guard|"
           "formal-regression-matrix-external-formal-summary-guard|"
           "formal-regression-matrix-external-formal-summary-v1-guard|"
           "formal-regression-matrix-provenance-guard|"
           "formal-regression-matrix-provenance-strict|"
           "formal-regression-matrix-native-lifecycle-strict|"
           "formal-regression-matrix-policy-mode-native-strict-contract|"
           "formal-regression-matrix-policy-mode-native-family-contract|"
           "formal-regression-matrix-runtime-smoke|"
           "formal-regression-matrix-runtime-nightly|"
           "formal-regression-matrix-runtime-trend|"
           "formal-regression-matrix-trend-history-quality|"
           "formal-regression-matrix-trend-history-quality-strict|"
           "formal-regression-matrix-runtime-strict|"
           "formal-regression-matrix-composite-smoke|"
           "formal-regression-matrix-composite-nightly|"
           "formal-regression-matrix-composite-strict|"
           "formal-regression-matrix-composite-native-strict|"
           "formal-regression-matrix-composite-trend-nightly|"
           "formal-regression-matrix-composite-trend-strict|"
           "formal-regression-matrix-composite-stop-on-fail-smoke|"
           "formal-regression-matrix-composite-stop-on-fail-nightly|"
           "formal-regression-matrix-composite-stop-on-fail-strict|"
           "formal-regression-matrix-composite-stop-on-fail-native-strict|"
           "formal-regression-matrix-composite-stop-on-fail-trend-nightly|"
           "formal-regression-matrix-composite-stop-on-fail-trend-strict)")
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

static bool pruneHistoryToMaxRuns(StringRef path, uint64_t maxRuns,
                                  uint64_t &prunedRuns, uint64_t &prunedRows,
                                  std::string &error) {
  prunedRuns = 0;
  prunedRows = 0;
  if (maxRuns == 0 || !sys::fs::exists(path))
    return true;

  auto bufferOrErr = MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    error = (Twine("circt-mut report: unable to read history file: ") + path).str();
    return false;
  }

  SmallVector<StringRef, 256> lines;
  bufferOrErr.get()->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);
  struct Row {
    uint64_t runID = 0;
    std::string timestamp;
    std::string key;
    std::string value;
  };
  std::vector<Row> rows;
  std::set<uint64_t> runIDs;
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
    Row row;
    row.runID = runID;
    row.timestamp = fields[1].trim().str();
    row.key = fields[2].trim().str();
    row.value = fields[3].trim().str();
    rows.push_back(std::move(row));
    runIDs.insert(runID);
  }

  if (runIDs.size() <= maxRuns)
    return true;

  std::vector<uint64_t> sortedRunIDs(runIDs.begin(), runIDs.end());
  uint64_t keepFromRunID = sortedRunIDs[sortedRunIDs.size() - maxRuns];
  prunedRuns = 0;
  for (uint64_t runID : sortedRunIDs)
    if (runID < keepFromRunID)
      ++prunedRuns;

  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_Text);
  if (ec) {
    error = (Twine("circt-mut report: failed to rewrite history file: ") + path +
             ": " + ec.message())
                .str();
    return false;
  }
  os << "run_id\ttimestamp_utc\tkey\tvalue\n";
  for (const auto &row : rows) {
    if (row.runID < keepFromRunID) {
      ++prunedRows;
      continue;
    }
    os << row.runID << "\t" << row.timestamp << "\t" << row.key << "\t"
       << row.value << "\n";
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
  uint64_t numericKeysFullHistory = 0;
  StringMap<uint64_t> keySampleCounts;
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
    keySampleCounts[currentRow.first] = samples;
    if (samples == selected.size())
      ++numericKeysFullHistory;
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
  rows.emplace_back("trend.numeric_keys_full_history",
                    std::to_string(numericKeysFullHistory));
  rows.emplace_back("trend.numeric_keys_partial_history",
                    std::to_string(numericKeys - numericKeysFullHistory));
  rows.emplace_back(
      "trend.numeric_keys_full_history_pct",
      numericKeys != 0
          ? formatDouble2((100.0 * static_cast<double>(numericKeysFullHistory)) /
                          static_cast<double>(numericKeys))
          : std::string("0.00"));

  static constexpr StringLiteral matrixTrendCoreKeys[] = {
      "matrix.detected_mutants_sum", "matrix.lanes_skip", "matrix.runtime_ns_avg",
      "matrix.runtime_ns_max", "matrix.runtime_ns_sum"};
  uint64_t matrixCoreCurrent = 0;
  uint64_t matrixCoreFullHistory = 0;
  std::string matrixCoreMissingHistoryList;
  for (StringRef coreKey : matrixTrendCoreKeys) {
    auto itCurrent = llvm::find_if(numericCurrentRows, [&](const auto &row) {
      return StringRef(row.first) == coreKey;
    });
    if (itCurrent == numericCurrentRows.end())
      continue;
    ++matrixCoreCurrent;
    auto itSamples = keySampleCounts.find(coreKey);
    if (itSamples != keySampleCounts.end() && itSamples->second == selected.size()) {
      ++matrixCoreFullHistory;
    } else {
      if (!matrixCoreMissingHistoryList.empty())
        matrixCoreMissingHistoryList.append(",");
      matrixCoreMissingHistoryList.append(coreKey.str());
    }
  }
  uint64_t matrixCoreMissingHistory =
      matrixCoreCurrent - matrixCoreFullHistory;
  rows.emplace_back("trend.matrix_core_numeric_keys_current",
                    std::to_string(matrixCoreCurrent));
  rows.emplace_back("trend.matrix_core_numeric_keys_full_history",
                    std::to_string(matrixCoreFullHistory));
  rows.emplace_back("trend.matrix_core_numeric_keys_missing_history",
                    std::to_string(matrixCoreMissingHistory));
  rows.emplace_back("trend.matrix_core_numeric_keys_missing_history_list",
                    matrixCoreMissingHistoryList.empty()
                        ? std::string("-")
                        : matrixCoreMissingHistoryList);
  rows.emplace_back(
      "trend.matrix_core_numeric_keys_full_history_pct",
      matrixCoreCurrent != 0
          ? formatDouble2(
                (100.0 * static_cast<double>(matrixCoreFullHistory)) /
                static_cast<double>(matrixCoreCurrent))
          : std::string("0.00"));
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
    std::string &error, uint64_t *prequalifyDriftNonZeroOut = nullptr,
    bool *prequalifyDriftComparableOut = nullptr,
    uint64_t *prequalifyDriftLaneRowsMismatchOut = nullptr,
    uint64_t *prequalifyDriftLaneRowsMissingInResultsOut = nullptr,
    uint64_t *prequalifyDriftLaneRowsMissingInNativeOut = nullptr,
    std::vector<MatrixLaneBudgetRow> *laneBudgetRowsOut = nullptr) {
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
  size_t runtimeCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("runtime_ns"); it != colIndex.end())
    runtimeCol = it->second;
  size_t metricsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("metrics_file"); it != colIndex.end())
    metricsCol = it->second;
  bool hasMetricsFileColumn = metricsCol != static_cast<size_t>(-1);
  size_t prequalifySummaryPresentCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_summary_present");
      it != colIndex.end())
    prequalifySummaryPresentCol = it->second;
  size_t prequalifyPairFileCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_pair_file"); it != colIndex.end())
    prequalifyPairFileCol = it->second;
  size_t prequalifyLogFileCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_log_file"); it != colIndex.end())
    prequalifyLogFileCol = it->second;
  bool hasPrequalifyPairFileColumn =
      prequalifyPairFileCol != static_cast<size_t>(-1);
  bool hasPrequalifyLogFileColumn =
      prequalifyLogFileCol != static_cast<size_t>(-1);
  size_t prequalifyTotalMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_total_mutants"); it != colIndex.end())
    prequalifyTotalMutantsCol = it->second;
  size_t prequalifyNotPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_not_propagated_mutants");
      it != colIndex.end())
    prequalifyNotPropagatedMutantsCol = it->second;
  size_t prequalifyPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_propagated_mutants");
      it != colIndex.end())
    prequalifyPropagatedMutantsCol = it->second;
  size_t prequalifyCreateMutatedErrorMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_create_mutated_error_mutants");
      it != colIndex.end())
    prequalifyCreateMutatedErrorMutantsCol = it->second;
  size_t prequalifyProbeErrorMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_probe_error_mutants");
      it != colIndex.end())
    prequalifyProbeErrorMutantsCol = it->second;
  size_t prequalifyCmdTokenNotPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_token_not_propagated_mutants");
      it != colIndex.end())
    prequalifyCmdTokenNotPropagatedMutantsCol = it->second;
  size_t prequalifyCmdTokenPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_token_propagated_mutants");
      it != colIndex.end())
    prequalifyCmdTokenPropagatedMutantsCol = it->second;
  size_t prequalifyCmdRCNotPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_rc_not_propagated_mutants");
      it != colIndex.end())
    prequalifyCmdRCNotPropagatedMutantsCol = it->second;
  size_t prequalifyCmdRCPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_rc_propagated_mutants");
      it != colIndex.end())
    prequalifyCmdRCPropagatedMutantsCol = it->second;
  size_t prequalifyCmdTimeoutPropagatedMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_timeout_propagated_mutants");
      it != colIndex.end())
    prequalifyCmdTimeoutPropagatedMutantsCol = it->second;
  size_t prequalifyCmdErrorMutantsCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("prequalify_cmd_error_mutants");
      it != colIndex.end())
    prequalifyCmdErrorMutantsCol = it->second;
  size_t configErrorCodeCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("config_error_code"); it != colIndex.end())
    configErrorCodeCol = it->second;
  size_t configErrorReasonCol = static_cast<size_t>(-1);
  if (auto it = colIndex.find("config_error_reason"); it != colIndex.end())
    configErrorReasonCol = it->second;
  bool hasResultsPrequalifyColumns =
      prequalifySummaryPresentCol != static_cast<size_t>(-1) ||
      prequalifyTotalMutantsCol != static_cast<size_t>(-1);

  uint64_t lanesTotal = 0;
  uint64_t lanesPass = 0;
  uint64_t lanesFail = 0;
  uint64_t lanesSkip = 0;
  uint64_t gatePass = 0;
  uint64_t gateFail = 0;
  uint64_t gateSkip = 0;
  uint64_t lanesWithMetrics = 0;
  uint64_t lanesMissingMetrics = 0;
  uint64_t resultsMetricsFileFallbackLanes = 0;
  uint64_t invalidMetricValues = 0;
  uint64_t totalMutantsSum = 0;
  uint64_t relevantMutantsSum = 0;
  uint64_t detectedMutantsSum = 0;
  uint64_t propagatedNotDetectedMutantsSum = 0;
  uint64_t notPropagatedMutantsSum = 0;
  uint64_t notActivatedMutantsSum = 0;
  uint64_t errorsSum = 0;
  uint64_t prequalifyResultsLanes = 0;
  uint64_t prequalifyResultsSummaryPresentLanes = 0;
  uint64_t prequalifyResultsSummaryMissingLanes = 0;
  uint64_t prequalifyResultsPairFilePresentLanes = 0;
  uint64_t prequalifyResultsLogFilePresentLanes = 0;
  uint64_t prequalifyResultsSummaryPresentMissingPairFileLanes = 0;
  uint64_t prequalifyResultsSummaryPresentMissingLogFileLanes = 0;
  uint64_t prequalifyResultsInvalidMetricValues = 0;
  uint64_t prequalifyResultsTotalMutantsSum = 0;
  uint64_t prequalifyResultsNotPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCreateMutatedErrorMutantsSum = 0;
  uint64_t prequalifyResultsProbeErrorMutantsSum = 0;
  uint64_t prequalifyResultsCmdTokenNotPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCmdTokenPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCmdRCNotPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCmdRCPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCmdTimeoutPropagatedMutantsSum = 0;
  uint64_t prequalifyResultsCmdErrorMutantsSum = 0;
  uint64_t laneBudgetMaxTimeoutMutants = 0;
  uint64_t laneBudgetMaxLECUnknownMutants = 0;
  uint64_t laneBudgetMaxBMCUnknownMutants = 0;
  uint64_t laneBudgetMaxErrors = 0;
  uint64_t laneBudgetMinDetectedMutants = 0;
  bool laneBudgetSawDetected = false;
  uint64_t laneBudgetLanesZeroDetectedMutants = 0;
  uint64_t laneBudgetLanesNonZeroTimeoutMutants = 0;
  uint64_t laneBudgetLanesNonZeroLECUnknownMutants = 0;
  uint64_t laneBudgetLanesNonZeroBMCUnknownMutants = 0;
  std::string laneBudgetWorstTimeoutLaneID = "-";
  std::string laneBudgetWorstLECUnknownLaneID = "-";
  std::string laneBudgetWorstBMCUnknownLaneID = "-";
  std::string laneBudgetWorstErrorsLaneID = "-";
  std::string laneBudgetLowestDetectedLaneID = "-";
  bool runtimeSummaryPresent = false;
  uint64_t runtimeSummaryRows = 0;
  uint64_t runtimeSummaryInvalidRows = 0;
  uint64_t runtimeSummaryMatchedRows = 0;
  uint64_t runtimeSummarySum = 0;
  uint64_t runtimeSummaryMax = 0;
  std::string runtimeSummaryMaxLane = "-";
  StringMap<uint64_t> runtimeNanosByLane;
  StringMap<MatrixPrequalifyLaneMetrics> resultsPrequalifyByLane;
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

  bool runtimeFromResults = runtimeCol != static_cast<size_t>(-1);
  SmallString<256> runtimeSummaryPath(matrixOutDir);
  sys::path::append(runtimeSummaryPath, "native_matrix_dispatch_runtime.tsv");
  if (!runtimeFromResults && sys::fs::exists(runtimeSummaryPath)) {
    runtimeSummaryPresent = true;
    auto runtimeBufferOrErr = MemoryBuffer::getFile(runtimeSummaryPath);
    if (!runtimeBufferOrErr) {
      error = (Twine("circt-mut report: unable to read matrix runtime summary file: ") +
               runtimeSummaryPath)
                  .str();
      return false;
    }
    SmallVector<StringRef, 256> runtimeLines;
    runtimeBufferOrErr.get()->getBuffer().split(runtimeLines, '\n',
                                                /*MaxSplit=*/-1,
                                                /*KeepEmpty=*/false);
    if (!runtimeLines.empty()) {
      SmallVector<StringRef, 8> runtimeHeader;
      splitTSVLine(runtimeLines.front().rtrim("\r"), runtimeHeader);
      int laneIDCol = -1;
      int runtimeCol = -1;
      for (size_t i = 0; i < runtimeHeader.size(); ++i) {
        StringRef h = runtimeHeader[i].trim();
        if (h == "lane_id")
          laneIDCol = static_cast<int>(i);
        else if (h == "runtime_ns")
          runtimeCol = static_cast<int>(i);
      }
      if (laneIDCol == -1 || runtimeCol == -1) {
        error = (Twine("circt-mut report: invalid matrix runtime summary header in ") +
                 runtimeSummaryPath + " (expected lane_id and runtime_ns columns)")
                    .str();
        return false;
      }
      SmallVector<StringRef, 8> runtimeFields;
      for (size_t lineNo = 1; lineNo < runtimeLines.size(); ++lineNo) {
        StringRef runtimeLine = runtimeLines[lineNo].rtrim("\r");
        if (runtimeLine.trim().empty())
          continue;
        splitTSVLine(runtimeLine, runtimeFields);
        if (static_cast<size_t>(laneIDCol) >= runtimeFields.size() ||
            static_cast<size_t>(runtimeCol) >= runtimeFields.size()) {
          ++runtimeSummaryInvalidRows;
          continue;
        }
        StringRef laneID = runtimeFields[laneIDCol].trim();
        StringRef runtimeValue = runtimeFields[runtimeCol].trim();
        if (laneID.empty() || runtimeValue.empty()) {
          ++runtimeSummaryInvalidRows;
          continue;
        }
        uint64_t parsed = 0;
        if (runtimeValue.getAsInteger(10, parsed)) {
          ++runtimeSummaryInvalidRows;
          continue;
        }
        ++runtimeSummaryRows;
        runtimeNanosByLane[laneID] = parsed;
        runtimeSummarySum += parsed;
        if (parsed > runtimeSummaryMax ||
            (parsed == runtimeSummaryMax &&
             (runtimeSummaryMaxLane == "-" || laneID.str() < runtimeSummaryMaxLane))) {
          runtimeSummaryMax = parsed;
          runtimeSummaryMaxLane = laneID.str();
        }
      }
    }
  } else if (runtimeFromResults) {
    runtimeSummaryPresent = true;
  }

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
  auto tryGetMetric = [&](const StringMap<std::string> &metrics, StringRef key,
                          uint64_t &value) -> bool {
    auto it = metrics.find(key);
    if (it == metrics.end() || it->second.empty())
      return false;
    uint64_t parsed = 0;
    if (StringRef(it->second).trim().getAsInteger(10, parsed)) {
      ++invalidMetricValues;
      return false;
    }
    value = parsed;
    return true;
  };
  auto addOptionalResultMetric = [&](ArrayRef<StringRef> rowFields, size_t col,
                                     uint64_t &accumulator,
                                     bool allowDash = false) {
    if (col == static_cast<size_t>(-1) || col >= rowFields.size())
      return;
    StringRef value = rowFields[col].trim();
    if (value.empty())
      return;
    if (allowDash && value == "-")
      return;
    uint64_t parsed = 0;
    if (value.getAsInteger(10, parsed)) {
      ++prequalifyResultsInvalidMetricValues;
      return;
    }
    accumulator += parsed;
  };
  auto readOptionalResultMetric = [&](ArrayRef<StringRef> rowFields, size_t col,
                                      uint64_t &value,
                                      bool allowDash = false) {
    value = 0;
    if (col == static_cast<size_t>(-1) || col >= rowFields.size())
      return;
    StringRef fieldValue = rowFields[col].trim();
    if (fieldValue.empty())
      return;
    if (allowDash && fieldValue == "-")
      return;
    uint64_t parsed = 0;
    if (fieldValue.getAsInteger(10, parsed)) {
      ++prequalifyResultsInvalidMetricValues;
      return;
    }
    value = parsed;
  };
  auto updateWorstLane = [&](uint64_t value, StringRef laneID,
                             uint64_t &currentWorstValue,
                             std::string &currentWorstLaneID) {
    if (value > currentWorstValue ||
        (value == currentWorstValue &&
         (currentWorstLaneID == "-" || laneID.str() < currentWorstLaneID))) {
      currentWorstValue = value;
      currentWorstLaneID = laneID.str();
    }
  };
  auto updateLowestLane = [&](uint64_t value, StringRef laneID,
                              bool &seenAny, uint64_t &currentLowestValue,
                              std::string &currentLowestLaneID) {
    if (!seenAny || value < currentLowestValue ||
        (value == currentLowestValue &&
         (currentLowestLaneID == "-" || laneID.str() < currentLowestLaneID))) {
      seenAny = true;
      currentLowestValue = value;
      currentLowestLaneID = laneID.str();
    }
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
    MatrixLaneBudgetRow laneBudgetRow;
    laneBudgetRow.laneID = laneID.str();
    laneBudgetRow.status = status.str();
    laneBudgetRow.gateStatus = gate.str();
    if (runtimeFromResults) {
      StringRef runtimeValue = getField(runtimeCol);
      if (!runtimeValue.empty() && runtimeValue != "-") {
        uint64_t parsed = 0;
        if (runtimeValue.getAsInteger(10, parsed)) {
          ++runtimeSummaryInvalidRows;
        } else {
          laneBudgetRow.hasRuntimeNanos = true;
          laneBudgetRow.runtimeNanos = parsed;
          ++runtimeSummaryRows;
          ++runtimeSummaryMatchedRows;
          runtimeSummarySum += parsed;
          if (parsed > runtimeSummaryMax ||
              (parsed == runtimeSummaryMax &&
               (runtimeSummaryMaxLane == "-" ||
                laneID.str() < runtimeSummaryMaxLane))) {
            runtimeSummaryMax = parsed;
            runtimeSummaryMaxLane = laneID.str();
          }
        }
      }
    } else if (!laneID.empty()) {
      if (auto runtimeIt = runtimeNanosByLane.find(laneID);
          runtimeIt != runtimeNanosByLane.end()) {
        laneBudgetRow.hasRuntimeNanos = true;
        laneBudgetRow.runtimeNanos = runtimeIt->second;
        ++runtimeSummaryMatchedRows;
      }
    }
    laneBudgetRow.configErrorCode =
        (configErrorCodeCol != static_cast<size_t>(-1)
             ? getField(configErrorCodeCol).str()
             : std::string());
    laneBudgetRow.configErrorReason =
        (configErrorReasonCol != static_cast<size_t>(-1)
             ? getField(configErrorReasonCol).str()
             : std::string());
    if (status == "PASS")
      ++lanesPass;
    else if (status == "FAIL")
      ++lanesFail;
    else if (status == "SKIP")
      ++lanesSkip;
    if (gate == "PASS")
      ++gatePass;
    else if (gate == "FAIL")
      ++gateFail;
    else if (gate == "SKIP")
      ++gateSkip;

    if (auto coverage = parseOptionalDouble(getField(coverageCol))) {
      coverageSum += *coverage;
      ++coverageCount;
    }
    if (hasResultsPrequalifyColumns) {
      ++prequalifyResultsLanes;
      StringRef pairFileValue = getField(prequalifyPairFileCol);
      StringRef logFileValue = getField(prequalifyLogFileCol);
      if (hasPrequalifyPairFileColumn && !pairFileValue.empty() &&
          pairFileValue != "-")
        ++prequalifyResultsPairFilePresentLanes;
      if (hasPrequalifyLogFileColumn && !logFileValue.empty() &&
          logFileValue != "-")
        ++prequalifyResultsLogFilePresentLanes;
      laneBudgetRow.prequalifyPairFile =
          (hasPrequalifyPairFileColumn && !pairFileValue.empty()
               ? pairFileValue.str()
               : std::string("-"));
      laneBudgetRow.prequalifyLogFile =
          (hasPrequalifyLogFileColumn && !logFileValue.empty()
               ? logFileValue.str()
               : std::string("-"));
      bool rowSummaryPresent = false;
      if (prequalifySummaryPresentCol != static_cast<size_t>(-1)) {
        StringRef presentValue = getField(prequalifySummaryPresentCol);
        if (!presentValue.empty() && presentValue != "-") {
          if (presentValue == "1")
            rowSummaryPresent = true;
          else if (presentValue != "0")
            ++prequalifyResultsInvalidMetricValues;
        }
      }
      if (rowSummaryPresent)
        ++prequalifyResultsSummaryPresentLanes;
      if (rowSummaryPresent && hasPrequalifyPairFileColumn &&
          (pairFileValue.empty() || pairFileValue == "-"))
        ++prequalifyResultsSummaryPresentMissingPairFileLanes;
      if (rowSummaryPresent && hasPrequalifyLogFileColumn &&
          (logFileValue.empty() || logFileValue == "-"))
        ++prequalifyResultsSummaryPresentMissingLogFileLanes;
      if (prequalifyTotalMutantsCol != static_cast<size_t>(-1)) {
        StringRef totalValue = getField(prequalifyTotalMutantsCol);
        if (rowSummaryPresent && (totalValue.empty() || totalValue == "-"))
          ++prequalifyResultsSummaryMissingLanes;
      } else if (rowSummaryPresent) {
        ++prequalifyResultsSummaryMissingLanes;
      }
      addOptionalResultMetric(fields, prequalifyTotalMutantsCol,
                              prequalifyResultsTotalMutantsSum,
                              /*allowDash=*/true);
      addOptionalResultMetric(fields, prequalifyNotPropagatedMutantsCol,
                              prequalifyResultsNotPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyPropagatedMutantsCol,
                              prequalifyResultsPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCreateMutatedErrorMutantsCol,
                              prequalifyResultsCreateMutatedErrorMutantsSum);
      addOptionalResultMetric(fields, prequalifyProbeErrorMutantsCol,
                              prequalifyResultsProbeErrorMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdTokenNotPropagatedMutantsCol,
                              prequalifyResultsCmdTokenNotPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdTokenPropagatedMutantsCol,
                              prequalifyResultsCmdTokenPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdRCNotPropagatedMutantsCol,
                              prequalifyResultsCmdRCNotPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdRCPropagatedMutantsCol,
                              prequalifyResultsCmdRCPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdTimeoutPropagatedMutantsCol,
                              prequalifyResultsCmdTimeoutPropagatedMutantsSum);
      addOptionalResultMetric(fields, prequalifyCmdErrorMutantsCol,
                              prequalifyResultsCmdErrorMutantsSum);
      laneBudgetRow.hasPrequalifySummary = rowSummaryPresent;
      readOptionalResultMetric(fields, prequalifyTotalMutantsCol,
                               laneBudgetRow.prequalifyTotalMutants,
                               /*allowDash=*/true);
      readOptionalResultMetric(fields, prequalifyNotPropagatedMutantsCol,
                               laneBudgetRow.prequalifyNotPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyPropagatedMutantsCol,
                               laneBudgetRow.prequalifyPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCreateMutatedErrorMutantsCol,
                               laneBudgetRow.prequalifyCreateMutatedErrorMutants);
      readOptionalResultMetric(fields, prequalifyProbeErrorMutantsCol,
                               laneBudgetRow.prequalifyProbeErrorMutants);
      readOptionalResultMetric(
          fields, prequalifyCmdTokenNotPropagatedMutantsCol,
          laneBudgetRow.prequalifyCmdTokenNotPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCmdTokenPropagatedMutantsCol,
                               laneBudgetRow.prequalifyCmdTokenPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCmdRCNotPropagatedMutantsCol,
                               laneBudgetRow.prequalifyCmdRCNotPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCmdRCPropagatedMutantsCol,
                               laneBudgetRow.prequalifyCmdRCPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCmdTimeoutPropagatedMutantsCol,
                               laneBudgetRow.prequalifyCmdTimeoutPropagatedMutants);
      readOptionalResultMetric(fields, prequalifyCmdErrorMutantsCol,
                               laneBudgetRow.prequalifyCmdErrorMutants);
      if (!laneID.empty()) {
        MatrixPrequalifyLaneMetrics laneMetrics;
        laneMetrics.hasSummary = rowSummaryPresent;
        laneMetrics.totalMutants = laneBudgetRow.prequalifyTotalMutants;
        laneMetrics.notPropagatedMutants =
            laneBudgetRow.prequalifyNotPropagatedMutants;
        laneMetrics.propagatedMutants = laneBudgetRow.prequalifyPropagatedMutants;
        laneMetrics.createMutatedErrorMutants =
            laneBudgetRow.prequalifyCreateMutatedErrorMutants;
        laneMetrics.probeErrorMutants = laneBudgetRow.prequalifyProbeErrorMutants;
        laneMetrics.cmdTokenNotPropagatedMutants =
            laneBudgetRow.prequalifyCmdTokenNotPropagatedMutants;
        laneMetrics.cmdTokenPropagatedMutants =
            laneBudgetRow.prequalifyCmdTokenPropagatedMutants;
        laneMetrics.cmdRCNotPropagatedMutants =
            laneBudgetRow.prequalifyCmdRCNotPropagatedMutants;
        laneMetrics.cmdRCPropagatedMutants =
            laneBudgetRow.prequalifyCmdRCPropagatedMutants;
        laneMetrics.cmdTimeoutPropagatedMutants =
            laneBudgetRow.prequalifyCmdTimeoutPropagatedMutants;
        laneMetrics.cmdErrorMutants = laneBudgetRow.prequalifyCmdErrorMutants;
        resultsPrequalifyByLane[laneID] = laneMetrics;
      }
    }

    std::string metricsPath;
    bool usedMetricsFileFallback = false;
    if (metricsCol != static_cast<size_t>(-1)) {
      StringRef metricsValue = getField(metricsCol);
      if (!metricsValue.empty() && metricsValue != "-")
        metricsPath = resolveRelativeTo(matrixOutDir, metricsValue);
    }
    if (metricsPath.empty() && !laneID.empty()) {
      usedMetricsFileFallback = true;
      SmallString<256> fallbackPath(matrixOutDir);
      sys::path::append(fallbackPath, laneID, "metrics.tsv");
      metricsPath = std::string(fallbackPath.str());
    }
    if (usedMetricsFileFallback)
      ++resultsMetricsFileFallbackLanes;
    if (metricsPath.empty() || !sys::fs::exists(metricsPath)) {
      ++lanesMissingMetrics;
      if (laneBudgetRowsOut)
        laneBudgetRowsOut->push_back(std::move(laneBudgetRow));
      continue;
    }

    StringMap<std::string> metrics;
    if (!parseKeyValueTSV(metricsPath, metrics, error))
      return false;
    ++lanesWithMetrics;
    laneBudgetRow.hasMetrics = true;
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

    uint64_t detectedMutants = 0;
    if (tryGetMetric(metrics, "detected_mutants", detectedMutants)) {
      laneBudgetRow.detectedMutants = detectedMutants;
      updateLowestLane(detectedMutants, laneID, laneBudgetSawDetected,
                       laneBudgetMinDetectedMutants,
                       laneBudgetLowestDetectedLaneID);
      if (detectedMutants == 0)
        ++laneBudgetLanesZeroDetectedMutants;
    }

    uint64_t timeoutMutants = 0;
    if (tryGetMetric(metrics, "global_filter_timeout_mutants", timeoutMutants)) {
      laneBudgetRow.timeoutMutants = timeoutMutants;
      updateWorstLane(timeoutMutants, laneID, laneBudgetMaxTimeoutMutants,
                      laneBudgetWorstTimeoutLaneID);
      if (timeoutMutants > 0)
        ++laneBudgetLanesNonZeroTimeoutMutants;
    }
    uint64_t lecUnknownMutants = 0;
    if (tryGetMetric(metrics, "global_filter_lec_unknown_mutants",
                     lecUnknownMutants)) {
      laneBudgetRow.lecUnknownMutants = lecUnknownMutants;
      updateWorstLane(lecUnknownMutants, laneID, laneBudgetMaxLECUnknownMutants,
                      laneBudgetWorstLECUnknownLaneID);
      if (lecUnknownMutants > 0)
        ++laneBudgetLanesNonZeroLECUnknownMutants;
    }
    uint64_t bmcUnknownMutants = 0;
    if (tryGetMetric(metrics, "global_filter_bmc_unknown_mutants",
                     bmcUnknownMutants)) {
      laneBudgetRow.bmcUnknownMutants = bmcUnknownMutants;
      updateWorstLane(bmcUnknownMutants, laneID, laneBudgetMaxBMCUnknownMutants,
                      laneBudgetWorstBMCUnknownLaneID);
      if (bmcUnknownMutants > 0)
        ++laneBudgetLanesNonZeroBMCUnknownMutants;
    }
    uint64_t errors = 0;
    if (tryGetMetric(metrics, "errors", errors)) {
      laneBudgetRow.errors = errors;
      updateWorstLane(errors, laneID, laneBudgetMaxErrors,
                      laneBudgetWorstErrorsLaneID);
    }

    if (laneBudgetRowsOut)
      laneBudgetRowsOut->push_back(std::move(laneBudgetRow));
  }

  rows.emplace_back("matrix.out_dir", std::string(matrixOutDir));
  rows.emplace_back("matrix.results_file", std::string(resultsPath.str()));
  rows.emplace_back("matrix.runtime_summary_file",
                    runtimeSummaryPresent ? std::string(runtimeSummaryPath.str())
                                          : std::string("-"));
  rows.emplace_back("matrix.runtime_summary_present",
                    runtimeSummaryPresent ? "1" : "0");
  rows.emplace_back("matrix.runtime_summary_rows",
                    std::to_string(runtimeSummaryRows));
  rows.emplace_back("matrix.runtime_summary_invalid_rows",
                    std::to_string(runtimeSummaryInvalidRows));
  rows.emplace_back("matrix.runtime_summary_matched_rows",
                    std::to_string(runtimeSummaryMatchedRows));
  rows.emplace_back("matrix.runtime_ns_sum", std::to_string(runtimeSummarySum));
  uint64_t runtimeAvg =
      runtimeSummaryRows ? (runtimeSummarySum / runtimeSummaryRows) : 0;
  rows.emplace_back("matrix.runtime_ns_avg", std::to_string(runtimeAvg));
  rows.emplace_back("matrix.runtime_ns_max", std::to_string(runtimeSummaryMax));
  rows.emplace_back("matrix.runtime_ns_max_lane", runtimeSummaryMaxLane);
  rows.emplace_back("matrix.lanes_total", std::to_string(lanesTotal));
  rows.emplace_back("matrix.lanes_pass", std::to_string(lanesPass));
  rows.emplace_back("matrix.lanes_fail", std::to_string(lanesFail));
  rows.emplace_back("matrix.lanes_skip", std::to_string(lanesSkip));
  rows.emplace_back("matrix.gate_pass", std::to_string(gatePass));
  rows.emplace_back("matrix.gate_fail", std::to_string(gateFail));
  rows.emplace_back("matrix.gate_skip", std::to_string(gateSkip));
  rows.emplace_back("matrix.lanes_with_metrics", std::to_string(lanesWithMetrics));
  rows.emplace_back("matrix.lanes_missing_metrics",
                    std::to_string(lanesMissingMetrics));
  rows.emplace_back("matrix.invalid_metric_values",
                    std::to_string(invalidMetricValues));
  rows.emplace_back("matrix.results_metrics_file_column_present",
                    hasMetricsFileColumn ? "1" : "0");
  rows.emplace_back("matrix.results_metrics_file_fallback_lanes",
                    std::to_string(resultsMetricsFileFallbackLanes));
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
  rows.emplace_back("matrix.lane_budget.max_global_filter_timeout_mutants",
                    std::to_string(laneBudgetMaxTimeoutMutants));
  rows.emplace_back("matrix.lane_budget.max_global_filter_lec_unknown_mutants",
                    std::to_string(laneBudgetMaxLECUnknownMutants));
  rows.emplace_back("matrix.lane_budget.max_global_filter_bmc_unknown_mutants",
                    std::to_string(laneBudgetMaxBMCUnknownMutants));
  rows.emplace_back("matrix.lane_budget.max_errors",
                    std::to_string(laneBudgetMaxErrors));
  rows.emplace_back("matrix.lane_budget.worst_global_filter_timeout_mutants_lane",
                    laneBudgetWorstTimeoutLaneID);
  rows.emplace_back("matrix.lane_budget.worst_global_filter_timeout_mutants_value",
                    std::to_string(laneBudgetMaxTimeoutMutants));
  rows.emplace_back(
      "matrix.lane_budget.worst_global_filter_lec_unknown_mutants_lane",
      laneBudgetWorstLECUnknownLaneID);
  rows.emplace_back(
      "matrix.lane_budget.worst_global_filter_lec_unknown_mutants_value",
      std::to_string(laneBudgetMaxLECUnknownMutants));
  rows.emplace_back(
      "matrix.lane_budget.worst_global_filter_bmc_unknown_mutants_lane",
      laneBudgetWorstBMCUnknownLaneID);
  rows.emplace_back(
      "matrix.lane_budget.worst_global_filter_bmc_unknown_mutants_value",
      std::to_string(laneBudgetMaxBMCUnknownMutants));
  rows.emplace_back("matrix.lane_budget.worst_errors_lane",
                    laneBudgetWorstErrorsLaneID);
  rows.emplace_back("matrix.lane_budget.worst_errors_value",
                    std::to_string(laneBudgetMaxErrors));
  rows.emplace_back("matrix.lane_budget.lowest_detected_mutants_lane",
                    laneBudgetLowestDetectedLaneID);
  rows.emplace_back("matrix.lane_budget.lowest_detected_mutants_value",
                    std::to_string(laneBudgetSawDetected ? laneBudgetMinDetectedMutants
                                                         : 0));
  rows.emplace_back("matrix.lane_budget.min_detected_mutants",
                    std::to_string(laneBudgetSawDetected ? laneBudgetMinDetectedMutants
                                                         : 0));
  rows.emplace_back("matrix.lane_budget.lanes_zero_detected_mutants",
                    std::to_string(laneBudgetLanesZeroDetectedMutants));
  rows.emplace_back(
      "matrix.lane_budget.lanes_nonzero_global_filter_timeout_mutants",
      std::to_string(laneBudgetLanesNonZeroTimeoutMutants));
  rows.emplace_back(
      "matrix.lane_budget.lanes_nonzero_global_filter_lec_unknown_mutants",
      std::to_string(laneBudgetLanesNonZeroLECUnknownMutants));
  rows.emplace_back(
      "matrix.lane_budget.lanes_nonzero_global_filter_bmc_unknown_mutants",
      std::to_string(laneBudgetLanesNonZeroBMCUnknownMutants));
  rows.emplace_back("matrix.lane_budget.rows_total", std::to_string(lanesTotal));
  if (laneBudgetRowsOut) {
    StringMap<uint64_t> laneBaseCounts;
    for (const auto &lane : *laneBudgetRowsOut) {
      std::string safeLane = sanitizeReportKeySegment(lane.laneID);
      uint64_t ordinal = ++laneBaseCounts[safeLane];
      std::string laneBase =
          ordinal == 1
              ? (Twine("matrix.lane_budget.lane.") + safeLane).str()
              : (Twine("matrix.lane_budget.lane.") + safeLane + "__" +
                 Twine(ordinal))
                    .str();
      rows.emplace_back((Twine(laneBase) + ".lane_id").str(), lane.laneID);
      rows.emplace_back((Twine(laneBase) + ".status").str(),
                        lane.status.empty() ? std::string("-") : lane.status);
      rows.emplace_back((Twine(laneBase) + ".gate_status").str(),
                        lane.gateStatus.empty() ? std::string("-")
                                                : lane.gateStatus);
      bool laneIsSkip = lane.status == "SKIP" || lane.gateStatus == "SKIP";
      bool laneIsStopOnFailSkip =
          laneIsSkip && lane.configErrorCode == "STOP_ON_FAIL";
      rows.emplace_back((Twine(laneBase) + ".is_skip").str(),
                        laneIsSkip ? "1" : "0");
      rows.emplace_back((Twine(laneBase) + ".is_stop_on_fail_skip").str(),
                        laneIsStopOnFailSkip ? "1" : "0");
      rows.emplace_back((Twine(laneBase) + ".skip_reason_code").str(),
                        lane.configErrorCode.empty() ? std::string("-")
                                                     : lane.configErrorCode);
      rows.emplace_back((Twine(laneBase) + ".skip_reason").str(),
                        lane.configErrorReason.empty() ? std::string("-")
                                                       : lane.configErrorReason);
      rows.emplace_back((Twine(laneBase) + ".has_metrics").str(),
                        lane.hasMetrics ? "1" : "0");
      rows.emplace_back((Twine(laneBase) + ".has_runtime_ns").str(),
                        lane.hasRuntimeNanos ? "1" : "0");
      rows.emplace_back((Twine(laneBase) + ".runtime_ns").str(),
                        lane.hasRuntimeNanos ? std::to_string(lane.runtimeNanos)
                                             : std::string("-"));
      rows.emplace_back((Twine(laneBase) + ".prequalify_summary_present").str(),
                        lane.hasPrequalifySummary ? "1" : "0");
      rows.emplace_back((Twine(laneBase) + ".prequalify_pair_file").str(),
                        lane.prequalifyPairFile.empty() ? std::string("-")
                                                        : lane.prequalifyPairFile);
      rows.emplace_back((Twine(laneBase) + ".prequalify_log_file").str(),
                        lane.prequalifyLogFile.empty() ? std::string("-")
                                                       : lane.prequalifyLogFile);
      rows.emplace_back((Twine(laneBase) + ".prequalify_total_mutants").str(),
                        lane.hasPrequalifySummary
                            ? std::to_string(lane.prequalifyTotalMutants)
                            : std::string("-"));
      rows.emplace_back(
          (Twine(laneBase) + ".prequalify_not_propagated_mutants").str(),
          std::to_string(lane.prequalifyNotPropagatedMutants));
      rows.emplace_back((Twine(laneBase) + ".prequalify_propagated_mutants").str(),
                        std::to_string(lane.prequalifyPropagatedMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".prequalify_create_mutated_error_mutants").str(),
          std::to_string(lane.prequalifyCreateMutatedErrorMutants));
      rows.emplace_back((Twine(laneBase) + ".prequalify_probe_error_mutants").str(),
                        std::to_string(lane.prequalifyProbeErrorMutants));
      rows.emplace_back(
          (Twine(laneBase) +
           ".prequalify_cmd_token_not_propagated_mutants")
              .str(),
          std::to_string(lane.prequalifyCmdTokenNotPropagatedMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".prequalify_cmd_token_propagated_mutants").str(),
          std::to_string(lane.prequalifyCmdTokenPropagatedMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".prequalify_cmd_rc_not_propagated_mutants").str(),
          std::to_string(lane.prequalifyCmdRCNotPropagatedMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".prequalify_cmd_rc_propagated_mutants").str(),
          std::to_string(lane.prequalifyCmdRCPropagatedMutants));
      rows.emplace_back(
          (Twine(laneBase) +
           ".prequalify_cmd_timeout_propagated_mutants")
              .str(),
          std::to_string(lane.prequalifyCmdTimeoutPropagatedMutants));
      rows.emplace_back((Twine(laneBase) + ".prequalify_cmd_error_mutants").str(),
                        std::to_string(lane.prequalifyCmdErrorMutants));
      rows.emplace_back((Twine(laneBase) + ".detected_mutants").str(),
                        std::to_string(lane.detectedMutants));
      rows.emplace_back((Twine(laneBase) + ".errors").str(),
                        std::to_string(lane.errors));
      rows.emplace_back(
          (Twine(laneBase) + ".global_filter_timeout_mutants").str(),
          std::to_string(lane.timeoutMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".global_filter_lec_unknown_mutants").str(),
          std::to_string(lane.lecUnknownMutants));
      rows.emplace_back(
          (Twine(laneBase) + ".global_filter_bmc_unknown_mutants").str(),
          std::to_string(lane.bmcUnknownMutants));
    }
  }
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
  rows.emplace_back("matrix.prequalify_results_columns_present",
                    hasResultsPrequalifyColumns ? "1" : "0");
  rows.emplace_back("matrix.prequalify_results_pair_file_column_present",
                    hasPrequalifyPairFileColumn ? "1" : "0");
  rows.emplace_back("matrix.prequalify_results_log_file_column_present",
                    hasPrequalifyLogFileColumn ? "1" : "0");
  rows.emplace_back("matrix.prequalify_results_lanes",
                    std::to_string(prequalifyResultsLanes));
  rows.emplace_back("matrix.prequalify_results_summary_present_lanes",
                    std::to_string(prequalifyResultsSummaryPresentLanes));
  rows.emplace_back("matrix.prequalify_results_summary_missing_lanes",
                    std::to_string(prequalifyResultsSummaryMissingLanes));
  rows.emplace_back("matrix.prequalify_results_pair_file_present_lanes",
                    std::to_string(prequalifyResultsPairFilePresentLanes));
  rows.emplace_back("matrix.prequalify_results_log_file_present_lanes",
                    std::to_string(prequalifyResultsLogFilePresentLanes));
  rows.emplace_back("matrix.prequalify_results_summary_present_missing_pair_file_lanes",
                    std::to_string(
                        prequalifyResultsSummaryPresentMissingPairFileLanes));
  rows.emplace_back("matrix.prequalify_results_summary_present_missing_log_file_lanes",
                    std::to_string(
                        prequalifyResultsSummaryPresentMissingLogFileLanes));
  rows.emplace_back("matrix.prequalify_results_invalid_metric_values",
                    std::to_string(prequalifyResultsInvalidMetricValues));
  rows.emplace_back("matrix.prequalify_results_total_mutants_sum",
                    std::to_string(prequalifyResultsTotalMutantsSum));
  rows.emplace_back("matrix.prequalify_results_not_propagated_mutants_sum",
                    std::to_string(prequalifyResultsNotPropagatedMutantsSum));
  rows.emplace_back("matrix.prequalify_results_propagated_mutants_sum",
                    std::to_string(prequalifyResultsPropagatedMutantsSum));
  rows.emplace_back(
      "matrix.prequalify_results_create_mutated_error_mutants_sum",
      std::to_string(prequalifyResultsCreateMutatedErrorMutantsSum));
  rows.emplace_back("matrix.prequalify_results_probe_error_mutants_sum",
                    std::to_string(prequalifyResultsProbeErrorMutantsSum));
  rows.emplace_back(
      "matrix.prequalify_results_cmd_token_not_propagated_mutants_sum",
      std::to_string(prequalifyResultsCmdTokenNotPropagatedMutantsSum));
  rows.emplace_back("matrix.prequalify_results_cmd_token_propagated_mutants_sum",
                    std::to_string(prequalifyResultsCmdTokenPropagatedMutantsSum));
  rows.emplace_back("matrix.prequalify_results_cmd_rc_not_propagated_mutants_sum",
                    std::to_string(prequalifyResultsCmdRCNotPropagatedMutantsSum));
  rows.emplace_back("matrix.prequalify_results_cmd_rc_propagated_mutants_sum",
                    std::to_string(prequalifyResultsCmdRCPropagatedMutantsSum));
  rows.emplace_back(
      "matrix.prequalify_results_cmd_timeout_propagated_mutants_sum",
      std::to_string(prequalifyResultsCmdTimeoutPropagatedMutantsSum));
  rows.emplace_back("matrix.prequalify_results_cmd_error_mutants_sum",
                    std::to_string(prequalifyResultsCmdErrorMutantsSum));

  SmallString<256> nativeSummaryPath(matrixOutDir);
  sys::path::append(nativeSummaryPath, "native_matrix_prequalify_summary.tsv");
  rows.emplace_back("matrix.native_prequalify_summary_file",
                    std::string(nativeSummaryPath.str()));
  uint64_t nativeSummaryFileExists = sys::fs::exists(nativeSummaryPath) ? 1 : 0;
  rows.emplace_back("matrix.native_prequalify_summary_file_exists",
                    std::to_string(nativeSummaryFileExists));

  uint64_t nativeSummaryLanes = 0;
  uint64_t nativeSummaryMissingLanes = 0;
  uint64_t nativeSummaryInvalidValues = 0;
  uint64_t nativeTotalMutants = 0;
  uint64_t nativeNotPropagatedMutants = 0;
  uint64_t nativePropagatedMutants = 0;
  uint64_t nativeCreateMutatedErrorMutants = 0;
  uint64_t nativeProbeErrorMutants = 0;
  uint64_t nativeCmdTokenNotPropagatedMutants = 0;
  uint64_t nativeCmdTokenPropagatedMutants = 0;
  uint64_t nativeCmdRCNotPropagatedMutants = 0;
  uint64_t nativeCmdRCPropagatedMutants = 0;
  uint64_t nativeCmdTimeoutPropagatedMutants = 0;
  uint64_t nativeCmdErrorMutants = 0;
  StringMap<MatrixPrequalifyLaneMetrics> nativePrequalifyByLane;
  if (nativeSummaryFileExists) {
    auto summaryBufferOrErr = MemoryBuffer::getFile(nativeSummaryPath);
    if (!summaryBufferOrErr) {
      error = (Twine("circt-mut report: unable to read native matrix "
                     "prequalify summary file: ") +
               nativeSummaryPath)
                  .str();
      return false;
    }
    SmallVector<StringRef, 256> summaryLines;
    summaryBufferOrErr.get()->getBuffer().split(summaryLines, '\n',
                                                /*MaxSplit=*/-1,
                                                /*KeepEmpty=*/false);
    if (!summaryLines.empty()) {
      splitTSVLine(summaryLines.front().rtrim("\r"), fields);
      StringMap<size_t> summaryColumns;
      for (size_t i = 0; i < fields.size(); ++i)
        summaryColumns[fields[i].trim()] = i;

      auto getSummaryColumn = [&](StringRef name,
                                  size_t &dst) -> bool {
        auto it = summaryColumns.find(name);
        if (it == summaryColumns.end()) {
          error = (Twine("circt-mut report: missing required native "
                         "matrix prequalify summary column: ") +
                   name + " in " + nativeSummaryPath)
                      .str();
          return false;
        }
        dst = it->second;
        return true;
      };
      size_t totalCol = 0, notPropCol = 0, propCol = 0, createErrCol = 0,
             probeErrCol = 0, cmdTokenNotPropCol = 0, cmdTokenPropCol = 0,
             cmdRCNotPropCol = 0, cmdRCPropCol = 0, cmdTimeoutPropCol = 0,
             cmdErrCol = 0;
      if (!getSummaryColumn("prequalify_total_mutants", totalCol) ||
          !getSummaryColumn("prequalify_not_propagated_mutants", notPropCol) ||
          !getSummaryColumn("prequalify_propagated_mutants", propCol) ||
          !getSummaryColumn("prequalify_create_mutated_error_mutants",
                            createErrCol) ||
          !getSummaryColumn("prequalify_probe_error_mutants", probeErrCol) ||
          !getSummaryColumn("prequalify_cmd_token_not_propagated_mutants",
                            cmdTokenNotPropCol) ||
          !getSummaryColumn("prequalify_cmd_token_propagated_mutants",
                            cmdTokenPropCol) ||
          !getSummaryColumn("prequalify_cmd_rc_not_propagated_mutants",
                            cmdRCNotPropCol) ||
          !getSummaryColumn("prequalify_cmd_rc_propagated_mutants",
                            cmdRCPropCol) ||
          !getSummaryColumn("prequalify_cmd_timeout_propagated_mutants",
                            cmdTimeoutPropCol) ||
          !getSummaryColumn("prequalify_cmd_error_mutants", cmdErrCol))
        return false;

      auto addSummaryMetric = [&](ArrayRef<StringRef> summaryFields, size_t col,
                                  uint64_t &dst) {
        if (col >= summaryFields.size())
          return;
        StringRef value = summaryFields[col].trim();
        if (value.empty() || value == "-")
          return;
        uint64_t parsed = 0;
        if (value.getAsInteger(10, parsed)) {
          ++nativeSummaryInvalidValues;
          return;
        }
        dst += parsed;
      };

      for (size_t lineNo = 1; lineNo < summaryLines.size(); ++lineNo) {
        StringRef summaryLine = summaryLines[lineNo].rtrim("\r");
        if (summaryLine.trim().empty())
          continue;
        splitTSVLine(summaryLine, fields);
        StringRef summaryLaneID;
        if (auto it = summaryColumns.find("lane_id"); it != summaryColumns.end() &&
            it->second < fields.size())
          summaryLaneID = fields[it->second].trim();
        ++nativeSummaryLanes;
        if (totalCol >= fields.size() || fields[totalCol].trim().empty() ||
            fields[totalCol].trim() == "-")
          ++nativeSummaryMissingLanes;
        addSummaryMetric(fields, totalCol, nativeTotalMutants);
        addSummaryMetric(fields, notPropCol, nativeNotPropagatedMutants);
        addSummaryMetric(fields, propCol, nativePropagatedMutants);
        addSummaryMetric(fields, createErrCol, nativeCreateMutatedErrorMutants);
        addSummaryMetric(fields, probeErrCol, nativeProbeErrorMutants);
        addSummaryMetric(fields, cmdTokenNotPropCol,
                         nativeCmdTokenNotPropagatedMutants);
        addSummaryMetric(fields, cmdTokenPropCol,
                         nativeCmdTokenPropagatedMutants);
        addSummaryMetric(fields, cmdRCNotPropCol, nativeCmdRCNotPropagatedMutants);
        addSummaryMetric(fields, cmdRCPropCol, nativeCmdRCPropagatedMutants);
        addSummaryMetric(fields, cmdTimeoutPropCol,
                         nativeCmdTimeoutPropagatedMutants);
        addSummaryMetric(fields, cmdErrCol, nativeCmdErrorMutants);
        if (!summaryLaneID.empty()) {
          MatrixPrequalifyLaneMetrics laneMetrics;
          StringRef totalValue =
              totalCol < fields.size() ? fields[totalCol].trim() : StringRef();
          laneMetrics.hasSummary = !totalValue.empty() && totalValue != "-";
          auto parseLaneMetric = [&](size_t col, uint64_t &dst) {
            if (col >= fields.size())
              return;
            StringRef value = fields[col].trim();
            if (value.empty() || value == "-")
              return;
            uint64_t parsed = 0;
            if (value.getAsInteger(10, parsed)) {
              ++nativeSummaryInvalidValues;
              return;
            }
            dst = parsed;
          };
          parseLaneMetric(totalCol, laneMetrics.totalMutants);
          parseLaneMetric(notPropCol, laneMetrics.notPropagatedMutants);
          parseLaneMetric(propCol, laneMetrics.propagatedMutants);
          parseLaneMetric(createErrCol, laneMetrics.createMutatedErrorMutants);
          parseLaneMetric(probeErrCol, laneMetrics.probeErrorMutants);
          parseLaneMetric(cmdTokenNotPropCol,
                          laneMetrics.cmdTokenNotPropagatedMutants);
          parseLaneMetric(cmdTokenPropCol, laneMetrics.cmdTokenPropagatedMutants);
          parseLaneMetric(cmdRCNotPropCol, laneMetrics.cmdRCNotPropagatedMutants);
          parseLaneMetric(cmdRCPropCol, laneMetrics.cmdRCPropagatedMutants);
          parseLaneMetric(cmdTimeoutPropCol,
                          laneMetrics.cmdTimeoutPropagatedMutants);
          parseLaneMetric(cmdErrCol, laneMetrics.cmdErrorMutants);
          nativePrequalifyByLane[summaryLaneID] = laneMetrics;
        }
      }
    }
  }
  rows.emplace_back("matrix.native_prequalify_summary_lanes",
                    std::to_string(nativeSummaryLanes));
  rows.emplace_back("matrix.native_prequalify_summary_missing_lanes",
                    std::to_string(nativeSummaryMissingLanes));
  rows.emplace_back("matrix.native_prequalify_invalid_metric_values",
                    std::to_string(nativeSummaryInvalidValues));
  rows.emplace_back("matrix.native_prequalify_total_mutants_sum",
                    std::to_string(nativeTotalMutants));
  rows.emplace_back("matrix.native_prequalify_not_propagated_mutants_sum",
                    std::to_string(nativeNotPropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_propagated_mutants_sum",
                    std::to_string(nativePropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_create_mutated_error_mutants_sum",
                    std::to_string(nativeCreateMutatedErrorMutants));
  rows.emplace_back("matrix.native_prequalify_probe_error_mutants_sum",
                    std::to_string(nativeProbeErrorMutants));
  rows.emplace_back(
      "matrix.native_prequalify_cmd_token_not_propagated_mutants_sum",
      std::to_string(nativeCmdTokenNotPropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_cmd_token_propagated_mutants_sum",
                    std::to_string(nativeCmdTokenPropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_cmd_rc_not_propagated_mutants_sum",
                    std::to_string(nativeCmdRCNotPropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_cmd_rc_propagated_mutants_sum",
                    std::to_string(nativeCmdRCPropagatedMutants));
  rows.emplace_back(
      "matrix.native_prequalify_cmd_timeout_propagated_mutants_sum",
      std::to_string(nativeCmdTimeoutPropagatedMutants));
  rows.emplace_back("matrix.native_prequalify_cmd_error_mutants_sum",
                    std::to_string(nativeCmdErrorMutants));
  uint64_t prequalifyDriftNonZeroMetrics = 0;
  auto appendDriftMetric = [&](StringRef name, uint64_t resultsValue,
                               uint64_t nativeValue) {
    int64_t delta = static_cast<int64_t>(resultsValue) -
                    static_cast<int64_t>(nativeValue);
    uint64_t absDelta =
        delta >= 0 ? static_cast<uint64_t>(delta)
                   : static_cast<uint64_t>(-delta);
    rows.emplace_back((Twine("matrix.prequalify_drift.") + name + ".delta").str(),
                      std::to_string(delta));
    rows.emplace_back(
        (Twine("matrix.prequalify_drift.") + name + ".abs_delta").str(),
        std::to_string(absDelta));
    if (absDelta != 0)
      ++prequalifyDriftNonZeroMetrics;
  };
  bool prequalifyDriftComparable =
      hasResultsPrequalifyColumns && nativeSummaryFileExists != 0;
  rows.emplace_back("matrix.prequalify_drift_comparable",
                    prequalifyDriftComparable ? "1" : "0");
  uint64_t prequalifyDriftLaneRowsCompared = 0;
  uint64_t prequalifyDriftLaneRowsMismatch = 0;
  uint64_t prequalifyDriftLaneRowsMissingInResults = 0;
  uint64_t prequalifyDriftLaneRowsMissingInNative = 0;
  if (prequalifyDriftComparable) {
    appendDriftMetric("summary_present_lanes",
                      prequalifyResultsSummaryPresentLanes, nativeSummaryLanes);
    appendDriftMetric("summary_missing_lanes",
                      prequalifyResultsSummaryMissingLanes,
                      nativeSummaryMissingLanes);
    appendDriftMetric("total_mutants", prequalifyResultsTotalMutantsSum,
                      nativeTotalMutants);
    appendDriftMetric("not_propagated_mutants",
                      prequalifyResultsNotPropagatedMutantsSum,
                      nativeNotPropagatedMutants);
    appendDriftMetric("propagated_mutants",
                      prequalifyResultsPropagatedMutantsSum,
                      nativePropagatedMutants);
    appendDriftMetric("create_mutated_error_mutants",
                      prequalifyResultsCreateMutatedErrorMutantsSum,
                      nativeCreateMutatedErrorMutants);
    appendDriftMetric("probe_error_mutants",
                      prequalifyResultsProbeErrorMutantsSum,
                      nativeProbeErrorMutants);
    appendDriftMetric("cmd_token_not_propagated_mutants",
                      prequalifyResultsCmdTokenNotPropagatedMutantsSum,
                      nativeCmdTokenNotPropagatedMutants);
    appendDriftMetric("cmd_token_propagated_mutants",
                      prequalifyResultsCmdTokenPropagatedMutantsSum,
                      nativeCmdTokenPropagatedMutants);
    appendDriftMetric("cmd_rc_not_propagated_mutants",
                      prequalifyResultsCmdRCNotPropagatedMutantsSum,
                      nativeCmdRCNotPropagatedMutants);
    appendDriftMetric("cmd_rc_propagated_mutants",
                      prequalifyResultsCmdRCPropagatedMutantsSum,
                      nativeCmdRCPropagatedMutants);
    appendDriftMetric("cmd_timeout_propagated_mutants",
                      prequalifyResultsCmdTimeoutPropagatedMutantsSum,
                      nativeCmdTimeoutPropagatedMutants);
    appendDriftMetric("cmd_error_mutants", prequalifyResultsCmdErrorMutantsSum,
                      nativeCmdErrorMutants);

    StringSet<> laneUnion;
    for (const auto &it : resultsPrequalifyByLane)
      laneUnion.insert(it.getKey());
    for (const auto &it : nativePrequalifyByLane)
      laneUnion.insert(it.getKey());
    for (const auto &laneIt : laneUnion) {
      StringRef laneID = laneIt.getKey();
      auto resIt = resultsPrequalifyByLane.find(laneID);
      auto natIt = nativePrequalifyByLane.find(laneID);
      std::string laneKey = sanitizeReportKeySegment(laneID);
      std::string base =
          (Twine("matrix.prequalify_drift.lane.") + laneKey).str();
      if (resIt == resultsPrequalifyByLane.end()) {
        ++prequalifyDriftLaneRowsMissingInResults;
        rows.emplace_back((Twine(base) + ".status").str(), "missing_in_results");
        continue;
      }
      if (natIt == nativePrequalifyByLane.end()) {
        ++prequalifyDriftLaneRowsMissingInNative;
        rows.emplace_back((Twine(base) + ".status").str(), "missing_in_native");
        continue;
      }
      ++prequalifyDriftLaneRowsCompared;
      SmallVector<std::string, 16> mismatchFields;
      const auto &res = resIt->second;
      const auto &nat = natIt->second;
      auto checkEq = [&](StringRef name, uint64_t lhs, uint64_t rhs) {
        if (lhs != rhs)
          mismatchFields.push_back(name.str());
      };
      checkEq("summary_present", res.hasSummary ? 1 : 0,
              nat.hasSummary ? 1 : 0);
      if (res.hasSummary && nat.hasSummary)
        checkEq("total_mutants", res.totalMutants, nat.totalMutants);
      checkEq("not_propagated_mutants", res.notPropagatedMutants,
              nat.notPropagatedMutants);
      checkEq("propagated_mutants", res.propagatedMutants,
              nat.propagatedMutants);
      checkEq("create_mutated_error_mutants", res.createMutatedErrorMutants,
              nat.createMutatedErrorMutants);
      checkEq("probe_error_mutants", res.probeErrorMutants,
              nat.probeErrorMutants);
      checkEq("cmd_token_not_propagated_mutants",
              res.cmdTokenNotPropagatedMutants,
              nat.cmdTokenNotPropagatedMutants);
      checkEq("cmd_token_propagated_mutants", res.cmdTokenPropagatedMutants,
              nat.cmdTokenPropagatedMutants);
      checkEq("cmd_rc_not_propagated_mutants", res.cmdRCNotPropagatedMutants,
              nat.cmdRCNotPropagatedMutants);
      checkEq("cmd_rc_propagated_mutants", res.cmdRCPropagatedMutants,
              nat.cmdRCPropagatedMutants);
      checkEq("cmd_timeout_propagated_mutants",
              res.cmdTimeoutPropagatedMutants,
              nat.cmdTimeoutPropagatedMutants);
      checkEq("cmd_error_mutants", res.cmdErrorMutants, nat.cmdErrorMutants);

      if (mismatchFields.empty()) {
        rows.emplace_back((Twine(base) + ".status").str(), "match");
        rows.emplace_back((Twine(base) + ".mismatch_count").str(), "0");
      } else {
        ++prequalifyDriftLaneRowsMismatch;
        rows.emplace_back((Twine(base) + ".status").str(), "mismatch");
        rows.emplace_back((Twine(base) + ".mismatch_count").str(),
                          std::to_string(mismatchFields.size()));
        for (size_t i = 0; i < mismatchFields.size(); ++i)
          rows.emplace_back((Twine(base) + ".mismatch_" + Twine(i + 1)).str(),
                            mismatchFields[i]);
      }
    }
  }
  rows.emplace_back("matrix.prequalify_drift_lane_rows_compared",
                    std::to_string(prequalifyDriftLaneRowsCompared));
  rows.emplace_back("matrix.prequalify_drift_lane_rows_mismatch",
                    std::to_string(prequalifyDriftLaneRowsMismatch));
  rows.emplace_back("matrix.prequalify_drift_lane_rows_missing_in_results",
                    std::to_string(prequalifyDriftLaneRowsMissingInResults));
  rows.emplace_back("matrix.prequalify_drift_lane_rows_missing_in_native",
                    std::to_string(prequalifyDriftLaneRowsMissingInNative));
  rows.emplace_back("matrix.prequalify_drift_nonzero_metrics",
                    std::to_string(prequalifyDriftNonZeroMetrics));
  if (prequalifyDriftNonZeroOut)
    *prequalifyDriftNonZeroOut = prequalifyDriftNonZeroMetrics;
  if (prequalifyDriftComparableOut)
    *prequalifyDriftComparableOut = prequalifyDriftComparable;
  if (prequalifyDriftLaneRowsMismatchOut)
    *prequalifyDriftLaneRowsMismatchOut = prequalifyDriftLaneRowsMismatch;
  if (prequalifyDriftLaneRowsMissingInResultsOut)
    *prequalifyDriftLaneRowsMissingInResultsOut =
        prequalifyDriftLaneRowsMissingInResults;
  if (prequalifyDriftLaneRowsMissingInNativeOut)
    *prequalifyDriftLaneRowsMissingInNativeOut =
        prequalifyDriftLaneRowsMissingInNative;
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

static bool writeLaneBudgetFile(StringRef path,
                                ArrayRef<MatrixLaneBudgetRow> laneRows,
                                std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (!parent.empty()) {
    std::error_code dirEC = sys::fs::create_directories(parent);
    if (dirEC) {
      error = (Twine("circt-mut report: failed to create lane budget output "
                     "directory: ") +
               parent + ": " + dirEC.message())
                  .str();
      return false;
    }
  }

  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_Text);
  if (ec) {
    error = (Twine("circt-mut report: failed to open --lane-budget-out file: ") +
             path + ": " + ec.message())
                .str();
    return false;
  }
  os << "lane_id\tstatus\tgate_status\thas_metrics\tdetected_mutants\t"
        "errors\tglobal_filter_timeout_mutants\t"
        "global_filter_lec_unknown_mutants\t"
        "global_filter_bmc_unknown_mutants\n";
  for (const auto &row : laneRows) {
    os << row.laneID << "\t" << row.status << "\t" << row.gateStatus << "\t"
       << (row.hasMetrics ? "1" : "0") << "\t" << row.detectedMutants << "\t"
       << row.errors << "\t" << row.timeoutMutants << "\t"
       << row.lecUnknownMutants << "\t" << row.bmcUnknownMutants << "\n";
  }
  return true;
}

static SkipBudgetSummary
computeSkipBudgetSummary(ArrayRef<MatrixLaneBudgetRow> laneRows) {
  SkipBudgetSummary summary;
  summary.totalRows = laneRows.size();
  for (const auto &row : laneRows) {
    bool isSkip = row.status == "SKIP" || row.gateStatus == "SKIP";
    if (isSkip) {
      ++summary.skipRows;
      if (row.configErrorCode == "STOP_ON_FAIL")
        ++summary.stopOnFailRows;
      else
        ++summary.nonStopOnFailSkipRows;
      if (!row.configErrorReason.empty() && row.configErrorReason != "-")
        ++summary.rowsWithReason;
    } else {
      ++summary.nonSkipRows;
    }
  }
  return summary;
}

static bool writeSkipBudgetFile(StringRef path,
                                ArrayRef<MatrixLaneBudgetRow> laneRows,
                                std::string &error) {
  SmallString<256> parent(path);
  sys::path::remove_filename(parent);
  if (!parent.empty()) {
    std::error_code dirEC = sys::fs::create_directories(parent);
    if (dirEC) {
      error = (Twine("circt-mut report: failed to create skip budget output "
                     "directory: ") +
               parent + ": " + dirEC.message())
                  .str();
      return false;
    }
  }

  std::error_code ec;
  raw_fd_ostream os(path, ec, sys::fs::OF_Text);
  if (ec) {
    error = (Twine("circt-mut report: failed to open --skip-budget-out file: ") +
             path + ": " + ec.message())
                .str();
    return false;
  }
  os << "lane_id\tstatus\tgate_status\tis_skip\thas_metrics\tconfig_error_code\t"
        "config_error_reason\n";
  for (const auto &row : laneRows) {
    bool isSkip = row.status == "SKIP" || row.gateStatus == "SKIP";
    os << row.laneID << "\t" << row.status << "\t" << row.gateStatus << "\t"
       << (isSkip ? "1" : "0") << "\t" << (row.hasMetrics ? "1" : "0") << "\t"
       << (row.configErrorCode.empty() ? "-" : row.configErrorCode) << "\t"
       << (row.configErrorReason.empty() ? "-" : row.configErrorReason) << "\n";
  }
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
  auto parseBoolValue = [&](StringRef value, StringRef optName)
      -> std::optional<bool> {
    StringRef lowered = value.trim().lower();
    if (lowered == "1" || lowered == "true" || lowered == "yes" ||
        lowered == "on")
      return true;
    if (lowered == "0" || lowered == "false" || lowered == "no" ||
        lowered == "off")
      return false;
    result.error =
        (Twine("circt-mut report: invalid ") + optName + " value: " + value +
         " (expected 1|0|true|false|yes|no|on|off)")
            .str();
    return std::nullopt;
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
    if (arg == "--external-formal-results" ||
        arg.starts_with("--external-formal-results=")) {
      auto v = consumeValue(i, arg, "--external-formal-results");
      if (!v)
        return result;
      StringRef file = v->trim();
      if (file.empty()) {
        result.error = "circt-mut report: --external-formal-results requires "
                       "non-empty value";
        return result;
      }
      result.opts.externalFormalResultsFiles.push_back(file.str());
      continue;
    }
    if (arg == "--external-formal-out-dir" ||
        arg.starts_with("--external-formal-out-dir=")) {
      auto v = consumeValue(i, arg, "--external-formal-out-dir");
      if (!v)
        return result;
      StringRef dir = v->trim();
      if (dir.empty()) {
        result.error = "circt-mut report: --external-formal-out-dir requires "
                       "non-empty value";
        return result;
      }
      result.opts.externalFormalOutDir = dir.str();
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
    if (arg == "--history" || arg.starts_with("--history=")) {
      auto v = consumeValue(i, arg, "--history");
      if (!v)
        return result;
      result.opts.historyFile = v->str();
      continue;
    }
    if (arg == "--history-bootstrap") {
      result.opts.historyBootstrap = true;
      continue;
    }
    if (arg == "--history-max-runs" || arg.starts_with("--history-max-runs=")) {
      auto v = consumeValue(i, arg, "--history-max-runs");
      if (!v)
        return result;
      uint64_t parsed = 0;
      if (StringRef(*v).trim().getAsInteger(10, parsed) || parsed == 0) {
        result.error =
            (Twine("circt-mut report: invalid --history-max-runs value: ") + *v +
             " (expected positive integer)")
                .str();
        return result;
      }
      result.opts.historyMaxRuns = parsed;
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
    if (arg == "--policy-mode" || arg.starts_with("--policy-mode=")) {
      auto v = consumeValue(i, arg, "--policy-mode");
      if (!v)
        return result;
      std::string mode = StringRef(*v).trim().lower();
      if (!isMatrixPolicyMode(mode)) {
        result.error =
            (Twine("circt-mut report: invalid --policy-mode value: ") + *v +
             (Twine(" (expected ") + kMatrixPolicyModeList + ")"))
                .str();
        return result;
      }
      result.opts.policyMode = mode;
      continue;
    }
    if (arg == "--policy-stop-on-fail" ||
        arg.starts_with("--policy-stop-on-fail=")) {
      auto v = consumeValue(i, arg, "--policy-stop-on-fail");
      if (!v)
        return result;
      auto parsed = parseBoolValue(*v, "--policy-stop-on-fail");
      if (!parsed)
        return result;
      result.opts.policyStopOnFail = *parsed;
      continue;
    }
    if (arg == "--append-history" || arg.starts_with("--append-history=")) {
      auto v = consumeValue(i, arg, "--append-history");
      if (!v)
        return result;
      result.opts.appendHistoryFile = v->str();
      continue;
    }
    if (arg == "--fail-if-value-gt" || arg.starts_with("--fail-if-value-gt=")) {
      auto v = consumeValue(i, arg, "--fail-if-value-gt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-value-gt", rule, result.error))
        return result;
      result.opts.failIfValueGtRules.push_back(rule);
      continue;
    }
    if (arg == "--fail-if-value-lt" || arg.starts_with("--fail-if-value-lt=")) {
      auto v = consumeValue(i, arg, "--fail-if-value-lt");
      if (!v)
        return result;
      DeltaGateRule rule;
      if (!parseDeltaGateRule(*v, "--fail-if-value-lt", rule, result.error))
        return result;
      result.opts.failIfValueLtRules.push_back(rule);
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
    if (arg == "--lane-budget-out" || arg.starts_with("--lane-budget-out=")) {
      auto v = consumeValue(i, arg, "--lane-budget-out");
      if (!v)
        return result;
      result.opts.laneBudgetOutFile = v->str();
      continue;
    }
    if (arg == "--skip-budget-out" || arg.starts_with("--skip-budget-out=")) {
      auto v = consumeValue(i, arg, "--skip-budget-out");
      if (!v)
        return result;
      result.opts.skipBudgetOutFile = v->str();
      continue;
    }
    if (arg == "--fail-on-prequalify-drift") {
      result.opts.failOnPrequalifyDrift = true;
      result.opts.failOnPrequalifyDriftOverrideSet = true;
      continue;
    }
    if (arg == "--no-fail-on-prequalify-drift") {
      result.opts.failOnPrequalifyDrift = false;
      result.opts.failOnPrequalifyDriftOverrideSet = true;
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
  if (!result.opts.historyFile.empty() &&
      (!result.opts.compareFile.empty() ||
       !result.opts.compareHistoryLatestFile.empty() ||
       !result.opts.trendHistoryFile.empty() ||
       !result.opts.appendHistoryFile.empty())) {
    result.error =
        "circt-mut report: --history is mutually exclusive with --compare, "
        "--compare-history-latest, --trend-history, and --append-history";
    return result;
  }
  if (result.opts.historyMaxRuns > 0 && result.opts.appendHistoryFile.empty()) {
    result.error = "circt-mut report: --history-max-runs requires --append-history "
                   "or --history";
    return result;
  }
  if (result.opts.policyStopOnFail.has_value() &&
      result.opts.policyMode.empty()) {
    result.error =
        "circt-mut report: --policy-stop-on-fail requires --policy-mode";
    return result;
  }
  if (!result.opts.policyMode.empty() && !result.opts.policyProfiles.empty()) {
    result.error = "circt-mut report: --policy-mode and --policy-profile are "
                   "mutually exclusive";
    return result;
  }
  if (!result.opts.policyMode.empty() &&
      result.opts.mode != "matrix" && result.opts.mode != "all") {
    result.error = "circt-mut report: --policy-mode requires --mode matrix or "
                   "--mode all";
    return result;
  }
  if (!result.opts.historyFile.empty()) {
    if (result.opts.compareFile.empty() &&
        result.opts.compareHistoryLatestFile.empty())
      result.opts.compareHistoryLatestFile = result.opts.historyFile;
    if (result.opts.trendHistoryFile.empty())
      result.opts.trendHistoryFile = result.opts.historyFile;
    if (result.opts.appendHistoryFile.empty())
      result.opts.appendHistoryFile = result.opts.historyFile;
  }
  bool failOnPrequalifyDriftOverrideValue = result.opts.failOnPrequalifyDrift;
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
  if (result.opts.failOnPrequalifyDriftOverrideSet)
    result.opts.failOnPrequalifyDrift = failOnPrequalifyDriftOverrideValue;
  result.ok = true;
  return result;
}

static int runNativeReport(const ReportOptions &opts) {
  ReportOptions effectiveOpts = opts;
  SmallString<256> defaultCover(opts.projectDir);
  sys::path::append(defaultCover, "out", "cover");
  SmallString<256> defaultMatrix(opts.projectDir);
  sys::path::append(defaultMatrix, "out", "matrix");

  std::string coverWorkDir = std::string(defaultCover.str());
  std::string matrixOutDir = std::string(defaultMatrix.str());
  std::string compareFile = effectiveOpts.compareFile;
  std::string compareHistoryLatestFile = effectiveOpts.compareHistoryLatestFile;
  std::string historyFile = effectiveOpts.historyFile;
  std::string trendHistoryFile = effectiveOpts.trendHistoryFile;
  SmallVector<std::string, 4> externalFormalResultsFiles =
      effectiveOpts.externalFormalResultsFiles;
  std::string externalFormalOutDir = effectiveOpts.externalFormalOutDir;
  uint64_t trendWindowRuns = effectiveOpts.trendWindowRuns;
  std::string appendHistoryFile = effectiveOpts.appendHistoryFile;
  uint64_t historyMaxRuns = effectiveOpts.historyMaxRuns;
  bool historyBootstrap = effectiveOpts.historyBootstrap;
  SmallVector<std::string, 4> policyProfiles = effectiveOpts.policyProfiles;
  bool hasCLIPolicyMode = !effectiveOpts.policyMode.empty();
  bool hasCLIPolicyProfile = !effectiveOpts.policyProfiles.empty();
  std::string appliedPolicyProfileSource =
      hasCLIPolicyProfile ? "cli" : "none";
  std::string appliedPolicyMode;
  std::string appliedPolicyModeSource = "none";
  std::optional<bool> appliedPolicyStopOnFail;
  std::optional<bool> appliedPolicyStopOnFailEffective;
  std::optional<bool> appliedPolicyStopOnFailIgnored;
  std::optional<bool> failOnPrequalifyDriftOverride;
  if (effectiveOpts.failOnPrequalifyDriftOverrideSet)
    failOnPrequalifyDriftOverride = effectiveOpts.failOnPrequalifyDrift;

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
    if (compareFile.empty() && compareHistoryLatestFile.empty()) {
      auto compareIt = cfg.report.find("compare");
      auto compareHistoryIt = cfg.report.find("compare_history_latest");
      bool hasConfigCompare =
          compareIt != cfg.report.end() && !compareIt->second.empty();
      bool hasConfigCompareHistory =
          compareHistoryIt != cfg.report.end() && !compareHistoryIt->second.empty();
      if (hasConfigCompare && hasConfigCompareHistory) {
        errs() << "circt-mut report: [report] keys 'compare' and "
                  "'compare_history_latest' are mutually exclusive\n";
        return 1;
      }
      if (hasConfigCompare)
        compareFile = compareIt->second;
      else if (hasConfigCompareHistory)
        compareHistoryLatestFile = compareHistoryIt->second;
    }
    if (appendHistoryFile.empty()) {
      if (auto it = cfg.report.find("append_history");
          it != cfg.report.end() && !it->second.empty())
        appendHistoryFile = it->second;
    }
    if (historyFile.empty()) {
      if (auto it = cfg.report.find("history");
          it != cfg.report.end() && !it->second.empty())
        historyFile = it->second;
    }
    if (trendHistoryFile.empty()) {
      if (auto it = cfg.report.find("trend_history");
          it != cfg.report.end() && !it->second.empty())
        trendHistoryFile = it->second;
    }
    if (trendWindowRuns == 0) {
      if (auto it = cfg.report.find("trend_window");
          it != cfg.report.end() && !it->second.empty()) {
        if (StringRef(it->second).trim().getAsInteger(10, trendWindowRuns)) {
          errs() << "circt-mut report: invalid [report] key 'trend_window' "
                    "value '"
                 << it->second << "' (expected non-negative integer)\n";
          return 1;
        }
      }
    }
    if (historyMaxRuns == 0) {
      if (auto it = cfg.report.find("history_max_runs");
          it != cfg.report.end() && !it->second.empty()) {
        if (StringRef(it->second).trim().getAsInteger(10, historyMaxRuns) ||
            historyMaxRuns == 0) {
          errs() << "circt-mut report: invalid [report] key 'history_max_runs' "
                    "value '"
                 << it->second << "' (expected positive integer)\n";
          return 1;
        }
      }
    }
    bool hasConfigHistory =
        cfg.report.contains("history") && !cfg.report.lookup("history").empty();
    bool hasConfigCompare =
        cfg.report.contains("compare") && !cfg.report.lookup("compare").empty();
    bool hasConfigCompareHistory = cfg.report.contains("compare_history_latest") &&
                                   !cfg.report.lookup("compare_history_latest")
                                        .empty();
    bool hasConfigTrendHistory = cfg.report.contains("trend_history") &&
                                 !cfg.report.lookup("trend_history").empty();
    bool hasConfigAppendHistory = cfg.report.contains("append_history") &&
                                  !cfg.report.lookup("append_history").empty();
    bool hasCLIHistorySelector =
        !opts.historyFile.empty() || !opts.compareFile.empty() ||
        !opts.compareHistoryLatestFile.empty() || !opts.trendHistoryFile.empty() ||
        !opts.appendHistoryFile.empty();
    if (!hasCLIHistorySelector && hasConfigHistory &&
        (hasConfigCompare || hasConfigCompareHistory || hasConfigTrendHistory ||
         hasConfigAppendHistory)) {
      errs() << "circt-mut report: [report] key 'history' is mutually "
                "exclusive with 'compare', 'compare_history_latest', "
                "'trend_history', and 'append_history'\n";
      return 1;
    }
    if (!historyBootstrap) {
      if (auto it = cfg.report.find("history_bootstrap");
          it != cfg.report.end() && !it->second.empty()) {
        std::string lowered = StringRef(it->second).trim().lower();
        if (lowered == "1" || lowered == "true" || lowered == "yes" ||
            lowered == "on")
          historyBootstrap = true;
        else if (lowered == "0" || lowered == "false" || lowered == "no" ||
                 lowered == "off")
          historyBootstrap = false;
        else {
          errs() << "circt-mut report: invalid [report] key "
                    "'history_bootstrap' value '"
                 << it->second
                 << "' (expected 1|0|true|false|yes|no|on|off)\n";
          return 1;
        }
      }
    }
    if (!failOnPrequalifyDriftOverride.has_value()) {
      if (auto it = cfg.report.find("fail_on_prequalify_drift");
          it != cfg.report.end() && !it->second.empty()) {
        StringRef raw = StringRef(it->second).trim().lower();
        if (raw == "1" || raw == "true" || raw == "yes" || raw == "on")
          failOnPrequalifyDriftOverride = true;
        else if (raw == "0" || raw == "false" || raw == "no" || raw == "off")
          failOnPrequalifyDriftOverride = false;
        else {
          errs() << "circt-mut report: invalid [report] key "
                    "'fail_on_prequalify_drift' value '"
                 << it->second
                 << "' (expected 1|0|true|false|yes|no|on|off)\n";
          return 1;
        }
      }
    }
    auto configPolicyModeIt = cfg.report.find("policy_mode");
    bool hasConfigPolicyMode =
        configPolicyModeIt != cfg.report.end() && !configPolicyModeIt->second.empty();
    auto configPolicyStopOnFailIt = cfg.report.find("policy_stop_on_fail");
    bool hasConfigPolicyStopOnFail =
        configPolicyStopOnFailIt != cfg.report.end() &&
        !configPolicyStopOnFailIt->second.empty();
    auto configPolicyProfileIt = cfg.report.find("policy_profile");
    bool hasConfigPolicyProfile =
        configPolicyProfileIt != cfg.report.end() &&
        !configPolicyProfileIt->second.empty();
    auto configPolicyProfilesIt = cfg.report.find("policy_profiles");
    bool hasConfigPolicyProfiles =
        configPolicyProfilesIt != cfg.report.end() &&
        !configPolicyProfilesIt->second.empty();
    if (!hasCLIPolicyMode && !hasCLIPolicyProfile && hasConfigPolicyMode &&
        (hasConfigPolicyProfile || hasConfigPolicyProfiles)) {
      errs() << "circt-mut report: [report] keys 'policy_mode' and "
                "'policy_profile(s)' are mutually exclusive\n";
      return 1;
    }
    if (!hasCLIPolicyMode && hasConfigPolicyStopOnFail &&
        !hasConfigPolicyMode) {
      errs() << "circt-mut report: [report] key 'policy_stop_on_fail' "
                "requires 'policy_mode'\n";
      return 1;
    }
    if (effectiveOpts.coverWorkDir.empty()) {
      if (auto it = cfg.report.find("cover_work_dir");
          it != cfg.report.end() && !it->second.empty())
        effectiveOpts.coverWorkDir = it->second;
    }
    if (effectiveOpts.matrixOutDir.empty()) {
      if (auto it = cfg.report.find("matrix_out_dir");
          it != cfg.report.end() && !it->second.empty())
        effectiveOpts.matrixOutDir = it->second;
    }
    if (effectiveOpts.laneBudgetOutFile.empty()) {
      if (auto it = cfg.report.find("lane_budget_out");
          it != cfg.report.end() && !it->second.empty())
        effectiveOpts.laneBudgetOutFile = it->second;
    }
    if (effectiveOpts.skipBudgetOutFile.empty()) {
      if (auto it = cfg.report.find("skip_budget_out");
          it != cfg.report.end() && !it->second.empty())
        effectiveOpts.skipBudgetOutFile = it->second;
    }
    if (effectiveOpts.outFile.empty()) {
      if (auto it = cfg.report.find("out");
          it != cfg.report.end() && !it->second.empty())
        effectiveOpts.outFile = it->second;
    }
    if (externalFormalResultsFiles.empty()) {
      if (auto it = cfg.report.find("external_formal_results");
          it != cfg.report.end() && !it->second.empty()) {
        SmallVector<StringRef, 8> elems;
        StringRef(it->second).split(elems, ',', /*MaxSplit=*/-1,
                                    /*KeepEmpty=*/false);
        for (StringRef raw : elems) {
          StringRef token = raw.trim();
          if (!token.empty())
            externalFormalResultsFiles.push_back(token.str());
        }
      }
    }
    if (externalFormalOutDir.empty()) {
      if (auto it = cfg.report.find("external_formal_out_dir");
          it != cfg.report.end() && !it->second.empty())
        externalFormalOutDir = it->second;
    }
    auto appendRulesFromReportCSV =
        [&](StringRef key, StringRef optionName,
            SmallVectorImpl<DeltaGateRule> &dstRules) -> bool {
      if (!dstRules.empty())
        return true;
      auto it = cfg.report.find(key);
      if (it == cfg.report.end() || it->second.empty())
        return true;
      SmallVector<StringRef, 8> entries;
      StringRef(it->second).split(entries, ',', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);
      for (StringRef raw : entries) {
        StringRef token = raw.trim();
        if (token.empty())
          continue;
        DeltaGateRule rule;
        if (!parseDeltaGateRule(token, optionName, rule, error))
          return false;
        dstRules.push_back(rule);
      }
      return true;
    };
    if (!appendRulesFromReportCSV("fail_if_value_gt",
                                  "[report] key 'fail_if_value_gt'",
                                  effectiveOpts.failIfValueGtRules) ||
        !appendRulesFromReportCSV("fail_if_value_lt",
                                  "[report] key 'fail_if_value_lt'",
                                  effectiveOpts.failIfValueLtRules) ||
        !appendRulesFromReportCSV("fail_if_delta_gt",
                                  "[report] key 'fail_if_delta_gt'",
                                  effectiveOpts.failIfDeltaGtRules) ||
        !appendRulesFromReportCSV("fail_if_delta_lt",
                                  "[report] key 'fail_if_delta_lt'",
                                  effectiveOpts.failIfDeltaLtRules) ||
        !appendRulesFromReportCSV("fail_if_trend_delta_gt",
                                  "[report] key 'fail_if_trend_delta_gt'",
                                  effectiveOpts.failIfTrendDeltaGtRules) ||
        !appendRulesFromReportCSV("fail_if_trend_delta_lt",
                                  "[report] key 'fail_if_trend_delta_lt'",
                                  effectiveOpts.failIfTrendDeltaLtRules)) {
      errs() << error << "\n";
      return 1;
    }
    if (policyProfiles.empty() && !hasCLIPolicyMode && !hasCLIPolicyProfile) {
      auto parseProfileCSV = [&](StringRef csv) {
        SmallVector<StringRef, 8> elems;
        csv.split(elems, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (StringRef raw : elems) {
          StringRef p = raw.trim();
          if (!p.empty())
            policyProfiles.push_back(p.str());
        }
      };
      if (auto it = cfg.report.find("policy_profile");
          it != cfg.report.end() && !it->second.empty())
        parseProfileCSV(it->second);
      if (auto it = cfg.report.find("policy_profiles");
          it != cfg.report.end() && !it->second.empty())
        parseProfileCSV(it->second);
      if (!policyProfiles.empty())
        appliedPolicyProfileSource = "config";
    }
    if (policyProfiles.empty()) {
      std::string mode;
      std::optional<bool> stopOnFail;
      if (hasCLIPolicyMode) {
        mode = effectiveOpts.policyMode;
        stopOnFail = effectiveOpts.policyStopOnFail;
      } else {
        auto policyModeIt = cfg.report.find("policy_mode");
        auto policyStopOnFailIt = cfg.report.find("policy_stop_on_fail");
        bool hasPolicyMode =
            policyModeIt != cfg.report.end() && !policyModeIt->second.empty();
        bool hasPolicyStopOnFail = policyStopOnFailIt != cfg.report.end() &&
                                   !policyStopOnFailIt->second.empty();
        if (hasPolicyStopOnFail && !hasPolicyMode) {
          errs() << "circt-mut report: [report] key 'policy_stop_on_fail' "
                    "requires 'policy_mode'\n";
          return 1;
        }
        if (hasPolicyMode)
          mode = StringRef(policyModeIt->second).trim().lower();
        if (hasPolicyStopOnFail) {
          StringRef raw = StringRef(policyStopOnFailIt->second).trim().lower();
          if (raw == "1" || raw == "true" || raw == "yes" || raw == "on")
            stopOnFail = true;
          else if (raw == "0" || raw == "false" || raw == "no" ||
                   raw == "off")
            stopOnFail = false;
          else {
            errs() << "circt-mut report: invalid [report] key "
                      "'policy_stop_on_fail' value '"
                   << policyStopOnFailIt->second
                   << "' (expected 1|0|true|false|yes|no|on|off)\n";
            return 1;
          }
        }
      }
      if (!mode.empty()) {
        if (opts.mode != "matrix" && opts.mode != "all") {
          errs() << (hasCLIPolicyMode
                         ? "circt-mut report: --policy-mode requires --mode "
                           "matrix or --mode all\n"
                         : "circt-mut report: [report] key 'policy_mode' "
                           "requires --mode matrix or --mode all\n");
          return 1;
        }
        if (!hasCLIPolicyMode && !isMatrixPolicyMode(mode)) {
          errs() << "circt-mut report: invalid [report] key 'policy_mode' "
                    "value '"
                 << mode << "' (expected " << kMatrixPolicyModeList << ")\n";
          return 1;
        }
        std::string modeError;
        if (!appendMatrixPolicyModeProfiles(mode, stopOnFail.value_or(false),
                                            policyProfiles, modeError,
                                            hasCLIPolicyMode ? "circt-mut report:"
                                                             : "circt-mut report: [report] key 'policy_mode'")) {
          errs() << modeError << "\n";
          return 1;
        }
        appliedPolicyMode = mode;
        appliedPolicyModeSource = hasCLIPolicyMode ? "cli" : "config";
        bool requestedStopOnFail = stopOnFail.value_or(false);
        bool usesStopOnFail = matrixPolicyModeUsesStopOnFail(mode);
        appliedPolicyStopOnFail = requestedStopOnFail;
        appliedPolicyStopOnFailEffective =
            usesStopOnFail ? requestedStopOnFail : false;
        appliedPolicyStopOnFailIgnored = (!usesStopOnFail && requestedStopOnFail);
        appliedPolicyProfileSource = "mode";
      }
    }
  }
  if (policyProfiles.empty() && hasCLIPolicyMode) {
    if (opts.mode != "matrix" && opts.mode != "all") {
      errs() << "circt-mut report: --policy-mode requires --mode matrix or "
                "--mode all\n";
      return 1;
    }
    std::string modeError;
    if (!appendMatrixPolicyModeProfiles(
            effectiveOpts.policyMode, effectiveOpts.policyStopOnFail.value_or(false),
            policyProfiles, modeError, "circt-mut report:")) {
      errs() << modeError << "\n";
      return 1;
    }
    appliedPolicyMode = effectiveOpts.policyMode;
    appliedPolicyModeSource = "cli";
    bool requestedStopOnFail = effectiveOpts.policyStopOnFail.value_or(false);
    bool usesStopOnFail = matrixPolicyModeUsesStopOnFail(effectiveOpts.policyMode);
    appliedPolicyStopOnFail = requestedStopOnFail;
    appliedPolicyStopOnFailEffective =
        usesStopOnFail ? requestedStopOnFail : false;
    appliedPolicyStopOnFailIgnored = (!usesStopOnFail && requestedStopOnFail);
    appliedPolicyProfileSource = "mode";
  }
  if (!policyProfiles.empty()) {
    SmallVector<std::string, 4> uniqueProfiles;
    StringSet<> seen;
    std::string parseError;
    for (const auto &profile : policyProfiles) {
      if (!seen.insert(profile).second)
        continue;
      uniqueProfiles.push_back(profile);
      if (!applyPolicyProfile(profile, effectiveOpts, parseError)) {
        errs() << parseError << "\n";
        return 1;
      }
    }
    policyProfiles = std::move(uniqueProfiles);
  }
  if (failOnPrequalifyDriftOverride.has_value())
    effectiveOpts.failOnPrequalifyDrift = *failOnPrequalifyDriftOverride;
  if (!historyFile.empty()) {
    if (compareFile.empty() && compareHistoryLatestFile.empty())
      compareHistoryLatestFile = historyFile;
    if (trendHistoryFile.empty())
      trendHistoryFile = historyFile;
    if (appendHistoryFile.empty())
      appendHistoryFile = historyFile;
  }

  if (!effectiveOpts.coverWorkDir.empty())
    coverWorkDir = resolveRelativeTo(opts.projectDir, effectiveOpts.coverWorkDir);
  if (!effectiveOpts.matrixOutDir.empty())
    matrixOutDir = resolveRelativeTo(opts.projectDir, effectiveOpts.matrixOutDir);

  std::vector<std::pair<std::string, std::string>> rows;
  rows.emplace_back("report.mode", opts.mode);
  rows.emplace_back("policy.mode",
                    appliedPolicyMode.empty() ? std::string("-")
                                              : appliedPolicyMode);
  rows.emplace_back("policy.mode_is_set", appliedPolicyMode.empty() ? "0" : "1");
  rows.emplace_back("policy.mode_is_native_family",
                    (!appliedPolicyMode.empty() &&
                     StringRef(appliedPolicyMode).starts_with("native-"))
                        ? "1"
                        : "0");
  rows.emplace_back("policy.mode_is_native_strict",
                    (appliedPolicyMode == "native-strict" ||
                     appliedPolicyMode == "native-strict-formal" ||
                     appliedPolicyMode == "native-strict-formal-summary")
                        ? "1"
                        : "0");
  rows.emplace_back("policy.mode_source", appliedPolicyModeSource);
  rows.emplace_back("policy.stop_on_fail",
                    appliedPolicyStopOnFail.has_value()
                        ? (*appliedPolicyStopOnFail ? std::string("1")
                                                    : std::string("0"))
                        : std::string("-"));
  rows.emplace_back("policy.stop_on_fail_effective",
                    appliedPolicyStopOnFailEffective.has_value()
                        ? (*appliedPolicyStopOnFailEffective ? std::string("1")
                                                             : std::string("0"))
                        : std::string("-"));
  rows.emplace_back("policy.stop_on_fail_ignored",
                    appliedPolicyStopOnFailIgnored.has_value()
                        ? (*appliedPolicyStopOnFailIgnored ? std::string("1")
                                                           : std::string("0"))
                        : std::string("-"));
  rows.emplace_back("policy.profile_source", appliedPolicyProfileSource);
  std::string policyProfileResolvedCSV = "-";
  if (!policyProfiles.empty()) {
    policyProfileResolvedCSV.clear();
    for (size_t i = 0; i < policyProfiles.size(); ++i) {
      if (i)
        policyProfileResolvedCSV.append(",");
      policyProfileResolvedCSV.append(policyProfiles[i]);
    }
  }
  rows.emplace_back("policy.profile_resolved_csv", policyProfileResolvedCSV);
  if (!policyProfiles.empty()) {
    rows.emplace_back("policy.profile_count",
                      std::to_string(policyProfiles.size()));
    for (size_t i = 0; i < policyProfiles.size(); ++i)
      rows.emplace_back((Twine("policy.profile_") + Twine(i + 1)).str(),
                        policyProfiles[i]);
  }
  std::string discoveryError;
  SmallVector<std::string, 4> discoveredExternalFormalResultsFiles;
  std::string resolvedExternalFormalOutDir;
  if (!externalFormalOutDir.empty()) {
    resolvedExternalFormalOutDir =
        resolveRelativeTo(opts.projectDir, externalFormalOutDir);
    if (!discoverExternalFormalResultsFromOutDir(resolvedExternalFormalOutDir,
                                                 discoveredExternalFormalResultsFiles,
                                                 discoveryError)) {
      errs() << discoveryError << "\n";
      return 1;
    }
  }
  SmallVector<std::string, 4> resolvedExternalFormalResultsFiles;
  for (const auto &path : externalFormalResultsFiles) {
    if (path.empty())
      continue;
    std::string resolved = resolveRelativeTo(opts.projectDir, path);
    if (llvm::is_contained(resolvedExternalFormalResultsFiles, resolved))
      continue;
    resolvedExternalFormalResultsFiles.push_back(std::move(resolved));
  }
  for (const auto &path : discoveredExternalFormalResultsFiles) {
    if (path.empty())
      continue;
    if (llvm::is_contained(resolvedExternalFormalResultsFiles, path))
      continue;
    resolvedExternalFormalResultsFiles.push_back(path);
  }
  rows.emplace_back("external_formal.out_dir",
                    resolvedExternalFormalOutDir.empty()
                        ? std::string("-")
                        : resolvedExternalFormalOutDir);
  rows.emplace_back("external_formal.files_discovered",
                    std::to_string(discoveredExternalFormalResultsFiles.size()));
  for (size_t i = 0; i < discoveredExternalFormalResultsFiles.size(); ++i) {
    rows.emplace_back((Twine("external_formal.discovered_file_") + Twine(i + 1)).str(),
                      discoveredExternalFormalResultsFiles[i]);
  }
  rows.emplace_back("external_formal.files_configured",
                    std::to_string(resolvedExternalFormalResultsFiles.size()));
  for (size_t i = 0; i < resolvedExternalFormalResultsFiles.size(); ++i) {
    rows.emplace_back((Twine("external_formal.file_") + Twine(i + 1)).str(),
                      resolvedExternalFormalResultsFiles[i]);
  }
  int finalRC = 0;
  std::string error;
  uint64_t prequalifyDriftNonZero = 0;
  bool prequalifyDriftComparable = false;
  uint64_t prequalifyDriftLaneRowsMismatch = 0;
  uint64_t prequalifyDriftLaneRowsMissingInResults = 0;
  uint64_t prequalifyDriftLaneRowsMissingInNative = 0;
  std::vector<MatrixLaneBudgetRow> laneBudgetRows;
  if (!effectiveOpts.laneBudgetOutFile.empty() &&
      !(opts.mode == "matrix" || opts.mode == "all")) {
    errs() << "circt-mut report: --lane-budget-out requires --mode matrix or "
              "--mode all\n";
    return 1;
  }
  if (!effectiveOpts.skipBudgetOutFile.empty() &&
      !(opts.mode == "matrix" || opts.mode == "all")) {
    errs() << "circt-mut report: --skip-budget-out requires --mode matrix or "
              "--mode all\n";
    return 1;
  }
  if (opts.mode == "cover" || opts.mode == "all") {
    if (!collectCoverReport(coverWorkDir, rows, error)) {
      errs() << error << "\n";
      return 1;
    }
  }
  if (opts.mode == "matrix" || opts.mode == "all") {
    if (!collectMatrixReport(matrixOutDir, rows, error, &prequalifyDriftNonZero,
                             &prequalifyDriftComparable,
                             &prequalifyDriftLaneRowsMismatch,
                             &prequalifyDriftLaneRowsMissingInResults,
                             &prequalifyDriftLaneRowsMissingInNative,
                             &laneBudgetRows)) {
      errs() << error << "\n";
      return 1;
    }
    SkipBudgetSummary skipSummary = computeSkipBudgetSummary(laneBudgetRows);
    rows.emplace_back("matrix.skip_budget_rows_total",
                      std::to_string(skipSummary.totalRows));
    rows.emplace_back("matrix.skip_budget_rows_skip",
                      std::to_string(skipSummary.skipRows));
    rows.emplace_back("matrix.skip_budget_rows_non_skip",
                      std::to_string(skipSummary.nonSkipRows));
    rows.emplace_back("matrix.skip_budget_rows_stop_on_fail",
                      std::to_string(skipSummary.stopOnFailRows));
    rows.emplace_back("matrix.skip_budget_rows_non_stop_on_fail",
                      std::to_string(skipSummary.nonStopOnFailSkipRows));
    rows.emplace_back("matrix.skip_budget_rows_with_reason",
                      std::to_string(skipSummary.rowsWithReason));
  }
  if (!resolvedExternalFormalResultsFiles.empty()) {
    if (!collectExternalFormalSummary(resolvedExternalFormalResultsFiles, rows,
                                      error)) {
      errs() << error << "\n";
      return 1;
    }
  } else {
    rows.emplace_back("external_formal.files", "0");
    rows.emplace_back("external_formal.lines", "0");
    rows.emplace_back("external_formal.parsed_status_lines", "0");
    rows.emplace_back("external_formal.parsed_summary_lines", "0");
    rows.emplace_back("external_formal.unparsed_lines", "0");
    rows.emplace_back("external_formal.pass", "0");
    rows.emplace_back("external_formal.fail", "0");
    rows.emplace_back("external_formal.error", "0");
    rows.emplace_back("external_formal.skip", "0");
    rows.emplace_back("external_formal.xfail", "0");
    rows.emplace_back("external_formal.xpass", "0");
    rows.emplace_back("external_formal.summary_total", "0");
    rows.emplace_back("external_formal.summary_pass", "0");
    rows.emplace_back("external_formal.summary_fail", "0");
    rows.emplace_back("external_formal.summary_error", "0");
    rows.emplace_back("external_formal.summary_skip", "0");
    rows.emplace_back("external_formal.summary_xfail", "0");
    rows.emplace_back("external_formal.summary_xpass", "0");
    rows.emplace_back("external_formal.summary_tsv_files", "0");
    rows.emplace_back("external_formal.summary_tsv_rows", "0");
    rows.emplace_back("external_formal.summary_tsv_schema_valid_files", "0");
    rows.emplace_back("external_formal.summary_tsv_schema_invalid_files", "0");
    rows.emplace_back("external_formal.summary_tsv_parse_errors", "0");
    rows.emplace_back("external_formal.summary_tsv_consistent_rows", "0");
    rows.emplace_back("external_formal.summary_tsv_inconsistent_rows", "0");
    rows.emplace_back("external_formal.summary_tsv_schema_version_rows", "0");
    rows.emplace_back("external_formal.summary_tsv_schema_version_invalid_rows",
                      "0");
    rows.emplace_back("external_formal.summary_tsv_schema_version_min", "0");
    rows.emplace_back("external_formal.summary_tsv_schema_version_max", "0");
    rows.emplace_back("external_formal.summary_tsv_duplicate_rows", "0");
    rows.emplace_back("external_formal.summary_tsv_unique_rows", "0");
    rows.emplace_back("external_formal.fail_like_sum", "0");
  }
  if (!effectiveOpts.laneBudgetOutFile.empty()) {
    std::string laneBudgetOut =
        resolveRelativeTo(opts.projectDir, effectiveOpts.laneBudgetOutFile);
    if (!writeLaneBudgetFile(laneBudgetOut, laneBudgetRows, error)) {
      errs() << error << "\n";
      return 1;
    }
    rows.emplace_back("matrix.lane_budget_file", laneBudgetOut);
    rows.emplace_back("matrix.lane_budget_file_rows",
                      std::to_string(laneBudgetRows.size()));
  }
  if (!effectiveOpts.skipBudgetOutFile.empty()) {
    std::string skipBudgetOut =
        resolveRelativeTo(opts.projectDir, effectiveOpts.skipBudgetOutFile);
    if (!writeSkipBudgetFile(skipBudgetOut, laneBudgetRows, error)) {
      errs() << error << "\n";
      return 1;
    }
    SkipBudgetSummary skipSummary = computeSkipBudgetSummary(laneBudgetRows);
    rows.emplace_back("matrix.skip_budget_file", skipBudgetOut);
    rows.emplace_back("matrix.skip_budget_file_rows",
                      std::to_string(laneBudgetRows.size()));
    rows.emplace_back("matrix.skip_budget_rows_skip",
                      std::to_string(skipSummary.skipRows));
    rows.emplace_back("matrix.skip_budget_rows_non_skip",
                      std::to_string(skipSummary.nonSkipRows));
    rows.emplace_back("matrix.skip_budget_rows_stop_on_fail",
                      std::to_string(skipSummary.stopOnFailRows));
    rows.emplace_back("matrix.skip_budget_rows_non_stop_on_fail",
                      std::to_string(skipSummary.nonStopOnFailSkipRows));
    rows.emplace_back("matrix.skip_budget_rows_with_reason",
                      std::to_string(skipSummary.rowsWithReason));
  }

  if (compareFile.empty() && compareHistoryLatestFile.empty() &&
      (!effectiveOpts.failIfDeltaGtRules.empty() ||
       !effectiveOpts.failIfDeltaLtRules.empty())) {
    errs() << "circt-mut report: --fail-if-delta-gt/--fail-if-delta-lt "
              "require --compare or --compare-history-latest\n";
    return 1;
  }
  if (trendHistoryFile.empty() &&
      (!effectiveOpts.failIfTrendDeltaGtRules.empty() ||
       !effectiveOpts.failIfTrendDeltaLtRules.empty())) {
    errs() << "circt-mut report: --fail-if-trend-delta-gt/"
              "--fail-if-trend-delta-lt require --trend-history\n";
    return 1;
  }

  StringMap<double> numericDeltas;
  StringMap<double> trendDeltas;
  bool skipCompareGatesForBootstrap = false;
  bool skipTrendGatesForBootstrap = false;
  bool historyBootstrapActivated = false;
  if (!compareFile.empty()) {
    std::string baselinePath = resolveRelativeTo(opts.projectDir, compareFile);
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
  if (!compareHistoryLatestFile.empty()) {
    std::string historyPath =
        resolveRelativeTo(opts.projectDir, compareHistoryLatestFile);
    if (!sys::fs::exists(historyPath)) {
      if (historyBootstrap) {
        historyBootstrapActivated = true;
        skipCompareGatesForBootstrap = true;
        rows.emplace_back("history.bootstrap", "1");
        rows.emplace_back("history.bootstrap.compare", "1");
        rows.emplace_back("history.bootstrap.compare_file", historyPath);
      } else {
        errs() << "circt-mut report: compare history file not found: "
               << historyPath << "\n";
        return 1;
      }
    }
    if (!skipCompareGatesForBootstrap) {
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
  }
  if (!trendHistoryFile.empty()) {
    std::string historyPath = resolveRelativeTo(opts.projectDir, trendHistoryFile);
    if (!sys::fs::exists(historyPath)) {
      if (historyBootstrap) {
        historyBootstrapActivated = true;
        skipTrendGatesForBootstrap = true;
        rows.emplace_back("history.bootstrap", "1");
        rows.emplace_back("history.bootstrap.trend", "1");
        rows.emplace_back("history.bootstrap.trend_file", historyPath);
      } else {
        errs() << "circt-mut report: trend history file not found: "
               << historyPath << "\n";
        return 1;
      }
    }
    if (!skipTrendGatesForBootstrap) {
      std::vector<HistorySnapshot> snapshots;
      if (!loadHistorySnapshots(historyPath, snapshots, error)) {
        errs() << error << "\n";
        return 1;
      }
      auto trendBaseRows = rows;
      appendTrendRows(trendBaseRows, snapshots, historyPath, trendWindowRuns,
                      rows, trendDeltas);
    }
  }

  if (!effectiveOpts.failIfDeltaGtRules.empty() ||
      !effectiveOpts.failIfDeltaLtRules.empty()) {
    if (skipCompareGatesForBootstrap) {
      rows.emplace_back("compare.gate_status", "bootstrap_skipped");
      rows.emplace_back("compare.gate_reason", "missing_history_bootstrap");
    } else {
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

    for (const auto &rule : effectiveOpts.failIfDeltaGtRules)
      if (!evaluateRule(rule, /*isUpperBound=*/true))
        return 1;
    for (const auto &rule : effectiveOpts.failIfDeltaLtRules)
      if (!evaluateRule(rule, /*isUpperBound=*/false))
        return 1;

    rows.emplace_back("compare.gate_rules_total",
                      std::to_string(effectiveOpts.failIfDeltaGtRules.size() +
                                     effectiveOpts.failIfDeltaLtRules.size()));
    rows.emplace_back("compare.gate_failure_count",
                      std::to_string(failures.size()));
    rows.emplace_back("compare.gate_status", failures.empty() ? "pass" : "fail");
    for (size_t i = 0; i < failures.size(); ++i)
      rows.emplace_back((Twine("compare.gate_failure_") + Twine(i + 1)).str(),
                        failures[i]);
    if (!failures.empty())
      finalRC = 2;
    }
  }
  if (!effectiveOpts.failIfValueGtRules.empty() ||
      !effectiveOpts.failIfValueLtRules.empty()) {
    SmallVector<std::string, 8> failures;
    StringMap<std::string> currentValues;
    for (const auto &row : rows)
      currentValues[row.first] = row.second;
    auto evaluateValueRule = [&](const DeltaGateRule &rule,
                                 bool isUpperBound) -> bool {
      auto it = currentValues.find(rule.key);
      if (it == currentValues.end()) {
        errs() << "circt-mut report: current numeric key missing for value gate "
                  "rule key '"
               << rule.key << "'\n";
        return false;
      }
      auto parsed = parseOptionalDouble(it->second);
      if (!parsed) {
        errs() << "circt-mut report: current value for gate rule key '" << rule.key
               << "' is not numeric: '" << it->second << "'\n";
        return false;
      }
      double value = *parsed;
      if ((isUpperBound && value > rule.threshold) ||
          (!isUpperBound && value < rule.threshold)) {
        failures.push_back((Twine(rule.key) + " value=" + formatDouble2(value) +
                            (isUpperBound ? " > " : " < ") +
                            formatDouble2(rule.threshold))
                               .str());
      }
      return true;
    };

    for (const auto &rule : effectiveOpts.failIfValueGtRules)
      if (!evaluateValueRule(rule, /*isUpperBound=*/true))
        return 1;
    for (const auto &rule : effectiveOpts.failIfValueLtRules)
      if (!evaluateValueRule(rule, /*isUpperBound=*/false))
        return 1;

    rows.emplace_back("value_gate.rules_total",
                      std::to_string(effectiveOpts.failIfValueGtRules.size() +
                                     effectiveOpts.failIfValueLtRules.size()));
    rows.emplace_back("value_gate.failure_count",
                      std::to_string(failures.size()));
    rows.emplace_back("value_gate.status", failures.empty() ? "pass" : "fail");
    for (size_t i = 0; i < failures.size(); ++i)
      rows.emplace_back((Twine("value_gate.failure_") + Twine(i + 1)).str(),
                        failures[i]);
    if (!failures.empty())
      finalRC = std::max(finalRC, 2);
  }
  if (!effectiveOpts.failIfTrendDeltaGtRules.empty() ||
      !effectiveOpts.failIfTrendDeltaLtRules.empty()) {
    if (skipTrendGatesForBootstrap) {
      rows.emplace_back("trend.gate_status", "bootstrap_skipped");
      rows.emplace_back("trend.gate_reason", "missing_history_bootstrap");
    } else {
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

    for (const auto &rule : effectiveOpts.failIfTrendDeltaGtRules)
      if (!evaluateTrendRule(rule, /*isUpperBound=*/true))
        return 1;
    for (const auto &rule : effectiveOpts.failIfTrendDeltaLtRules)
      if (!evaluateTrendRule(rule, /*isUpperBound=*/false))
        return 1;

    rows.emplace_back("trend.gate_rules_total",
                      std::to_string(
                          effectiveOpts.failIfTrendDeltaGtRules.size() +
                          effectiveOpts.failIfTrendDeltaLtRules.size()));
    rows.emplace_back("trend.gate_failure_count",
                      std::to_string(failures.size()));
    rows.emplace_back("trend.gate_status", failures.empty() ? "pass" : "fail");
    for (size_t i = 0; i < failures.size(); ++i)
      rows.emplace_back((Twine("trend.gate_failure_") + Twine(i + 1)).str(),
                        failures[i]);
    if (!failures.empty())
      finalRC = 2;
    }
  }
  if (effectiveOpts.failOnPrequalifyDrift) {
    if (!(opts.mode == "matrix" || opts.mode == "all")) {
      errs() << "circt-mut report: --fail-on-prequalify-drift requires "
                "--mode matrix or --mode all\n";
      return 1;
    }
    rows.emplace_back("matrix.prequalify_drift_gate_enabled", "1");
    rows.emplace_back("matrix.prequalify_drift_gate_comparable",
                      prequalifyDriftComparable ? "1" : "0");
    rows.emplace_back("matrix.prequalify_drift_gate_lane_rows_mismatch",
                      std::to_string(prequalifyDriftLaneRowsMismatch));
    rows.emplace_back("matrix.prequalify_drift_gate_lane_rows_missing_in_results",
                      std::to_string(prequalifyDriftLaneRowsMissingInResults));
    rows.emplace_back("matrix.prequalify_drift_gate_lane_rows_missing_in_native",
                      std::to_string(prequalifyDriftLaneRowsMissingInNative));
    if (!prequalifyDriftComparable) {
      rows.emplace_back("matrix.prequalify_drift_gate_status", "error");
      rows.emplace_back("matrix.prequalify_drift_gate_reason",
                        "matrix results prequalify columns or native summary "
                        "artifact missing");
      finalRC = std::max(finalRC, 2);
    } else if (prequalifyDriftNonZero == 0 &&
               prequalifyDriftLaneRowsMismatch == 0 &&
               prequalifyDriftLaneRowsMissingInResults == 0 &&
               prequalifyDriftLaneRowsMissingInNative == 0) {
      rows.emplace_back("matrix.prequalify_drift_gate_status", "pass");
      rows.emplace_back("matrix.prequalify_drift_gate_reason", "-");
    } else {
      rows.emplace_back("matrix.prequalify_drift_gate_status", "fail");
      rows.emplace_back("matrix.prequalify_drift_gate_reason",
                        (Twine("nonzero_drift_metrics=") +
                         Twine(prequalifyDriftNonZero) + ",lane_rows_mismatch=" +
                         Twine(prequalifyDriftLaneRowsMismatch) +
                         ",lane_rows_missing_in_results=" +
                         Twine(prequalifyDriftLaneRowsMissingInResults) +
                         ",lane_rows_missing_in_native=" +
                         Twine(prequalifyDriftLaneRowsMissingInNative))
                            .str());
      finalRC = std::max(finalRC, 2);
    }
  }

  if (historyMaxRuns > 0 && appendHistoryFile.empty()) {
    errs() << "circt-mut report: --history-max-runs requires --append-history "
              "or --history\n";
    return 1;
  }

  if (!appendHistoryFile.empty()) {
    std::string historyPath =
        resolveRelativeTo(opts.projectDir, appendHistoryFile);
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
    if (historyMaxRuns > 0) {
      uint64_t prunedRuns = 0;
      uint64_t prunedRows = 0;
      if (!pruneHistoryToMaxRuns(historyPath, historyMaxRuns, prunedRuns,
                                 prunedRows, error)) {
        errs() << error << "\n";
        return 1;
      }
      rows.emplace_back("history.max_runs", std::to_string(historyMaxRuns));
      rows.emplace_back("history.pruned_runs", std::to_string(prunedRuns));
      rows.emplace_back("history.pruned_rows", std::to_string(prunedRows));
    }
  }
  if (historyBootstrapActivated)
    rows.emplace_back("history.bootstrap_active", "1");

  std::string outFile;
  if (!effectiveOpts.outFile.empty())
    outFile = resolveRelativeTo(opts.projectDir, effectiveOpts.outFile);
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
  for (const std::string &modeEntry : opts.modeList) {
    StringRef mode = StringRef(modeEntry).trim();
    if (mode.empty())
      continue;
    if (!isKnownMutationMode(mode)) {
      errs() << "circt-mut generate: unknown --mode value: " << mode
             << " (expected inv|const0|const1|cnot0|cnot1|"
                "arith|control|balanced|all|stuck|invert|connect)\n";
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
    if (!isKnownMutationMode(modeName)) {
      errs() << "circt-mut generate: unknown --mode-count mode: " << modeName
             << " (expected inv|const0|const1|cnot0|cnot1|"
                "arith|control|balanced|all|stuck|invert|connect)\n";
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
    if (!isKnownMutationMode(modeName)) {
      errs() << "circt-mut generate: unknown --mode-weight mode: " << modeName
             << " (expected inv|const0|const1|cnot0|cnot1|"
                "arith|control|balanced|all|stuck|invert|connect)\n";
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
