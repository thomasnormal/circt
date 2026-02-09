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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

static void printHelp(raw_ostream &os) {
  os << "circt-mut - CIRCT mutation workflow frontend\n\n";
  os << "Usage:\n";
  os << "  circt-mut <subcommand> [args...]\n\n";
  os << "Subcommands:\n";
  os << "  cover     Run mutation coverage flow (run_mutation_cover.sh)\n";
  os << "  matrix    Run mutation lane matrix flow (run_mutation_matrix.sh)\n";
  os << "  generate  Generate mutation lists (native path; script fallback)\n\n";
  os << "Environment:\n";
  os << "  CIRCT_MUT_SCRIPTS_DIR  Override script directory location\n\n";
  os << "Examples:\n";
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
  os << "  --modes CSV               Comma-separated mutate modes\n";
  os << "  --mode-count NAME=COUNT   Explicit mutation count for a mode (repeatable)\n";
  os << "  --mode-counts CSV         Comma-separated NAME=COUNT mode allocations\n";
  os << "  --profile NAME            Named mutation profile (repeatable)\n";
  os << "  --profiles CSV            Comma-separated named mutation profiles\n";
  os << "  --cfg KEY=VALUE           Mutate config entry (repeatable)\n";
  os << "  --cfgs CSV                Comma-separated KEY=VALUE config entries\n";
  os << "  --select EXPR             Additional mutate select expression (repeatable)\n";
  os << "  --selects CSV             Comma-separated mutate select expressions\n";
  os << "  --cache-dir DIR           Use script backend (native cache migration pending)\n";
  os << "  -h, --help                Show help\n\n";
  os << "Output format:\n";
  os << "  Each line in --out is \"<id> <mutation-spec>\" (MCY-compatible).\n";
}

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
  std::string top;
  uint64_t count = 1000;
  uint64_t seed = 1;
  std::string yosys = "yosys";
  SmallVector<std::string, 8> modeList;
  SmallVector<std::string, 8> modeCountList;
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

    // Keep full compatibility by deferring unsupported/unknown options to the
    // script backend while native migration is in progress.
    if (arg == "--cache-dir") {
      auto v = requireValue(arg);
      if (!v)
        return result;
      (void)v;
      result.fallbackToScript = true;
      result.ok = true;
      return result;
    }
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
  if (profileName == "cover") {
    profileCfgs.push_back("weight_cover=5");
    profileCfgs.push_back("pick_cover_prcnt=80");
    return true;
  }
  if (profileName == "none")
    return true;

  error = (Twine("circt-mut generate: unknown --profile value: ") + profileName +
           " (expected arith-depth|control-depth|balanced-depth|cover|none)")
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

static int runNativeGenerate(const GenerateOptions &opts) {
  if (!sys::fs::exists(opts.design)) {
    errs() << "circt-mut generate: design file not found: " << opts.design << "\n";
    return 1;
  }

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
  if (!modeCountsEnabled) {
    baseCount = opts.count / modeCount;
    extraCount = opts.count % modeCount;
  }

  for (size_t i = 0; i < finalModes.size(); ++i) {
    uint64_t listCount = 0;
    StringRef mode = finalModes[i];
    if (modeCountsEnabled) {
      auto it = modeCountByMode.find(mode);
      if (it != modeCountByMode.end())
        listCount = it->second;
    } else {
      listCount = baseCount + (i < extraCount ? 1 : 0);
    }
    if (listCount == 0)
      continue;

    SmallVector<std::string, 8> familyTargets;
    modeFamilyTargets(mode, familyTargets);
    uint64_t familyCount = familyTargets.size();
    uint64_t familyBase = listCount / familyCount;
    uint64_t familyExtra = listCount % familyCount;
    for (size_t j = 0; j < familyTargets.size(); ++j) {
      uint64_t familyListCount = familyBase + (j < familyExtra ? 1 : 0);
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

  auto start = std::chrono::steady_clock::now();

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
    yosysArgs.push_back(opts.yosys);
    yosysArgs.push_back("-ql");
    yosysArgs.push_back(logPath);
    yosysArgs.push_back(scriptPath);

    std::string errMsg;
    int rc = sys::ExecuteAndWait(opts.yosys, yosysArgs, /*Env=*/std::nullopt,
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

    SmallVector<uint64_t, 16> topupCounts;
    topupCounts.reserve(targetCount);
    for (uint64_t i = 0; i < targetCount; ++i)
      topupCounts.push_back(topupBase + (i < topupExtra ? 1 : 0));

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

  auto end = std::chrono::steady_clock::now();
  uint64_t runtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end - start)
                           .count();

  outs() << "Generated mutations: " << (nextID - 1) << "\n";
  outs() << "Mutation file: " << opts.outFile << "\n";
  outs() << "Mutation generation runtime_ns: " << runtimeNs << "\n";
  outs() << "Mutation cache saved_runtime_ns: 0\n";
  outs() << "Mutation cache lock_wait_ns: 0\n";
  outs() << "Mutation cache lock_contended: 0\n";
  outs() << "Mutation cache status: disabled\n";

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
