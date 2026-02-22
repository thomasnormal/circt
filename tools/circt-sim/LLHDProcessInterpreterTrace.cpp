//===- LLHDProcessInterpreterTrace.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains LLHDProcessInterpreter tracing/diagnostics helpers
// extracted from LLHDProcessInterpreter.cpp.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>

using namespace mlir;
using namespace circt;
using namespace circt::sim;

static size_t countRegionOps(mlir::Region &region) {
  llvm::SmallVector<mlir::Region *, 16> regionWorklist;
  regionWorklist.push_back(&region);
  size_t count = 0;

  while (!regionWorklist.empty()) {
    mlir::Region *curRegion = regionWorklist.pop_back_val();
    for (mlir::Block &block : *curRegion) {
      for (mlir::Operation &op : block) {
        ++count;
        for (mlir::Region &nested : op.getRegions())
          regionWorklist.push_back(&nested);
      }
    }
  }

  return count;
}

static bool isDisableForkTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_DISABLE_FORK");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isWaitEventCacheTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_WAIT_EVENT_CACHE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isWaitEventNoopTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_WAIT_EVENT_NOOP");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isInterfaceSensitivityTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_IFACE_SENS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isMultiDriverTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_MULTI_DRV");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isModuleDriveTraceEnabled() {
  static bool enabled = std::getenv("CIRCT_SIM_TRACE_MOD_DRV") != nullptr;
  return enabled;
}

static bool isInterfacePropagationTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_IFACE_PROP");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isInterfaceTriStateTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_INTERFACE_TRISTATE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isCondBranchTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_CONDBR");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isFuncCacheTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_FUNC_CACHE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isInstanceOutputTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_INST_OUT");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isInstanceOutputUpdateTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_INST_OUT_UPDATE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isCombTraceThroughEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_COMB_TT");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isContinuousFallbackTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_CONT_FALLBACK");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isDriveScheduleTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_DRIVE_SCHEDULE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isDriveFailureTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_DRIVE_FAILURE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isArrayDriveTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_ARRAY_DRIVE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isI3CAddressBitTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_ADDR_BITS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isI3CRefCastTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_REF_CASTS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isI3CCastLayoutTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_CAST_LAYOUT");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isRefArgResolveTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_REF_ARG_RESOLVE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool isI3CFieldDriveTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_FIELD_DRIVES");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

static bool shouldTraceI3CFieldDriveName(llvm::StringRef fieldName) {
  return fieldName == "targetAddress" || fieldName == "targetAddressStatus" ||
         fieldName == "operation" || fieldName == "writeData" ||
         fieldName == "readData" || fieldName == "writeDataStatus" ||
         fieldName == "readDataStatus" ||
         fieldName == "no_of_i3c_bits_transfer";
}

static bool isI3CTransferStructType(Type type) {
  auto structType = dyn_cast<hw::StructType>(type);
  if (!structType)
    return false;
  return structType.getFieldIndex("targetAddress").has_value() &&
         structType.getFieldIndex("operation").has_value() &&
         structType.getFieldIndex("targetAddressStatus").has_value() &&
         structType.getFieldIndex("no_of_i3c_bits_transfer").has_value();
}

static bool isFirRegTraceEnabled() {
  static bool enabled = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_FIRREG");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

void LLHDProcessInterpreter::maybeTraceFilteredCall(
    ProcessId procId, llvm::StringRef callKind, llvm::StringRef calleeName,
    int64_t nowFs, uint64_t deltaStep) {
  static bool enabled = []() {
    if (const char *env = std::getenv("CIRCT_SIM_TRACE_CALL_FILTER"))
      return env[0] != '\0';
    return false;
  }();
  if (!enabled)
    return;

  static bool traceAll = []() {
    if (const char *env = std::getenv("CIRCT_SIM_TRACE_CALL_FILTER"))
      return llvm::StringRef(env).trim() == "1";
    return false;
  }();

  static llvm::SmallVector<std::string, 8> filters = []() {
    llvm::SmallVector<std::string, 8> parsed;
    const char *env = std::getenv("CIRCT_SIM_TRACE_CALL_FILTER");
    if (!env)
      return parsed;
    llvm::StringRef raw(env);
    if (raw.trim() == "1")
      return parsed;
    llvm::SmallVector<llvm::StringRef, 8> pieces;
    raw.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef piece : pieces) {
      llvm::StringRef trimmed = piece.trim();
      if (!trimmed.empty())
        parsed.push_back(trimmed.str());
    }
    return parsed;
  }();

  if (!traceAll) {
    bool matched = false;
    for (const std::string &filter : filters) {
      if (calleeName.contains(filter)) {
        matched = true;
        break;
      }
    }
    if (!matched)
      return;
  }

  llvm::errs() << "[CALL-TRACE] " << callKind << " proc=" << procId;
  if (nowFs >= 0)
    llvm::errs() << " t=" << nowFs << " d=" << deltaStep;
  llvm::errs() << " callee=" << calleeName << "\n";
}

void LLHDProcessInterpreter::maybeTraceCallIndirectSiteCacheHit(
    int64_t methodIndex) const {
  if (!traceCallIndirectSiteCacheEnabled)
    return;
  llvm::errs() << "[CI-SITE-CACHE] hit method_index=" << methodIndex << "\n";
}

void LLHDProcessInterpreter::maybeTraceCallIndirectSiteCacheStore(
    bool hasStaticMethodIndex, int64_t methodIndex) const {
  if (!traceCallIndirectSiteCacheEnabled)
    return;
  if (hasStaticMethodIndex)
    llvm::errs() << "[CI-SITE-CACHE] store method_index=" << methodIndex
                 << "\n";
  else
    llvm::errs() << "[CI-SITE-CACHE] store method_index=dynamic\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceSensitivityBegin(
    ProcessId procId, llvm::StringRef processName, size_t sourceCount) const {
  if (!isInterfaceSensitivityTraceEnabled())
    return;
  llvm::errs() << "[IFACE-SENS] proc=" << procId
               << " name='" << processName << "' src=" << sourceCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceSensitivitySource(
    SignalId sourceSignalId, size_t fieldCount) const {
  if (!isInterfaceSensitivityTraceEnabled())
    return;
  llvm::errs() << "[IFACE-SENS]   srcSig=" << sourceSignalId
               << " fields=" << fieldCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceSensitivityAddedField(
    SignalId fieldSignalId) const {
  if (!isInterfaceSensitivityTraceEnabled())
    return;
  llvm::StringRef name = "<unknown>";
  auto it = signalIdToName.find(fieldSignalId);
  if (it != signalIdToName.end())
    name = it->second;
  llvm::errs() << "[IFACE-SENS]     + fieldSig=" << fieldSignalId
               << " (" << name << ")\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceFieldShadowScanSummary(
    size_t topLevelSignalCount, size_t instanceSignalMapCount,
    size_t childCopyPairCount) const {
  llvm::dbgs() << "createInterfaceFieldShadowSignals: valueToSignal="
               << topLevelSignalCount
               << " instanceValueToSignal=" << instanceSignalMapCount
               << " childModuleCopyPairs=" << childCopyPairCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceParentGepStructName(
    llvm::StringRef structName) const {
  llvm::dbgs() << "parent GEP struct name: '" << structName << "'\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceParentScanResult(
    SignalId signalId, unsigned userCount, unsigned probeCount,
    unsigned gepCount, bool foundInterfaceStruct,
    uint64_t mallocAddress) const {
  llvm::dbgs() << "parent sig " << signalId
               << ": users=" << userCount
               << " probes=" << probeCount << " geps=" << gepCount
               << " ifaceFound=" << (foundInterfaceStruct ? "yes" : "no")
               << " addr=0x" << llvm::format_hex(mallocAddress, 10) << "\n";
}

void LLHDProcessInterpreter::maybeTraceChildInstanceFound(
    llvm::StringRef instanceName, llvm::StringRef moduleName) const {
  llvm::dbgs() << "  Found instance '" << instanceName
               << "' of module '" << moduleName << "'\n";
}

void LLHDProcessInterpreter::maybeTraceChildInstanceMissingRootModule() const {
  llvm::dbgs() << "    Warning: No root module for symbol lookup\n";
}

void LLHDProcessInterpreter::maybeTraceChildInstanceMissingModule(
    llvm::StringRef moduleName) const {
  llvm::dbgs() << "    Warning: Could not find module '" << moduleName
               << "'\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredChildSignal(
    llvm::StringRef hierarchicalName, SignalId signalId) const {
  llvm::dbgs() << "    Registered child signal '" << hierarchicalName
               << "' with ID " << signalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredChildProcess(
    llvm::StringRef processName, ProcessId processId) const {
  llvm::dbgs() << "    Registered child process '" << processName
               << "' with ID " << processId << "\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredChildCombinational(
    llvm::StringRef combinationalName, ProcessId processId,
    size_t sensitivityCount) const {
  llvm::dbgs() << "    Registered child combinational '"
               << combinationalName << "' with ID " << processId << " and "
               << sensitivityCount << " sensitivities\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredChildInitialBlock(
    llvm::StringRef initialName, ProcessId processId) const {
  llvm::dbgs() << "    Registered child initial block '" << initialName
               << "' with ID " << processId << "\n";
}

void LLHDProcessInterpreter::maybeTraceChildInputMapped(
    llvm::StringRef portName, SignalId signalId) const {
  llvm::dbgs() << "    Mapped child input '" << portName
               << "' to signal " << signalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceChildInstanceOutputCountMismatch(
    llvm::StringRef instanceName, unsigned resultCount,
    unsigned outputCount) const {
  llvm::dbgs() << "    Warning: Instance output count mismatch for '"
               << instanceName << "' (results=" << resultCount
               << ", outputs=" << outputCount << ")\n";
}

void LLHDProcessInterpreter::maybeTraceInitializationRegistrationSummary(
    size_t signalCount, size_t processCount) const {
  llvm::dbgs() << "LLHDProcessInterpreter: Registered "
               << signalCount << " signals and " << processCount
               << " processes\n";
}

void LLHDProcessInterpreter::maybeTraceIterativeDiscoverySummary(
    size_t instanceCount, size_t signalCount, size_t outputCount,
    size_t processCount, size_t combinationalCount, size_t initialCount,
    size_t moduleDriveCount, size_t firRegCount,
    size_t clockedAssertCount) const {
  llvm::dbgs() << "  Iterative discovery found: "
               << instanceCount << " instances, "
               << signalCount << " signals, "
               << outputCount << " outputs, "
               << processCount << " processes, "
               << combinationalCount << " combinationals, "
               << initialCount << " initials, "
               << moduleDriveCount << " module drives, "
               << firRegCount << " firRegs, "
               << clockedAssertCount << " clocked assertions\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredPortSignal(
    llvm::StringRef portName, SignalId signalId, unsigned width) const {
  llvm::dbgs() << "  Registered port signal '" << portName
               << "' with ID " << signalId << " (width=" << width << ")\n";
}

void LLHDProcessInterpreter::maybeTraceMappedExternalPortSignal(
    llvm::StringRef portName, SignalId signalId) const {
  llvm::dbgs() << "  Mapped external port signal '" << portName
               << "' block arg to signal ID " << signalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredOutputSignal(
    llvm::StringRef outputName, SignalId signalId, unsigned width) const {
  llvm::dbgs() << "  Registered output signal '" << outputName
               << "' with ID " << signalId
               << " (width=" << width << ")\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredSignalInitialValue(
    const llvm::APInt &initialValue) const {
  llvm::dbgs() << "  Set initial value to " << initialValue << "\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredSignal(
    llvm::StringRef signalName, SignalId signalId, unsigned width) const {
  llvm::dbgs() << "  Registered signal '" << signalName
               << "' with ID " << signalId
               << " (width=" << width << ")\n";
}

void LLHDProcessInterpreter::maybeTraceExportSignalsBegin(
    size_t signalCount) const {
  llvm::dbgs() << "LLHDProcessInterpreter: Exporting "
               << signalCount << " signals to MooreRuntime registry\n";
}

void LLHDProcessInterpreter::maybeTraceExportedSignal(
    llvm::StringRef hierarchicalPath, SignalId signalId,
    uint32_t width) const {
  llvm::dbgs() << "  Exported '" << hierarchicalPath
               << "' (ID=" << signalId
               << ", width=" << width << ")\n";
}

void LLHDProcessInterpreter::maybeTraceSignalRegistryEntryCount(
    size_t entryCount) const {
  llvm::dbgs() << "LLHDProcessInterpreter: Signal registry now has "
               << entryCount << " entries\n";
}

void LLHDProcessInterpreter::maybeTraceRegistryAccessorsConfigured(
    bool connected) const {
  llvm::dbgs() << "LLHDProcessInterpreter: Registry accessors configured, "
               << "connected=" << connected << "\n";
}

void LLHDProcessInterpreter::maybeTraceFoundProcessOp(
    unsigned resultCount) const {
  llvm::dbgs() << "  Found llhd.process op (numResults="
               << resultCount << ")\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredTopProcess(
    llvm::StringRef processName, ProcessId processId) const {
  llvm::dbgs() << "  Registered process '" << processName
               << "' with ID " << processId << "\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredTopCombinationalProcess(
    llvm::StringRef processName, ProcessId processId,
    size_t sensitivityCount) const {
  llvm::dbgs() << "  Registered combinational process '"
               << processName << "' with ID " << processId
               << " and " << sensitivityCount << " sensitivities\n";
}

void LLHDProcessInterpreter::maybeTraceRegisteredTopInitialBlock(
    llvm::StringRef processName, ProcessId processId) const {
  llvm::dbgs() << "  Registered initial block '" << processName
               << "' with ID " << processId << "\n";
}

void LLHDProcessInterpreter::traceInterfaceSignalOverwrite(
    uint64_t fieldAddress, SignalId oldSignalId, SignalId newSignalId,
    llvm::StringRef newSignalName) const {
  llvm::errs() << "[IFACE-OVERWRITE] addr=0x"
               << llvm::format_hex(fieldAddress, 10)
               << " old sig=" << oldSignalId
               << " new sig=" << newSignalId
               << " (" << newSignalName << ")\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverPostNbaConditionalSignals(
    size_t pendingFirRegCount) const {
  if (!isMultiDriverTraceEnabled() || processConditionalDriveValues.empty())
    return;
  llvm::errs() << "[MULTI-DRV-POST-NBA] conditional signals: {";
  for (const auto &[signalId, _] : processConditionalDriveValues)
    llvm::errs() << signalId << ",";
  llvm::errs() << "} pendingFirRegs=" << pendingFirRegCount
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverPostNbaApply(
    SignalId signalId, const SignalValue &driveValue) const {
  if (!isMultiDriverTraceEnabled())
    return;
  llvm::SmallString<64> hexBuf;
  driveValue.getAPInt().toString(hexBuf, 16, false);
  llvm::errs() << "[MULTI-DRV-POST-NBA] applying conditional drive for sig="
               << signalId << " w=" << driveValue.getWidth()
               << " isX=" << driveValue.isUnknown() << " val=0x" << hexBuf
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverPostNbaSkipFirReg(
    SignalId signalId) const {
  if (!isMultiDriverTraceEnabled())
    return;
  llvm::errs() << "[MULTI-DRV-POST-NBA] SKIPPED firreg update for sig="
               << signalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverSuppressedUnconditional(
    SignalId signalId) const {
  if (!isMultiDriverTraceEnabled())
    return;
  llvm::errs() << "[MULTI-DRV] suppressed unconditional drive to sig="
               << signalId << " (conditional already claimed)\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverConditionalEnable(
    SignalId signalId, const InterpretedValue &enableValue) const {
  if (!isMultiDriverTraceEnabled() || signalId != 7)
    return;
  llvm::errs() << "[MULTI-DRV] conditional drive to sig=" << signalId
               << " enable="
               << (enableValue.isX() ? "X"
                                     : std::to_string(enableValue.getUInt64()))
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

void LLHDProcessInterpreter::maybeTraceMultiDriverStoredConditional(
    SignalId signalId, const SignalValue &driveValue) const {
  if (!isMultiDriverTraceEnabled() || signalId != 7)
    return;
  llvm::SmallString<64> hexBuf;
  driveValue.getAPInt().toString(hexBuf, 16, false);
  llvm::errs() << "[MULTI-DRV] stored conditional drive for sig=" << signalId
               << " t=" << scheduler.getCurrentTime().realTime
               << " w=" << driveValue.getWidth()
               << " isX=" << driveValue.isUnknown()
               << " nonzero=" << (driveValue.getAPInt() != 0)
               << " val=0x" << hexBuf << "\n";
}

void LLHDProcessInterpreter::maybeTraceExecuteModuleDrives(
    ProcessId procId) const {
  if (!isModuleDriveTraceEnabled())
    return;
  llvm::errs() << "[EXEC-MOD-DRV] t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep
               << " proc=" << procId << "\n";
}

void LLHDProcessInterpreter::maybeTraceModuleDriveTrigger(
    SignalId triggerSignalId, SignalId destinationSignalId,
    ProcessId processId) const {
  if (!isModuleDriveTraceEnabled())
    return;

  const auto &signalNames = scheduler.getSignalNames();
  auto sourceNameIt = signalNames.find(triggerSignalId);
  auto destinationNameIt = signalNames.find(destinationSignalId);
  llvm::errs() << "[MOD-DRV] t=" << scheduler.getCurrentTime().realTime
               << " trigger_sig=" << triggerSignalId
               << " (" << (sourceNameIt != signalNames.end()
                               ? sourceNameIt->second
                               : "<?>")
               << ")"
               << " -> dst_sig=" << destinationSignalId
               << " (" << (destinationNameIt != signalNames.end()
                               ? destinationNameIt->second
                               : "<?>")
               << ")"
               << " proc=" << processId << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfacePropagationSource(
    SignalId sourceSignalId, const InterpretedValue &sourceValue,
    size_t fanoutCount) const {
  if (!isInterfacePropagationTraceEnabled())
    return;

  auto signalName = [&](SignalId sigId) -> llvm::StringRef {
    auto it = signalIdToName.find(sigId);
    if (it != signalIdToName.end())
      return it->second;
    return "<unknown>";
  };
  auto formatValue = [](const InterpretedValue &iv) -> std::string {
    if (iv.isX())
      return "X";
    llvm::SmallString<64> bits;
    iv.getAPInt().toStringUnsigned(bits, /*Radix=*/2);
    return std::string(bits);
  };

  SimTime now = scheduler.getCurrentTime();
  llvm::errs() << "[IFACE-PROP] src sig=" << sourceSignalId << " ("
               << signalName(sourceSignalId) << ") val="
               << formatValue(sourceValue) << " fanout=" << fanoutCount
               << " t=" << now.realTime << " d=" << now.deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfacePropagationChild(
    SignalId childSignalId, const InterpretedValue &childValue) const {
  if (!isInterfacePropagationTraceEnabled())
    return;

  auto signalName = [&](SignalId sigId) -> llvm::StringRef {
    auto it = signalIdToName.find(sigId);
    if (it != signalIdToName.end())
      return it->second;
    return "<unknown>";
  };
  auto formatValue = [](const InterpretedValue &iv) -> std::string {
    if (iv.isX())
      return "X";
    llvm::SmallString<64> bits;
    iv.getAPInt().toStringUnsigned(bits, /*Radix=*/2);
    return std::string(bits);
  };

  SimTime now = scheduler.getCurrentTime();
  llvm::errs() << "[IFACE-PROP]   -> child sig=" << childSignalId << " ("
               << signalName(childSignalId) << ") val="
               << formatValue(childValue) << " t=" << now.realTime
               << " d=" << now.deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceCopyPairLink(
    SignalId parentSignalId, uint64_t parentAddress, SignalId childSignalId,
    uint64_t childAddress) const {
  llvm::errs() << "[circt-sim] CopyPair link: signal " << parentSignalId
               << " (0x" << llvm::format_hex(parentAddress, 16)
               << ", w=" << scheduler.getSignalValue(parentSignalId).getWidth()
               << ") -> signal " << childSignalId << " (0x"
               << llvm::format_hex(childAddress, 16)
               << ", w=" << scheduler.getSignalValue(childSignalId).getWidth()
               << ")\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceDeferredSameInterfaceLink(
    SignalId sourceSignalId, SignalId destinationSignalId) const {
  llvm::errs() << "[circt-sim] Deferred same-interface link: signal "
               << sourceSignalId << " -> signal " << destinationSignalId
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceCopyPairSummary(
    size_t totalPairs, unsigned resolvedPairs, unsigned unresolvedSourceCount,
    unsigned unresolvedDestCount, unsigned resolvedDeferredCount,
    size_t childToParentCount, size_t propagationCount) const {
  llvm::errs() << "[circt-sim] childModuleCopyPairs: "
               << totalPairs << " total, "
               << resolvedPairs << " resolved, "
               << unresolvedSourceCount << " unresolved-src, "
               << unresolvedDestCount << " unresolved-dest\n";
  if (resolvedDeferredCount > 0)
    llvm::errs() << "[circt-sim] same-interface deferred links: "
                 << resolvedDeferredCount << " resolved\n";
  llvm::errs() << "[circt-sim] childToParentFieldAddr has "
               << childToParentCount << " entries, "
               << "interfaceFieldPropagation has "
               << propagationCount << " entries\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceSignalCopyLink(
    SignalId sourceSignalId, SignalId destinationSignalId,
    uint64_t destinationAddress) const {
  llvm::errs() << "[circt-sim] SignalCopy link: signal " << sourceSignalId
               << " -> field signal " << destinationSignalId << " (0x"
               << llvm::format_hex(destinationAddress, 16) << ")\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceSignalCopySummary(
    size_t totalPairs, unsigned resolvedPairs,
    unsigned unresolvedDestCount) const {
  llvm::errs() << "[circt-sim] interfaceSignalCopyPairs: "
               << totalPairs << " total, "
               << resolvedPairs << " resolved, " << unresolvedDestCount
               << " unresolved-dest\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceFieldSignalDumpHeader(
    size_t totalEntries) const {
  llvm::dbgs() << "[circt-sim] Interface field signals: "
               << totalEntries << " entries\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceFieldSignalDumpEntry(
    uint64_t address, SignalId signalId) const {
  llvm::dbgs() << "  addr=0x" << llvm::format_hex(address, 16)
               << " -> signal " << signalId;
  auto nameIt = signalIdToName.find(signalId);
  if (nameIt != signalIdToName.end())
    llvm::dbgs() << " (" << nameIt->second << ")";
  llvm::dbgs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceAutoLinkSignalDumpHeader(
    size_t totalInterfaces) const {
  llvm::errs() << "[circt-sim] Interface signal dump: "
               << totalInterfaces << " total interfaces\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceAutoLinkSignalDumpEntry(
    SignalId interfaceSignalId, llvm::ArrayRef<SignalId> fieldSignalIds) const {
  auto interfaceNameIt = signalIdToName.find(interfaceSignalId);
  llvm::StringRef interfaceName = interfaceNameIt != signalIdToName.end()
                                      ? interfaceNameIt->second
                                      : "unnamed";
  llvm::errs() << "  Interface signal " << interfaceSignalId
               << " (" << interfaceName << "): "
               << fieldSignalIds.size() << " fields, widths=[";
  for (size_t i = 0; i < fieldSignalIds.size(); ++i) {
    if (i > 0)
      llvm::errs() << ",";
    llvm::errs() << scheduler.getSignalValue(fieldSignalIds[i]).getWidth();
  }
  llvm::errs() << "]\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceParentSignals(
    llvm::ArrayRef<SignalId> parentSignalIds) const {
  llvm::errs() << "[circt-sim] Parent interface signals: "
               << parentSignalIds.size() << " [";
  for (SignalId signalId : parentSignalIds)
    llvm::errs() << signalId << " ";
  llvm::errs() << "]\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceInstanceAwareLink(
    SignalId childSignalId, SignalId parentSignalId) const {
  llvm::errs() << "[circt-sim] Instance-aware: child " << childSignalId
               << " -> parent " << parentSignalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceFieldPropagationMap() const {
  llvm::errs() << "[circt-sim] interfaceFieldPropagation map ("
               << interfaceFieldPropagation.size() << " parent fields):\n";
  for (auto &[parentSig, children] : interfaceFieldPropagation) {
    auto parentIt = signalIdToName.find(parentSig);
    std::string parentName =
        parentIt != signalIdToName.end() ? parentIt->second : "?";
    llvm::errs() << "  parent sig " << parentSig << " (" << parentName
                 << ", w=" << scheduler.getSignalValue(parentSig).getWidth()
                 << ") -> " << children.size() << " children:";
    for (SignalId cid : children) {
      auto childIt = signalIdToName.find(cid);
      std::string childName = childIt != signalIdToName.end()
                                  ? childIt->second
                                  : "?";
      llvm::errs() << " " << cid << "(" << childName
                   << ",w=" << scheduler.getSignalValue(cid).getWidth()
                   << ")";
    }
    llvm::errs() << "\n";
  }
}

void LLHDProcessInterpreter::maybeTraceInterfaceAutoLinkMatch(
    unsigned bestMatchCount, SignalId childInterfaceSignalId,
    SignalId parentInterfaceSignalId) const {
  llvm::errs() << "[circt-sim] Auto-linked " << bestMatchCount
               << " fields from child interface " << childInterfaceSignalId
               << " to parent interface " << parentInterfaceSignalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceAutoLinkTotal(
    unsigned autoLinkedCount) const {
  llvm::errs() << "[circt-sim] Total auto-linked " << autoLinkedCount
               << " BFM interface fields to parent interfaces\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkDetection(
    size_t reverseTargetCount, size_t interfaceGroupCount) const {
  llvm::errs() << "[circt-sim] Intra-link detection: "
               << reverseTargetCount << " reverse targets, "
               << interfaceGroupCount << " interface groups\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkReverseTarget(
    SignalId signalId, size_t childCount) const {
  auto it = signalIdToName.find(signalId);
  llvm::StringRef name = it != signalIdToName.end() ? it->second : "?";
  llvm::errs() << "  reverse target sig " << signalId << " (" << name
               << ") from " << childCount << " children\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkBlock(
    size_t fieldCount, size_t publicCount, size_t danglingCount) const {
  llvm::errs() << "[circt-sim] Interface block: " << fieldCount
               << " fields, " << publicCount << " public, "
               << danglingCount << " dangling\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkChildBlocks(
    size_t childBlockCount) const {
  llvm::errs() << "[circt-sim]   child blocks: " << childBlockCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkMatch(
    SignalId danglingSignalId, SignalId publicSignalId,
    size_t publicChildCount) const {
  auto danglingIt = signalIdToName.find(danglingSignalId);
  llvm::StringRef danglingName =
      danglingIt != signalIdToName.end() ? danglingIt->second : "?";
  auto publicIt = signalIdToName.find(publicSignalId);
  llvm::StringRef publicName =
      publicIt != signalIdToName.end() ? publicIt->second : "?";
  llvm::errs() << "[circt-sim] Intra-interface link: " << danglingSignalId
               << " (" << danglingName << ") -> " << publicSignalId
               << " (" << publicName << ") + " << publicChildCount
               << " children\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceIntraLinkTotal(
    unsigned intraLinkCount) const {
  llvm::errs() << "[circt-sim] Added " << intraLinkCount
               << " intra-interface field propagation links\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceTriStateCandidateInstall(
    SignalId conditionSignalId, SignalId sourceSignalId,
    SignalId destinationSignalId, unsigned conditionBitIndex,
    const InterpretedValue &elseValue) const {
  llvm::errs() << "[circt-sim] TriState rule: cond=" << conditionSignalId
               << " src=" << sourceSignalId
               << " dest=" << destinationSignalId
               << " condBit=" << conditionBitIndex << " else=0x"
               << llvm::format_hex(elseValue.isX() ? 0ULL : elseValue.getUInt64(), 10)
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceTriStateCandidateSummary(
    size_t candidateCount, unsigned installedCount, unsigned unresolvedCount,
    size_t totalRuleCount) const {
  llvm::errs() << "[circt-sim] TriState candidates: "
               << candidateCount
               << ", installed=" << installedCount
               << ", unresolved=" << unresolvedCount
               << ", total_rules=" << totalRuleCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceTriStateTrigger(
    SignalId sourceSignalId, size_t ruleCount) const {
  if (!isInterfaceTriStateTraceEnabled())
    return;

  auto signalName = [&](SignalId sigId) -> llvm::StringRef {
    auto it = signalIdToName.find(sigId);
    if (it != signalIdToName.end())
      return it->second;
    return "<unknown>";
  };
  llvm::errs() << "[TRI-RULE] trigger sig=" << sourceSignalId << " ("
               << signalName(sourceSignalId) << ") rules=" << ruleCount
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceInterfaceTriStateRule(
    unsigned ruleIndex, SignalId conditionSignalId, unsigned conditionBitIndex,
    const InterpretedValue &conditionValue, bool conditionTrue,
    SignalId sourceSignalId, const InterpretedValue &sourceValue,
    SignalId destinationSignalId, const InterpretedValue &destinationBefore,
    const InterpretedValue &selectedValue, bool selectedIsExplicitHighZ) const {
  if (!isInterfaceTriStateTraceEnabled())
    return;

  auto signalName = [&](SignalId sigId) -> llvm::StringRef {
    auto it = signalIdToName.find(sigId);
    if (it != signalIdToName.end())
      return it->second;
    return "<unknown>";
  };
  auto formatValue = [](const InterpretedValue &value) -> std::string {
    if (value.isX())
      return "X";
    llvm::SmallString<64> bits;
    value.getAPInt().toStringUnsigned(bits, /*Radix=*/2);
    return std::string(bits);
  };

  llvm::errs() << "[TRI-RULE] idx=" << ruleIndex
               << " cond=" << conditionSignalId << "("
               << signalName(conditionSignalId) << ")="
               << formatValue(conditionValue) << " bit=" << conditionBitIndex
               << " condTrue=" << (conditionTrue ? 1 : 0)
               << " src=" << sourceSignalId << "("
               << signalName(sourceSignalId) << ")="
               << formatValue(sourceValue) << " dest=" << destinationSignalId
               << "(" << signalName(destinationSignalId) << ") before="
               << formatValue(destinationBefore) << " sel="
               << formatValue(selectedValue)
               << " selHighZ=" << (selectedIsExplicitHighZ ? 1 : 0) << "\n";
}

void LLHDProcessInterpreter::maybeTraceCondBranch(
    ProcessId procId, const InterpretedValue &conditionValue,
    bool tookTrueBranch, mlir::Value conditionSsaValue) {
  if (!isCondBranchTraceEnabled())
    return;

  llvm::errs() << "[CONDBR] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  if (tookTrueBranch) {
    llvm::errs() << " cond=1";
  } else if (conditionValue.isX()) {
    llvm::errs() << " cond=X";
  } else {
    llvm::errs() << " cond=0";
  }
  llvm::errs() << " dest=" << (tookTrueBranch ? "true" : "false");

  if (Operation *defOp = conditionSsaValue.getDefiningOp()) {
    llvm::errs() << " def=" << defOp->getName().getStringRef();
    for (Value operand : defOp->getOperands()) {
      InterpretedValue operandValue = getValue(procId, operand);
      llvm::errs() << " opnd=";
      if (operandValue.isX()) {
        llvm::errs() << "X";
      } else {
        llvm::SmallString<32> bits;
        operandValue.getAPInt().toStringUnsigned(bits, /*Radix=*/2);
        llvm::errs() << bits;
      }
    }
  }
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceInstanceOutput(
    SignalId signalId, llvm::ArrayRef<SignalId> sourceSignals,
    size_t processCount) const {
  if (!isInstanceOutputTraceEnabled())
    return;

  auto nameIt = scheduler.getSignalNames().find(signalId);
  llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                ? llvm::StringRef(nameIt->second)
                                : llvm::StringRef("<unknown>");
  llvm::errs() << "[INST-OUT] sig=" << signalId << " name=" << sigName
               << " srcSignals=" << sourceSignals.size()
               << " processIds=" << processCount;
  for (SignalId sourceSignalId : sourceSignals) {
    auto srcNameIt = scheduler.getSignalNames().find(sourceSignalId);
    llvm::errs() << " src=" << sourceSignalId << "("
                 << (srcNameIt != scheduler.getSignalNames().end()
                         ? llvm::StringRef(srcNameIt->second)
                         : llvm::StringRef("?"))
                 << ")";
  }
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceInstanceOutputDependencySignals(
    SignalId signalId, llvm::ArrayRef<SignalId> sourceSignals) const {
  llvm::dbgs() << "Instance output signal " << signalId
               << " depends on " << sourceSignals.size()
               << " source signals:";
  for (SignalId sourceSignalId : sourceSignals)
    llvm::dbgs() << " " << sourceSignalId;
  llvm::dbgs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceInstanceOutputUpdate(
    SignalId signalId, const InterpretedValue &value) const {
  if (!isInstanceOutputUpdateTraceEnabled())
    return;

  auto nameIt = scheduler.getSignalNames().find(signalId);
  llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                ? llvm::StringRef(nameIt->second)
                                : llvm::StringRef("<unknown>");
  llvm::SmallString<64> bits;
  if (value.isX())
    bits = "X";
  else
    value.getAPInt().toString(bits, 16, false);
  llvm::errs() << "[INST-OUT-UPD] sig=" << signalId
               << " name=" << sigName
               << " val=0x" << bits
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

void LLHDProcessInterpreter::maybeTraceCombTraceThroughHit(
    SignalId signalId, const InterpretedValue &driveValue) const {
  if (!isCombTraceThroughEnabled())
    return;

  auto nameIt = scheduler.getSignalNames().find(signalId);
  llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                ? llvm::StringRef(nameIt->second)
                                : llvm::StringRef("<unknown>");
  llvm::SmallString<64> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 16, false);
  llvm::errs() << "[COMB-TT] sig=" << signalId
               << " name=" << sigName
               << " combDrv=0x" << bits
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

void LLHDProcessInterpreter::maybeTraceCombTraceThroughMiss(
    SignalId signalId, bool inCombMap, bool visited) const {
  if (!isCombTraceThroughEnabled())
    return;

  auto nameIt = scheduler.getSignalNames().find(signalId);
  llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                ? llvm::StringRef(nameIt->second)
                                : llvm::StringRef("<unknown>");
  llvm::errs() << "[COMB-TT-MISS] sig=" << signalId
               << " name=" << sigName
               << " inCombMap=" << (inCombMap ? 1 : 0)
               << " visited=" << (visited ? 1 : 0)
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

void LLHDProcessInterpreter::maybeTraceContinuousFallback(
    mlir::Value value, unsigned invalidatedCount) const {
  if (!isContinuousFallbackTraceEnabled())
    return;
  llvm::errs() << "[CONT-FALLBACK] value=";
  if (Operation *op = value.getDefiningOp())
    llvm::errs() << op->getName();
  else
    llvm::errs() << "<arg>";
  llvm::errs() << " invalidated=" << invalidatedCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceDriveSchedule(
    SignalId signalId, const InterpretedValue &driveValue,
    const SimTime &currentTime, const SimTime &targetTime,
    const SimTime &delay) const {
  if (!isDriveScheduleTraceEnabled())
    return;
  auto nameIt = signalIdToName.find(signalId);
  llvm::StringRef sigName =
      nameIt != signalIdToName.end() ? llvm::StringRef(nameIt->second)
                                     : llvm::StringRef("<unknown>");
  llvm::SmallString<64> valBits;
  if (driveValue.isX())
    valBits = "X";
  else
    driveValue.getAPInt().toString(valBits, 16, false);
  llvm::errs() << "[DRV-SCHED] sig=" << signalId << " (" << sigName << ")"
               << " val=0x" << valBits
               << " now=(" << currentTime.realTime << ",d"
               << currentTime.deltaStep << ")"
               << " target=(" << targetTime.realTime << ",d"
               << targetTime.deltaStep << ")"
               << " delay=(" << delay.realTime << ",d" << delay.deltaStep
               << ")\n";
}

void LLHDProcessInterpreter::maybeTraceDriveFailure(
    ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef reason,
    mlir::Value signalValue) const {
  if (!isDriveFailureTraceEnabled())
    return;
  llvm::errs() << "[DRIVE-FAIL] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  if (auto func = driveOp->getParentOfType<func::FuncOp>())
    llvm::errs() << " func='" << func.getSymName() << "'";
  llvm::errs() << " reason=" << reason << " signal=" << signalValue
               << " loc=" << driveOp.getLoc() << "\n";
}

void LLHDProcessInterpreter::maybeTraceRefArgResolveFailure(
    ProcessId procId, mlir::Value unresolvedValue, bool hasBlockArgSource,
    mlir::Value blockArgSourceValue, SignalId blockArgSourceSignal) const {
  if (!isRefArgResolveTraceEnabled())
    return;
  llvm::errs() << "[REF-RESOLVE] drive unresolved struct parent proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  llvm::errs() << " value=" << unresolvedValue << "\n";
  if (isa<BlockArgument>(unresolvedValue)) {
    if (hasBlockArgSource) {
      llvm::errs() << "[REF-RESOLVE]   block-arg source=" << blockArgSourceValue
                   << " sig=" << blockArgSourceSignal << "\n";
    } else {
      llvm::errs() << "[REF-RESOLVE]   block-arg source=<none>\n";
    }
  }
}

void LLHDProcessInterpreter::maybeTraceI3CFieldDriveSignalStruct(
    ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef fieldName,
    unsigned bitOffset, unsigned fieldWidth, SignalId parentSignalId,
    const InterpretedValue &driveValue) const {
  if (!isI3CFieldDriveTraceEnabled() ||
      !shouldTraceI3CFieldDriveName(fieldName))
    return;
  llvm::SmallString<64> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 16, false);
  llvm::StringRef procName = "<unknown>";
  if (const Process *process = scheduler.getProcess(procId))
    procName = process->getName();
  llvm::StringRef funcName = "<unknown>";
  if (auto func = driveOp->getParentOfType<func::FuncOp>())
    funcName = func.getSymName();
  llvm::errs() << "[I3C-FIELD-DRV-SIGNAL-STRUCT] proc=" << procId
               << " name='" << procName << "'"
               << " func='" << funcName << "'"
               << " field=" << fieldName
               << " bitOffset=" << bitOffset
               << " width=" << fieldWidth
               << " val=0x" << bits
               << " sig=" << parentSignalId
               << " t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CFieldDriveMemStruct(
    ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef fieldName,
    int64_t index, unsigned bitOffset, int64_t fieldWidth,
    const InterpretedValue &driveValue) const {
  if (!isI3CFieldDriveTraceEnabled() ||
      !shouldTraceI3CFieldDriveName(fieldName))
    return;
  llvm::SmallString<64> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 16, false);
  llvm::StringRef procName = "<unknown>";
  if (const Process *process = scheduler.getProcess(procId))
    procName = process->getName();
  llvm::StringRef funcName = "<unknown>";
  if (auto func = driveOp->getParentOfType<func::FuncOp>())
    funcName = func.getSymName();
  llvm::errs() << "[I3C-FIELD-DRV-MEM-STRUCT] proc=" << procId
               << " name='" << procName << "'"
               << " func='" << funcName << "'"
               << " field=" << fieldName;
  if (index >= 0)
    llvm::errs() << " idx=" << index;
  llvm::errs() << " bitOffset=" << bitOffset;
  if (fieldWidth >= 0)
    llvm::errs() << " width=" << fieldWidth;
  llvm::errs() << " val=0x" << bits
               << " t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CFieldDriveMem(
    ProcessId procId, llhd::DriveOp driveOp, uint64_t index,
    uint64_t elementOffset, const InterpretedValue &driveValue) const {
  if (!isI3CFieldDriveTraceEnabled())
    return;
  llvm::SmallString<64> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 16, false);
  llvm::StringRef procName = "<unknown>";
  if (const Process *process = scheduler.getProcess(procId))
    procName = process->getName();
  llvm::StringRef funcName = "<unknown>";
  if (auto func = driveOp->getParentOfType<func::FuncOp>())
    funcName = func.getSymName();
  llvm::errs() << "[I3C-FIELD-DRV-MEM] proc=" << procId
               << " name='" << procName << "'"
               << " func='" << funcName << "'"
               << " idx=" << index
               << " elemOffset=" << elementOffset
               << " val=0x" << bits
               << " t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CFieldDrive(
    ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef fieldName,
    uint64_t index, unsigned bitOffset,
    const InterpretedValue &driveValue) const {
  if (!isI3CFieldDriveTraceEnabled())
    return;
  llvm::SmallString<64> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 16, false);
  llvm::StringRef procName = "<unknown>";
  if (const Process *process = scheduler.getProcess(procId))
    procName = process->getName();
  llvm::StringRef funcName = "<unknown>";
  if (auto func = driveOp->getParentOfType<func::FuncOp>())
    funcName = func.getSymName();
  llvm::errs() << "[I3C-FIELD-DRV] proc=" << procId
               << " name='" << procName << "'"
               << " func='" << funcName << "'"
               << " field=" << fieldName
               << " idx=" << index
               << " bitOffset=" << bitOffset
               << " val=0x" << bits
               << " t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceArrayDriveRemap(
    ProcessId procId, mlir::Value originalSignal, mlir::Value remappedSignal) {
  if (!isArrayDriveTraceEnabled() || !isa<BlockArgument>(originalSignal))
    return;
  llvm::errs() << "[ARRAY-DRV] blockarg→";
  if (remappedSignal == originalSignal)
    llvm::errs() << "UNRESOLVED";
  else if (remappedSignal.getDefiningOp<llhd::SigArrayGetOp>())
    llvm::errs() << "SigArrayGet";
  else if (auto defOp = remappedSignal.getDefiningOp())
    llvm::errs() << defOp->getName().getStringRef();
  else
    llvm::errs() << "otherBlockArg";
  llvm::errs() << " sigId=" << getSignalId(remappedSignal);
  if (auto sigArrayGetOp = remappedSignal.getDefiningOp<llhd::SigArrayGetOp>()) {
    llvm::errs() << " parentSigId=" << getSignalId(sigArrayGetOp.getInput());
    InterpretedValue indexValue = getValue(procId, sigArrayGetOp.getIndex());
    llvm::errs() << " idx="
                 << (indexValue.isX() ? "X"
                                      : std::to_string(indexValue.getUInt64()));
  }
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceArrayDriveSchedule(
    SignalId parentSignalId, uint64_t index, unsigned bitOffset,
    unsigned elementWidth, const llvm::APInt &elementValue, bool driveValueIsX,
    unsigned parentWidth, unsigned resultWidth, const SimTime &delay) const {
  if (!isArrayDriveTraceEnabled())
    return;
  llvm::SmallString<64> elementBits;
  elementValue.toString(elementBits, 16, false);
  llvm::errs() << "[ARRAY-DRV-SCHED] sig=" << parentSignalId
               << " idx=" << index
               << " off=" << bitOffset
               << " elemW=" << elementWidth
               << " elemVal=0x" << elementBits
               << " driveVal=" << (driveValueIsX ? "X" : "known")
               << " parentW=" << parentWidth
               << " resW=" << resultWidth
               << " delay.rt=" << delay.realTime
               << " delay.d=" << delay.deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CAddressBitDrive(
    llhd::DriveOp driveOp, unsigned bitOffset,
    const InterpretedValue &driveValue) const {
  if (!isI3CAddressBitTraceEnabled())
    return;
  auto parentFunc = driveOp->getParentOfType<func::FuncOp>();
  if (!parentFunc || parentFunc.getSymName() !=
                         "i3c_target_driver_bfm::sample_target_address")
    return;
  llvm::SmallString<32> bits;
  if (driveValue.isX())
    bits = "X";
  else
    driveValue.getAPInt().toString(bits, 2, false);
  llvm::errs() << "[I3C-ADDR-BIT] bit=" << bitOffset << " val=" << bits
               << " t=" << scheduler.getCurrentTime().realTime << " d="
               << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CRefCast(
    ProcessId procId, bool resolved, uint64_t address, unsigned bitOffset,
    mlir::Value inputValue, mlir::Value outputValue) const {
  if (!isI3CRefCastTraceEnabled())
    return;
  llvm::errs() << "[I3C-REF-CAST] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  auto stateIt = processStates.find(procId);
  if (stateIt != processStates.end())
    llvm::errs() << " func='" << stateIt->second.currentFuncName << "'";
  if (resolved)
    llvm::errs() << " resolved=1 addr=0x"
                 << llvm::format_hex(address, 16)
                 << " bitOffset=" << bitOffset;
  else
    llvm::errs() << " resolved=0";
  llvm::errs() << " in=" << inputValue << " out=" << outputValue << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CCastLayout(
    ProcessId procId, mlir::Type outputType, const llvm::APInt &inputBits,
    const llvm::APInt &convertedBits) const {
  if (!isI3CCastLayoutTraceEnabled() || !isI3CTransferStructType(outputType))
    return;
  if (inputBits.getBitWidth() < 2353 || convertedBits.getBitWidth() < 2353)
    return;

  uint64_t inTa = inputBits.extractBits(7, 0).getZExtValue();
  uint64_t inAck = inputBits.extractBits(1, 8).getZExtValue();
  uint64_t inBits = inputBits.extractBits(32, 2313).getZExtValue();
  uint64_t outTa = convertedBits.extractBits(7, 2346).getZExtValue();
  uint64_t outAck = convertedBits.extractBits(1, 2344).getZExtValue();
  uint64_t outBits = convertedBits.extractBits(32, 8).getZExtValue();

  llvm::errs() << "[I3C-CAST] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  auto stateIt = processStates.find(procId);
  if (stateIt != processStates.end())
    llvm::errs() << " func='" << stateIt->second.currentFuncName << "'";
  llvm::errs() << " in{ta=" << inTa << " ack=" << inAck << " bits=" << inBits
               << "} out{ta=" << outTa << " ack=" << outAck
               << " bits=" << outBits
               << "} t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CConfigHandleGet(
    llvm::StringRef callee, llvm::StringRef key, uint64_t ptrPayload,
    uint64_t outRefAddr, llvm::StringRef fieldName) const {
  llvm::errs() << "[I3C-CFG] get callee=" << callee
               << " key=\"" << key << "\" value_ptr=0x"
               << llvm::format_hex(ptrPayload, 10) << " out_ref=0x"
               << llvm::format_hex(outRefAddr, 10)
               << " field=\"" << fieldName << "\"\n";
}

void LLHDProcessInterpreter::maybeTraceI3CConfigHandleSet(
    llvm::StringRef callee, llvm::StringRef key, uint64_t ptrPayload,
    llvm::StringRef fieldName) const {
  llvm::errs() << "[I3C-CFG] set callee=" << callee
               << " key=\"" << key << "\" value_ptr=0x"
               << llvm::format_hex(ptrPayload, 10)
               << " field=\"" << fieldName << "\"\n";
}

void LLHDProcessInterpreter::maybeTraceI3CHandleCall(
    ProcessId procId, llvm::StringRef calleeName,
    llvm::ArrayRef<InterpretedValue> args) const {
  if (args.empty() || args.front().isX())
    return;
  static uint64_t printed = 0;
  if (printed >= 512)
    return;
  ++printed;

  SimTime now = scheduler.getCurrentTime();
  llvm::errs() << "[I3C-HANDLE] call proc=" << procId
               << " t=" << now.realTime << " d=" << now.deltaStep
               << " callee=" << calleeName
               << " self=0x" << llvm::format_hex(args.front().getUInt64(), 10);
  if (args.size() >= 2 && !args[1].isX())
    llvm::errs() << " arg1=0x" << llvm::format_hex(args[1].getUInt64(), 10);
  if (Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " proc_name='" << proc->getName() << "'";
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CCallStackSave(
    ProcessId procId, llvm::StringRef funcName, size_t blockOpCount,
    size_t savedFrameCount) const {
  llvm::errs() << "[I3C-CS-SAVE] proc=" << procId
               << " func=" << funcName
               << " block_ops=" << blockOpCount
               << " saved_frames=" << savedFrameCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceI3CToClassArgs(
    ProcessId procId, llvm::StringRef calleeName,
    llvm::ArrayRef<InterpretedValue> args) const {
  if (args.empty())
    return;
  llvm::errs() << "[I3C-TO-CLASS] callee=" << calleeName << " proc=" << procId;
  if (Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " proc_name='" << proc->getName() << "'";
  auto stateIt = processStates.find(procId);
  if (stateIt != processStates.end()) {
    llvm::errs() << " caller_func='" << stateIt->second.currentFuncName << "'";
    if (stateIt->second.currentBlock) {
      llvm::errs() << " caller_block_ops="
                   << stateIt->second.currentBlock->getOperations().size();
    }
  }
  llvm::errs() << " t=" << scheduler.getCurrentTime().realTime
               << " d=" << scheduler.getCurrentTime().deltaStep;
  const InterpretedValue &structArg = args.front();
  if (structArg.isX()) {
    llvm::errs() << " struct=X";
  } else if (structArg.getWidth() >= 2353) {
    const APInt &bits = structArg.getAPInt();
    // LLVM-style (low-to-high) decode.
    uint64_t taLow = bits.extractBits(7, 0).getZExtValue();
    uint64_t opLow = bits.extractBits(1, 7).getZExtValue();
    uint64_t taAckLow = bits.extractBits(1, 8).getZExtValue();
    uint64_t wd0Low = bits.extractBits(8, 265).getZExtValue();
    uint64_t wd1Low = bits.extractBits(8, 273).getZExtValue();
    uint64_t bitsLow = bits.extractBits(32, 2313).getZExtValue();
    // HW-struct-style (first field in MSBs) decode.
    uint64_t taHw = bits.extractBits(7, 2346).getZExtValue();
    uint64_t opHw = bits.extractBits(1, 2345).getZExtValue();
    uint64_t taAckHw = bits.extractBits(1, 2344).getZExtValue();
    uint64_t wd0Hw = bits.extractBits(8, 1064).getZExtValue();
    uint64_t wd1Hw = bits.extractBits(8, 1072).getZExtValue();
    uint64_t bitsHw = bits.extractBits(32, 8).getZExtValue();
    llvm::errs() << " low{ta=0x" << llvm::format_hex_no_prefix(taLow, 2)
                 << " op=" << opLow << " ack=" << taAckLow
                 << " wd0=0x" << llvm::format_hex_no_prefix(wd0Low, 2)
                 << " wd1=0x" << llvm::format_hex_no_prefix(wd1Low, 2)
                 << " bits=" << bitsLow << "}"
                 << " hw{ta=0x" << llvm::format_hex_no_prefix(taHw, 2)
                 << " op=" << opHw << " ack=" << taAckHw
                 << " wd0=0x" << llvm::format_hex_no_prefix(wd0Hw, 2)
                 << " wd1=0x" << llvm::format_hex_no_prefix(wd1Hw, 2)
                 << " bits=" << bitsHw << "}";
  } else {
    llvm::errs() << " struct_width=" << structArg.getWidth();
  }
  if (args.size() >= 2 && !args[1].isX())
    llvm::errs() << " out_ref=0x" << llvm::format_hex(args[1].getUInt64(), 10);
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceConfigDbFuncCallGetBegin(
    llvm::StringRef callee, llvm::StringRef key, llvm::StringRef instName,
    llvm::StringRef fieldName, size_t entryCount) const {
  llvm::errs() << "[CFG-FC-GET] callee=" << callee
               << " key=\"" << key << "\" inst=\"" << instName
               << "\" field=\"" << fieldName
               << "\" entries=" << entryCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceConfigDbFuncCallGetHit(
    llvm::StringRef key, size_t byteCount) const {
  llvm::errs() << "[CFG-FC-GET] hit key=\"" << key
               << "\" bytes=" << byteCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceConfigDbFuncCallGetMiss(
    llvm::StringRef key) const {
  llvm::errs() << "[CFG-FC-GET] miss key=\"" << key << "\"\n";
}

void LLHDProcessInterpreter::maybeTraceConfigDbFuncCallSet(
    llvm::StringRef callee, llvm::StringRef key, unsigned valueBytes,
    size_t entryCount) const {
  llvm::errs() << "[CFG-FC-SET] callee=" << callee
               << " key=\"" << key << "\" valueBytes=" << valueBytes
               << " entries=" << entryCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallSize(
    llvm::StringRef calleeName, uint64_t selfAddr, int32_t count,
    size_t totalConnections) const {
  llvm::errs() << "[SEQ-SIZE] func.call: " << calleeName
               << " self=0x" << llvm::format_hex(selfAddr, 16)
               << " -> " << count
               << " (total_conns=" << totalConnections << ")\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallGet(
    llvm::StringRef calleeName, uint64_t portAddr, size_t fifoMapCount) const {
  llvm::errs() << "[SEQ-FC] get " << calleeName
               << " port=0x" << llvm::format_hex(portAddr, 16)
               << " fifoMaps=" << fifoMapCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallPop(
    uint64_t itemAddr, uint64_t portAddr, uint64_t queueAddr,
    bool fromFallbackSearch) const {
  llvm::errs() << "[SEQ-FC] pop item=0x" << llvm::format_hex(itemAddr, 16)
               << " port=0x" << llvm::format_hex(portAddr, 16)
               << " seqr_q=0x" << llvm::format_hex(queueAddr, 16)
               << " fallback=" << (fromFallbackSearch ? 1 : 0) << "\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallTryMiss(
    uint64_t portAddr, uint64_t refAddr) const {
  llvm::errs() << "[SEQ-FC] try_next_item miss port=0x"
               << llvm::format_hex(portAddr, 16)
               << " ref=0x" << llvm::format_hex(refAddr, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallWait(
    uint64_t portAddr, uint64_t queueAddr) const {
  llvm::errs() << "[SEQ-FC] wait port=0x" << llvm::format_hex(portAddr, 16)
               << " queue=0x" << llvm::format_hex(queueAddr, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTraceSequencerFuncCallItemDoneMiss(
    uint64_t callerAddr) const {
  llvm::errs() << "[SEQ-FC] item_done miss caller=0x"
               << llvm::format_hex(callerAddr, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallBegin(
    llvm::StringRef calleeName, uint64_t portAddr, bool inConnectionMap) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] " << calleeName
               << " portAddr=" << llvm::format_hex(portAddr, 0)
               << " inMap=" << inConnectionMap << "\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallTerminals(
    size_t terminalCount) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] " << terminalCount
               << " terminal(s)\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallMissingVtableHeader(
    uint64_t impAddr) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] imp 0x"
               << llvm::format_hex(impAddr, 0)
               << " missing readable vtable header\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallMissingAddressToGlobal(
    uint64_t vtableAddr) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] vtable 0x"
               << llvm::format_hex(vtableAddr, 0)
               << " not in addressToGlobal\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallMissingAddressToFunction(
    uint64_t writeFuncAddr) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] slot 11 func addr 0x"
               << llvm::format_hex(writeFuncAddr, 0)
               << " not in addressToFunction\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallMissingModuleFunction(
    llvm::StringRef functionName) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] function '"
               << functionName << "' not found in module\n";
}

void LLHDProcessInterpreter::maybeTraceAnalysisWriteFuncCallDispatch(
    llvm::StringRef functionName) const {
  llvm::errs() << "[ANALYSIS-WRITE-FC] dispatching to "
               << functionName << "\n";
}

void LLHDProcessInterpreter::maybeTraceFuncCacheSharedHit(
    llvm::StringRef functionName, uint64_t argHash) const {
  if (!isFuncCacheTraceEnabled())
    return;
  llvm::errs() << "[FUNC-CACHE] shared hit func=" << functionName
               << " arg_hash=" << llvm::format_hex(argHash, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTraceFuncCacheSharedStore(
    llvm::StringRef functionName, uint64_t argHash) const {
  if (!isFuncCacheTraceEnabled())
    return;
  llvm::errs() << "[FUNC-CACHE] shared store func=" << functionName
               << " arg_hash=" << llvm::format_hex(argHash, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderProcessPhase(
    uint64_t phaseAddr, uint64_t impAddr, std::optional<int> order) const {
  llvm::errs() << "[PHASE-ORDER] process_phase phase=0x"
               << llvm::format_hex(phaseAddr, 18) << " imp=0x"
               << llvm::format_hex(impAddr, 18);
  if (order.has_value())
    llvm::errs() << " order=" << *order;
  else
    llvm::errs() << " order=<unknown>";
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderProcessPhaseWaitPred(
    uint64_t phaseAddr, int order, uint64_t predImpAddr) const {
  llvm::errs() << "[PHASE-ORDER] process_phase wait: phase=0x"
               << llvm::format_hex(phaseAddr, 18)
               << " order=" << order
               << " pred_imp=0x" << llvm::format_hex(predImpAddr, 18)
               << " pred_done=0\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderProcessPhaseWaitUnknownImp(
    uint64_t phaseAddr, uint64_t impAddr) const {
  llvm::errs() << "[PHASE-ORDER] process_phase wait: phase=0x"
               << llvm::format_hex(phaseAddr, 18)
               << " unknown_imp=0x" << llvm::format_hex(impAddr, 18)
               << " waiting_for_all_function_phases\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderFinishPhase(
    uint64_t phaseAddr, uint64_t impAddr, int order, size_t waiterCount) const {
  llvm::errs() << "[PHASE-ORDER] finish_phase phase=0x"
               << llvm::format_hex(phaseAddr, 18) << " imp=0x"
               << llvm::format_hex(impAddr, 18)
               << " order=" << order
               << " completed=1"
               << " waiters=" << waiterCount << "\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderWakeWaiter(
    ProcessId waiterProcId, uint64_t impAddr) const {
  llvm::errs() << "[PHASE-ORDER] wake waiter proc="
               << waiterProcId << " on imp=0x"
               << llvm::format_hex(impAddr, 18) << "\n";
}

void LLHDProcessInterpreter::maybeTracePhaseOrderExecutePhase(
    uint64_t phaseAddr, llvm::StringRef phaseName) const {
  llvm::errs() << "[PHASE-ORDER] execute_phase phase=0x"
               << llvm::format_hex(phaseAddr, 18)
               << " name=\"" << phaseName << "\"\n";
}

void LLHDProcessInterpreter::maybeTraceMailboxTryPut(
    ProcessId procId, uint64_t mboxId, uint64_t message, bool success) const {
  llvm::errs() << "[MAILBOX-TRYPUT] proc=" << procId
               << " mbox=" << mboxId
               << " msg=" << message
               << " ok=" << (success ? 1 : 0) << "\n";
}

void LLHDProcessInterpreter::maybeTraceMailboxWakeGetByTryPut(
    ProcessId procId, ProcessId waiterProcId, uint64_t mboxId, uint64_t outAddr,
    uint64_t message) const {
  llvm::errs() << "[MAILBOX-WAKE-GET] by=tryput proc="
               << procId << " waiter=" << waiterProcId
               << " mbox=" << mboxId
               << " out=0x" << llvm::format_hex(outAddr, 0)
               << " msg=" << message << "\n";
}

void LLHDProcessInterpreter::maybeTraceMailboxGet(
    ProcessId procId, uint64_t mboxId, llvm::StringRef mode, uint64_t outAddr,
    std::optional<uint64_t> message) const {
  llvm::errs() << "[MAILBOX-GET] proc=" << procId
               << " mbox=" << mboxId
               << " mode=" << mode
               << " out=0x" << llvm::format_hex(outAddr, 0);
  if (message.has_value())
    llvm::errs() << " msg=" << *message;
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceRandClassSrandom(uint64_t objAddr,
                                                        uint32_t seed) const {
  llvm::errs() << "[RAND] class_srandom obj=0x"
               << llvm::format_hex(objAddr, 16)
               << " seed=" << seed << "\n";
}

void LLHDProcessInterpreter::maybeTraceRandBasic(uint64_t objAddr, uint64_t size,
                                                 int32_t rc) const {
  llvm::errs() << "[RAND] basic obj=0x"
               << llvm::format_hex(objAddr, 16)
               << " size=" << size
               << " rc=" << rc << "\n";
}

void LLHDProcessInterpreter::maybeTraceRandBytes(uint64_t dataAddr, int64_t size,
                                                 int32_t rc,
                                                 int firstByte) const {
  llvm::errs() << "[RAND] bytes ptr=0x"
               << llvm::format_hex(dataAddr, 16)
               << " size=" << size
               << " rc=" << rc
               << " b0=" << firstByte << "\n";
}

void LLHDProcessInterpreter::maybeTraceRandRange(uint64_t objAddr, int64_t minVal,
                                                 int64_t maxVal,
                                                 int64_t result) const {
  llvm::errs() << "[RAND] range obj=0x"
               << llvm::format_hex(objAddr, 16)
               << " min=" << minVal
               << " max=" << maxVal
               << " -> " << result << "\n";
}

void LLHDProcessInterpreter::maybeTraceTailWrapperSuspendElide(
    ProcessId procId, llvm::StringRef wrapperName, llvm::StringRef calleeName,
    bool monitorDeserializer, bool driveSample) const {
  static bool traceTailWrapperFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_TAIL_WRAPPER_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static bool traceMonitorDeserializerFastPath = []() {
    const char *env =
        std::getenv("CIRCT_SIM_TRACE_MONITOR_DESERIALIZER_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static bool traceDriveSampleFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_DRIVE_SAMPLE_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  if (!(traceTailWrapperFastPath || traceMonitorDeserializerFastPath ||
        traceDriveSampleFastPath))
    return;

  static unsigned tailWrapperSuspendElideHits = 0;
  if (tailWrapperSuspendElideHits >= 100)
    return;
  ++tailWrapperSuspendElideHits;

  if (monitorDeserializer && traceMonitorDeserializerFastPath)
    llvm::errs() << "[MON-DESER-FP]";
  else if (driveSample && traceDriveSampleFastPath)
    llvm::errs() << "[DRV-SAMPLE-FP]";
  else
    llvm::errs() << "[TAIL-WRAP-FP]";
  llvm::errs() << " suspend-elide proc=" << procId
               << " wrapper=" << wrapperName
               << " callee=" << calleeName << "\n";
}

void LLHDProcessInterpreter::maybeTraceOnDemandLoadSignal(
    uint64_t addr, SignalId signalId, llvm::StringRef signalName) const {
  llvm::errs() << "[ONDEMAND-LOAD] addr=0x"
               << llvm::format_hex(addr, 16) << " -> sig="
               << signalId << " (" << signalName << ")\n";
}

void LLHDProcessInterpreter::maybeTraceOnDemandLoadNoSignal(uint64_t addr) const {
  llvm::errs() << "[ONDEMAND-LOAD] addr=0x"
               << llvm::format_hex(addr, 16)
               << " no interface field signal\n";
}

void LLHDProcessInterpreter::maybeTraceStructInjectX(
    llvm::StringRef fieldName, bool structIsX, bool newValueIsX,
    unsigned totalWidth) const {
  llvm::errs() << "[STRUCT-INJECT-X] field=" << fieldName
               << " structX=" << structIsX
               << " newValX=" << newValueIsX
               << " totalWidth=" << totalWidth << "\n";
}

void LLHDProcessInterpreter::maybeTraceAhbTxnFieldWrite(
    ProcessId procId, llvm::StringRef funcName, uint64_t txnAddr,
    llvm::StringRef fieldName, std::optional<uint64_t> index,
    unsigned bitOffset, unsigned width, const InterpretedValue &value,
    const SimTime &now) const {
  llvm::SmallString<64> bits;
  if (value.isX())
    bits = "X";
  else
    value.getAPInt().toString(bits, 16, false);

  llvm::errs() << "[AHB-TXN-FIELD] proc=" << procId
               << " func=" << funcName
               << " txn=" << llvm::format_hex(txnAddr, 16)
               << " field=" << fieldName;
  if (index.has_value())
    llvm::errs() << " idx=" << *index;
  llvm::errs() << " bitOffset=" << bitOffset
               << " width=" << width
               << " val=0x" << bits
               << " t=" << now.realTime
               << " d=" << now.deltaStep
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceFreadSignalPath(
    SignalId signalId, int32_t elemWidth, int32_t elemStorageBytes,
    int32_t numElements, uint64_t totalBytes) const {
  llvm::errs() << "[fread] signal path sigId=" << signalId
               << " elemWidth=" << elemWidth
               << " elemStorageBytes=" << elemStorageBytes
               << " numElems=" << numElements
               << " totalBytes=" << totalBytes << "\n";
}

void LLHDProcessInterpreter::maybeTraceFreadSignalWidth(
    uint64_t bitWidth, size_t rawBytes, int32_t result) const {
  llvm::errs() << "[fread] signal width=" << bitWidth
               << " rawBytes=" << rawBytes
               << " result=" << result << "\n";
}

void LLHDProcessInterpreter::maybeTraceFreadPointerPath(
    int32_t elemWidth, int32_t elemStorageBytes, int32_t numElements,
    uint64_t totalBytes) const {
  llvm::errs() << "[fread] pointer path elemWidth=" << elemWidth
               << " elemStorageBytes=" << elemStorageBytes
               << " numElems=" << numElements
               << " totalBytes=" << totalBytes << "\n";
}

void LLHDProcessInterpreter::maybeTraceFuncProgress(
    ProcessId procId, size_t funcBodySteps, size_t totalSteps,
    llvm::StringRef funcName, size_t callDepth, llvm::StringRef opName) const {
  llvm::errs() << "[circt-sim] func progress: process " << procId
               << " funcBodySteps=" << funcBodySteps
               << " totalSteps=" << totalSteps
               << " in '" << funcName << "'"
               << " (callDepth=" << callDepth << ")"
               << " op=" << opName
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceProcessStepOverflowInFunc(
    ProcessId procId, size_t effectiveMaxProcessSteps, size_t totalSteps,
    llvm::StringRef funcName, bool isLLVMFunction) const {
  llvm::errs() << "[circt-sim] ERROR(PROCESS_STEP_OVERFLOW in func): process "
               << procId << " exceeded " << effectiveMaxProcessSteps
               << " total steps in "
               << (isLLVMFunction ? "LLVM function '" : "function '")
               << funcName << "'"
               << " (totalSteps=" << totalSteps << ")\n";
}

void LLHDProcessInterpreter::maybeTraceProcessActivationStepLimitExceeded(
    ProcessId procId, size_t maxStepsPerActivation) const {
  llvm::errs() << "[circt-sim] WARNING: Process " << procId;
  if (auto *proc = scheduler.getProcess(procId))
    llvm::errs() << " '" << proc->getName() << "'";
  llvm::errs() << " exceeded per-activation step limit ("
               << maxStepsPerActivation << ") - killing\n";
}

void LLHDProcessInterpreter::maybeTraceProcessStepOverflow(
    ProcessId procId, size_t effectiveMaxProcessSteps, size_t totalSteps,
    size_t funcBodySteps, llvm::StringRef currentFuncName,
    llvm::StringRef lastOpName) const {
  llvm::errs() << "[circt-sim] ERROR(PROCESS_STEP_OVERFLOW): process " << procId;
  if (auto *proc = scheduler.getProcess(procId))
    llvm::errs() << " '" << proc->getName() << "'";
  llvm::errs() << " exceeded " << effectiveMaxProcessSteps << " total steps"
               << " (totalSteps=" << totalSteps
               << ", funcBodySteps=" << funcBodySteps << ")";
  if (!currentFuncName.empty())
    llvm::errs() << " [in " << currentFuncName << "]";
  if (!lastOpName.empty())
    llvm::errs() << " (lastOp=" << lastOpName << ")";
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceSvaAssertionFailed(
    llvm::StringRef label, int64_t timeFs, mlir::Location loc) const {
  llvm::errs() << "[circt-sim] SVA assertion failed";
  if (!label.empty())
    llvm::errs() << ": " << label;
  llvm::errs() << " at time " << timeFs
               << " fs (" << loc << ")\n";
}

void LLHDProcessInterpreter::maybeTraceImmediateAssertionFailed(
    llvm::StringRef label, mlir::Location loc) const {
  llvm::errs() << "[circt-sim] Assertion failed";
  if (!label.empty())
    llvm::errs() << ": " << label;
  llvm::errs() << " at " << loc << "\n";
}

void LLHDProcessInterpreter::maybeTraceUvmRunTestEntry(
    ProcessId procId, llvm::StringRef calleeName, uint64_t entryCount) const {
  llvm::errs() << "[UVM-RUN-TEST] count=" << entryCount
               << " proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name=" << proc->getName();
  auto stateIt = processStates.find(procId);
  if (stateIt != processStates.end())
    llvm::errs() << " call_stack=" << stateIt->second.callStack.size()
                 << " waiting=" << (stateIt->second.waiting ? 1 : 0)
                 << " halted=" << (stateIt->second.halted ? 1 : 0);
  llvm::errs() << " callee=" << calleeName << "\n";
}

void LLHDProcessInterpreter::maybeTraceUvmRunTestReentryError(
    ProcessId procId, llvm::StringRef calleeName, uint64_t entryCount) const {
  llvm::errs() << "[circt-sim] error: UVM run_test entered more than once"
               << " (count=" << entryCount
               << ", callee=" << calleeName;
  if (auto *proc = scheduler.getProcess(procId))
    llvm::errs() << ", process=" << proc->getName();
  llvm::errs() << ")\n";
}

void LLHDProcessInterpreter::maybeTraceFuncCallInternalFailureWarning(
    llvm::StringRef calleeName) const {
  static unsigned funcCallWarnCount = 0;
  if (funcCallWarnCount >= 5)
    return;
  ++funcCallWarnCount;
  llvm::errs() << "[circt-sim] WARNING: func.call to '" << calleeName
               << "' failed internally (absorbing)\n";
}

void LLHDProcessInterpreter::maybeTraceSimTerminateTriggered(
    ProcessId procId, mlir::Location loc) const {
  llvm::errs() << "[circt-sim] sim.terminate triggered in process ID "
               << procId << " at ";
  loc.print(llvm::errs());
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceUvmJitPromotionCandidate(
    llvm::StringRef stableKey, uint64_t hitCount, uint64_t hotThreshold,
    size_t budgetRemaining) const {
  llvm::errs() << "[circt-sim] UVM JIT promotion candidate: " << stableKey
               << " hits=" << hitCount
               << " threshold=" << hotThreshold
               << " budget_remaining=" << budgetRemaining << "\n";
}

void LLHDProcessInterpreter::maybeTraceGetNameLoop(
    ProcessId procId, uint64_t seq, const SimTime &now, size_t callDepth,
    size_t callStackSize, bool waiting, uint64_t sameSiteStreak,
    uint64_t getNameCount, uint64_t getFullNameCount,
    uint64_t inGetNameBodyCount, uint64_t randEnabledCount,
    llvm::StringRef calleeName, llvm::StringRef currentFuncName) const {
  llvm::errs() << "[GETNAME-LOOP] seq=" << seq
               << " proc=" << procId
               << " rt=" << now.realTime << " delta=" << now.deltaStep
               << " depth=" << callDepth << " stack=" << callStackSize
               << " waiting=" << (waiting ? 1 : 0)
               << " streak=" << sameSiteStreak
               << " get_name=" << getNameCount
               << " get_full_name=" << getFullNameCount
               << " in_get_body=" << inGetNameBodyCount
               << " rand_enabled=" << randEnabledCount
               << " callee=" << calleeName
               << " func=" << currentFuncName;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " proc_name=" << proc->getName();
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceGetNameLoopLLVM(
    ProcessId procId, uint64_t seq, const SimTime &now, size_t callDepth,
    size_t callStackSize, bool waiting, size_t dynamicStringCount,
    size_t internStringCount, uint64_t sameSiteStreak,
    uint64_t randEnabledCount, uint64_t getNameCount,
    uint64_t inGetNameBodyCount, llvm::StringRef calleeName,
    llvm::StringRef currentFuncName) const {
  llvm::errs() << "[GETNAME-LLVM] seq=" << seq
               << " proc=" << procId
               << " rt=" << now.realTime
               << " delta=" << now.deltaStep
               << " depth=" << callDepth
               << " stack=" << callStackSize
               << " waiting=" << (waiting ? 1 : 0)
               << " dyn_strings=" << dynamicStringCount
               << " intern_strings=" << internStringCount
               << " streak=" << sameSiteStreak
               << " rand_enabled=" << randEnabledCount
               << " get_name=" << getNameCount
               << " in_get_body=" << inGetNameBodyCount
               << " callee=" << calleeName
               << " func=" << currentFuncName;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " proc_name=" << proc->getName();
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathReject(
    ProcessId procId, llvm::StringRef calleeName, llvm::StringRef reason) const {
  llvm::errs() << "[BAUD-FP] reject proc=" << procId
               << " callee=" << calleeName
               << " reason=" << reason << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathNullSelfStall(
    ProcessId procId, llvm::StringRef calleeName) const {
  llvm::errs() << "[BAUD-FP] null-self-stall proc=" << procId
               << " callee=" << calleeName << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathGep(
    llvm::StringRef calleeName, bool baseMatches, size_t dynamicIndexCount,
    llvm::ArrayRef<int32_t> rawIndices) const {
  llvm::errs() << "[BAUD-FP] gep callee=" << calleeName
               << " baseMatches=" << (baseMatches ? 1 : 0)
               << " dynIdx=" << dynamicIndexCount
               << " rawIdx=[";
  for (size_t i = 0; i < rawIndices.size(); ++i) {
    if (i)
      llvm::errs() << ",";
    llvm::errs() << rawIndices[i];
  }
  llvm::errs() << "]\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathMissingFields(
    llvm::StringRef calleeName, unsigned gepSeen, bool sawClockField,
    bool sawOutputField) const {
  llvm::errs() << "[BAUD-FP] missing fields callee=" << calleeName
               << " gepSeen=" << gepSeen
               << " sawClock=" << (sawClockField ? 1 : 0)
               << " sawOutput=" << (sawOutputField ? 1 : 0) << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathBatchParityAdjust(
    ProcessId procId, llvm::StringRef calleeName, uint64_t adjustedEdges) const {
  static unsigned batchParityAdjustCount = 0;
  if (batchParityAdjustCount >= 50)
    return;
  ++batchParityAdjustCount;
  llvm::errs() << "[BAUD-FP] batch-parity-adjust proc=" << procId
               << " callee=" << calleeName
               << " adjustedEdges=" << adjustedEdges << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathBatchMismatch(
    ProcessId procId, llvm::StringRef calleeName, uint64_t pendingEdges,
    bool haveClockSample, bool clockSample, bool clockSampleValid,
    bool lastClockSample, uint64_t nowFs) const {
  static unsigned batchMismatchCount = 0;
  if (batchMismatchCount >= 50)
    return;
  ++batchMismatchCount;
  llvm::errs() << "[BAUD-FP] batch-mismatch proc=" << procId
               << " callee=" << calleeName
               << " pendingEdges=" << pendingEdges
               << " clockSample=" << (haveClockSample ? (clockSample ? 1 : 0) : -1)
               << " lastClock="
               << (clockSampleValid ? (lastClockSample ? 1 : 0) : -1)
               << " nowFs=" << nowFs << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathBatchSchedule(
    ProcessId procId, llvm::StringRef calleeName, uint64_t batchEdges,
    uint64_t edgeInterval, uint64_t stableIntervals, int32_t count,
    int32_t divider) const {
  static unsigned batchScheduleCount = 0;
  if (batchScheduleCount >= 50)
    return;
  ++batchScheduleCount;
  llvm::errs() << "[BAUD-FP] batch-schedule proc=" << procId
               << " callee=" << calleeName
               << " batchEdges=" << batchEdges
               << " edgeFs=" << edgeInterval
               << " stable=" << stableIntervals
               << " count=" << count
               << " divider=" << divider << "\n";
}

void LLHDProcessInterpreter::maybeTraceBaudFastPathHit(
    ProcessId procId, llvm::StringRef calleeName, int32_t divider, bool primed,
    bool countLocalOnly, SignalId clockSignalId, SignalId outputSignalId) const {
  llvm::errs() << "[BAUD-FP] hit proc=" << procId
               << " callee=" << calleeName
               << " divider=" << divider
               << " primed=" << primed
               << " localCountOnly=" << (countLocalOnly ? 1 : 0)
               << " clockSig=" << clockSignalId
               << " outputSig=" << outputSignalId << "\n";
}

void LLHDProcessInterpreter::maybeTraceFirRegUpdate(
    SignalId signalId, const InterpretedValue &newValue, bool posedge) const {
  if (!isFirRegTraceEnabled())
    return;

  auto nameIt = scheduler.getSignalNames().find(signalId);
  llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                ? llvm::StringRef(nameIt->second)
                                : llvm::StringRef("<unknown>");
  llvm::SmallString<64> bits;
  if (newValue.isX())
    bits = "X";
  else
    newValue.getAPInt().toString(bits, 16, false);

  llvm::errs() << "[FIRREG] sig=" << signalId
               << " name=" << sigName
               << " val=0x" << bits
               << " posedge=" << posedge
               << " t=" << scheduler.getCurrentTime().realTime << "\n";
}

LLHDProcessInterpreter::JitRuntimeIndirectSiteData &
LLHDProcessInterpreter::getOrCreateJitRuntimeIndirectSiteData(
    ProcessId procId, mlir::func::CallIndirectOp callOp) {
  auto *siteOp = callOp.getOperation();
  auto [it, inserted] = jitRuntimeIndirectSiteProfiles.try_emplace(
      siteOp, JitRuntimeIndirectSiteData{});
  if (!inserted)
    return it->second;

  auto &site = it->second;
  site.siteId = jitRuntimeIndirectNextSiteId++;

  if (auto parentFunc = siteOp->getParentOfType<mlir::func::FuncOp>())
    site.owner = parentFunc.getSymName().str();
  else if (const Process *proc = scheduler.getProcess(procId))
    site.owner = proc->getName();
  else
    site.owner = "<unknown>";

  {
    std::string locText;
    llvm::raw_string_ostream locOS(locText);
    siteOp->getLoc().print(locOS);
    locOS.flush();
    site.location = std::move(locText);
  }

  return site;
}

void LLHDProcessInterpreter::noteJitRuntimeIndirectResolvedTarget(
    ProcessId procId, mlir::func::CallIndirectOp callOp,
    llvm::StringRef calleeName) {
  if (calleeName.empty())
    return;
  auto &site = getOrCreateJitRuntimeIndirectSiteData(procId, callOp);
  ++site.callsTotal;
  auto [it, inserted] = site.targetCalls.try_emplace(calleeName, 0);
  ++it->second;
  if (inserted)
    ++site.targetSetVersion;
}

void LLHDProcessInterpreter::noteJitRuntimeIndirectUnresolved(
    ProcessId procId, mlir::func::CallIndirectOp callOp) {
  auto &site = getOrCreateJitRuntimeIndirectSiteData(procId, callOp);
  ++site.callsTotal;
  ++site.unresolvedCalls;
}

std::string
LLHDProcessInterpreter::getJitDeoptProcessName(ProcessId procId) const {
  if (const Process *proc = scheduler.getProcess(procId))
    return proc->getName();
  return {};
}

uint64_t LLHDProcessInterpreter::getJitCompileHotThreshold() const {
  return 1;
}

std::optional<LLHDProcessInterpreter::JitRuntimeIndirectSiteProfile>
LLHDProcessInterpreter::lookupJitRuntimeIndirectSiteProfile(
    mlir::func::CallIndirectOp callOp) const {
  auto buildSiteProfile = [&](const auto &siteData) {
    JitRuntimeIndirectSiteProfile site;
    site.siteId = siteData.siteId;
    site.owner = siteData.owner;
    site.location = siteData.location;
    site.callsTotal = siteData.callsTotal;
    site.unresolvedCalls = siteData.unresolvedCalls;
    site.targetSetVersion = siteData.targetSetVersion;

    site.targets.reserve(siteData.targetCalls.size());
    std::vector<std::string> targetNames;
    targetNames.reserve(siteData.targetCalls.size());
    for (const auto &target : siteData.targetCalls) {
      site.targets.push_back(
          JitRuntimeIndirectTargetEntry{target.getKey().str(), target.getValue()});
      targetNames.push_back(target.getKey().str());
    }
    llvm::sort(site.targets, [](const auto &lhs, const auto &rhs) {
      if (lhs.calls != rhs.calls)
        return lhs.calls > rhs.calls;
      return lhs.targetName < rhs.targetName;
    });
    llvm::sort(targetNames);
    if (!targetNames.empty())
      site.targetSetHash = static_cast<uint64_t>(
          llvm::hash_combine_range(targetNames.begin(), targetNames.end()));
    return site;
  };

  if (!callOp)
    return std::nullopt;
  auto it = jitRuntimeIndirectSiteProfiles.find(callOp.getOperation());
  if (it == jitRuntimeIndirectSiteProfiles.end())
    return std::nullopt;
  return buildSiteProfile(it->second);
}

std::vector<LLHDProcessInterpreter::JitRuntimeIndirectSiteProfile>
LLHDProcessInterpreter::getJitRuntimeIndirectSiteProfiles() const {
  auto buildSiteProfile = [&](const auto &siteData) {
    JitRuntimeIndirectSiteProfile site;
    site.siteId = siteData.siteId;
    site.owner = siteData.owner;
    site.location = siteData.location;
    site.callsTotal = siteData.callsTotal;
    site.unresolvedCalls = siteData.unresolvedCalls;
    site.targetSetVersion = siteData.targetSetVersion;

    site.targets.reserve(siteData.targetCalls.size());
    std::vector<std::string> targetNames;
    targetNames.reserve(siteData.targetCalls.size());
    for (const auto &target : siteData.targetCalls) {
      site.targets.push_back(
          JitRuntimeIndirectTargetEntry{target.getKey().str(), target.getValue()});
      targetNames.push_back(target.getKey().str());
    }
    llvm::sort(site.targets, [](const auto &lhs, const auto &rhs) {
      if (lhs.calls != rhs.calls)
        return lhs.calls > rhs.calls;
      return lhs.targetName < rhs.targetName;
    });
    llvm::sort(targetNames);
    if (!targetNames.empty())
      site.targetSetHash = static_cast<uint64_t>(
          llvm::hash_combine_range(targetNames.begin(), targetNames.end()));
    return site;
  };

  std::vector<JitRuntimeIndirectSiteProfile> sites;
  sites.reserve(jitRuntimeIndirectSiteProfiles.size());

  for (const auto &entry : jitRuntimeIndirectSiteProfiles)
    sites.push_back(buildSiteProfile(entry.second));

  llvm::sort(sites, [](const auto &lhs, const auto &rhs) {
    if (lhs.unresolvedCalls != rhs.unresolvedCalls)
      return lhs.unresolvedCalls > rhs.unresolvedCalls;
    if (lhs.callsTotal != rhs.callsTotal)
      return lhs.callsTotal > rhs.callsTotal;
    return lhs.siteId < rhs.siteId;
  });
  return sites;
}

void LLHDProcessInterpreter::dumpProcessStates(llvm::raw_ostream &os) const {
  os << "[circt-sim] Process states:\n";
  for (const auto &entry : processStates) {
    ProcessId procId = entry.first;
    const ProcessExecutionState &state = entry.second;
    const Process *proc = scheduler.getProcess(procId);
    os << "  proc " << procId;
    if (proc)
      os << " '" << proc->getName() << "'";
    llvm::StringRef kind = "process";
    if (state.getCombinationalOp())
      kind = "combinational";
    else if (state.isInitialBlock)
      kind = "initial";
    os << " type=" << kind;
    if (proc)
      os << " state=" << getProcessStateName(proc->getState());
    os << " waiting=" << (state.waiting ? "1" : "0")
       << " halted=" << (state.halted ? "1" : "0")
       << " steps=" << state.totalSteps;
    if (state.lastOp)
      os << " lastOp=" << state.lastOp->getName().getStringRef();
    if (auto funcCall = dyn_cast_or_null<func::CallOp>(state.lastOp))
      os << "(" << funcCall.getCallee() << ")";
    if (auto llvmCall = dyn_cast_or_null<LLVM::CallOp>(state.lastOp))
      if (auto callee = llvmCall.getCallee())
        os << "(" << *callee << ")";
    if (!state.currentFuncName.empty())
      os << " func=" << state.currentFuncName;
    if (state.parentProcessId != InvalidProcessId)
      os << " parent=" << state.parentProcessId;
    if (state.waitConditionQueueAddr != 0)
      os << " waitQ=0x" << llvm::format_hex(state.waitConditionQueueAddr, 16);
    if (!state.callStack.empty()) {
      const auto &top = state.callStack.back();
      os << " callStack=" << state.callStack.size()
         << " topFrame=" << (top.isLLVM() ? "llvm.call" : "func.call");
    }
    if (state.sequencerGetRetryCallOp)
      os << " seqRetry="
         << state.sequencerGetRetryCallOp->getName().getStringRef();
    os << "\n";
  }

  // Print function call profile (top 30 by call count)
  if (profilingEnabled && !funcCallProfile.empty()) {
    os << "[circt-sim] Function call profile (top 30):\n";
    llvm::SmallVector<std::pair<llvm::StringRef, uint64_t>> sorted;
    for (const auto &entry : funcCallProfile)
      sorted.push_back({entry.getKey(), entry.getValue()});
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    for (size_t i = 0; i < std::min(sorted.size(), (size_t)30); ++i)
      os << "  " << sorted[i].second << "x " << sorted[i].first << "\n";

    auto dumpFuncCount = [&](llvm::StringRef name) {
      auto it = funcCallProfile.find(name);
      if (it != funcCallProfile.end())
        os << "  [func-count] " << name << " = " << it->second << "\n";
    };
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::run_phases");
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::schedule_phase");
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::try_put");
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::get");
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::wait_for_objection");
    dumpFuncCount("uvm_pkg::uvm_phase_hopper::process_phase");
  }

  if (profilingEnabled && !uvmFastPathProfile.empty()) {
    os << "[circt-sim] UVM fast-path profile (top 20):\n";
    llvm::SmallVector<std::pair<llvm::StringRef, uint64_t>> sorted;
    for (const auto &entry : uvmFastPathProfile)
      sorted.push_back({entry.getKey(), entry.getValue()});
    llvm::sort(sorted, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });
    for (size_t i = 0; i < std::min(sorted.size(), (size_t)20); ++i)
      os << "  " << sorted[i].second << "x " << sorted[i].first << "\n";
  }

  if (profilingEnabled) {
    uint64_t localFuncCacheHits = 0;
    uint64_t localFuncCacheEntries = 0;
    for (const auto &entry : processStates) {
      localFuncCacheHits += entry.second.funcCacheHits;
      for (const auto &funcEntry : entry.second.funcResultCache)
        localFuncCacheEntries += funcEntry.second.size();
    }

    uint64_t sharedFuncCacheEntries = 0;
    for (const auto &funcEntry : sharedFuncResultCache)
      sharedFuncCacheEntries += funcEntry.second.size();

    os << "[circt-sim] UVM function-result cache: local_hits="
       << localFuncCacheHits << " shared_hits=" << sharedFuncCacheHits
       << " local_entries=" << localFuncCacheEntries
       << " shared_entries=" << sharedFuncCacheEntries << "\n";
  }

  if (!uvmJitPromotedFastPaths.empty()) {
    os << "[circt-sim] UVM JIT promotion candidates ("
       << uvmJitPromotedFastPaths.size() << "):\n";
    for (llvm::StringRef key : uvmJitPromotedFastPaths)
      os << "  " << key << "\n";
    os << "[circt-sim] UVM JIT budget remaining: " << uvmJitPromotionBudget
       << "\n";
  }

  uint64_t fifoItems = 0;
  for (const auto &entry : sequencerItemFifo)
    fifoItems += entry.second.size();
  uint64_t getWaiters = 0;
  for (const auto &entry : sequencerGetWaitersByQueue)
    getWaiters += entry.second.size();
  uint64_t lastDequeuedPending = 0;
  for (const auto &entry : lastDequeuedItem)
    lastDequeuedPending += entry.second.size();
  for (const auto &entry : lastDequeuedItemByProc)
    lastDequeuedPending += entry.second.size();

  if (uvmSeqItemOwnerStores || uvmSeqItemOwnerErases || !itemToSequencer.empty() ||
      !sequencerItemFifo.empty() || !finishItemWaiters.empty() ||
      !itemDoneReceived.empty() || !lastDequeuedItem.empty() ||
      getWaiters != 0) {
    os << "[circt-sim] UVM sequencer native state: item_map_live="
       << itemToSequencer.size() << " item_map_peak=" << uvmSeqItemOwnerPeak
       << " item_map_stores=" << uvmSeqItemOwnerStores
       << " item_map_erases=" << uvmSeqItemOwnerErases
       << " fifo_maps=" << sequencerItemFifo.size()
       << " fifo_items=" << fifoItems
       << " waiters=" << finishItemWaiters.size()
       << " get_waiters=" << getWaiters
       << " done_pending=" << itemDoneReceived.size()
       << " last_dequeued=" << lastDequeuedPending << "\n";
  }

  if (uvmSeqQueueCacheHits || uvmSeqQueueCacheMisses || uvmSeqQueueCacheInstalls ||
      uvmSeqQueueCacheCapacitySkips || uvmSeqQueueCacheEvictions ||
      !portToSequencerQueue.empty()) {
    os << "[circt-sim] UVM sequencer queue cache: hits=" << uvmSeqQueueCacheHits
       << " misses=" << uvmSeqQueueCacheMisses
       << " installs=" << uvmSeqQueueCacheInstalls
       << " entries=" << portToSequencerQueue.size()
       << " capacity_skips=" << uvmSeqQueueCacheCapacitySkips
       << " evictions=" << uvmSeqQueueCacheEvictions << "\n";
  }

  if (uvmSeqQueueCacheMaxEntries || uvmSeqQueueCacheCapacitySkips ||
      uvmSeqQueueCacheEvictions || uvmSeqQueueCacheEvictOnCap) {
    os << "[circt-sim] UVM sequencer queue cache limits: max_entries="
       << uvmSeqQueueCacheMaxEntries
       << " capacity_skips=" << uvmSeqQueueCacheCapacitySkips
       << " evictions=" << uvmSeqQueueCacheEvictions
       << " evict_on_cap=" << (uvmSeqQueueCacheEvictOnCap ? 1 : 0) << "\n";
  }

  // call_indirect resolution cache stats (printed when profile summary enabled
  // and any cache activity occurred).
  if (profileSummaryAtExitEnabled &&
      (ciResolutionCacheInstalls || ciResolutionCacheHits ||
       ciResolutionCacheMisses || ciResolutionCacheDeopts)) {
    os << "[circt-sim] call_indirect resolution cache: installs="
       << ciResolutionCacheInstalls << " hits=" << ciResolutionCacheHits
       << " misses=" << ciResolutionCacheMisses
       << " deopts=" << ciResolutionCacheDeopts
       << " entries=" << callIndirectResolutionCache.size() << "\n";
  }

  // call_indirect direct dispatch cache stats.
  if (profileSummaryAtExitEnabled &&
      !callIndirectDirectDispatchCacheDisabled &&
      (ciDispatchCacheInstalls || ciDispatchCacheHits ||
       ciDispatchCacheMisses || ciDispatchCacheDeopts)) {
    os << "[circt-sim] call_indirect direct dispatch cache: installs="
       << ciDispatchCacheInstalls << " hits=" << ciDispatchCacheHits
       << " misses=" << ciDispatchCacheMisses
       << " deopts=" << ciDispatchCacheDeopts
       << " entries=" << callIndirectDispatchCache.size() << "\n";
  }

  // Function symbol lookup cache stats.
  if (profileSummaryAtExitEnabled &&
      (funcLookupCacheHits || funcLookupCacheMisses ||
       funcLookupCacheNegativeHits)) {
    os << "[circt-sim] function symbol lookup cache: entries="
       << funcLookupCache.size() << " hits=" << funcLookupCacheHits
       << " misses=" << funcLookupCacheMisses
       << " negative_hits=" << funcLookupCacheNegativeHits << "\n";
  }

  // Analysis port terminal cache stats.
  if (profileSummaryAtExitEnabled &&
      (analysisPortTerminalCacheHits || analysisPortTerminalCacheMisses ||
       analysisPortTerminalCacheInvalidations)) {
    os << "[circt-sim] analysis port terminal cache: entries="
       << analysisPortTerminalCache.size()
       << " hits=" << analysisPortTerminalCacheHits
       << " misses=" << analysisPortTerminalCacheMisses
       << " invalidations=" << analysisPortTerminalCacheInvalidations << "\n";
  }

  // Dynamic string registry stats.
  if (profileSummaryAtExitEnabled && dynamicStringMaxEntries > 0) {
    os << "[circt-sim] Dynamic string registry: entries="
       << dynamicStrings.size()
       << " max_entries=" << dynamicStringMaxEntries
       << " registrations=" << dynamicStringRegistrations
       << " updates=" << dynamicStringUpdates
       << " evictions=" << dynamicStringEvictions << "\n";
  }

  if (profileSummaryAtExitEnabled) {
    MemoryStateSnapshot snapshot = collectMemoryStateSnapshot();
    os << "[circt-sim] Memory state: global_blocks=" << snapshot.globalBlocks
       << " global_bytes=" << snapshot.globalBytes
       << " malloc_blocks=" << snapshot.mallocBlocks
       << " malloc_bytes=" << snapshot.mallocBytes
       << " native_blocks=" << snapshot.nativeBlocks
       << " native_bytes=" << snapshot.nativeBytes
       << " process_blocks=" << snapshot.processBlocks
       << " process_bytes=" << snapshot.processBytes
       << " dynamic_strings=" << snapshot.dynamicStrings
       << " dynamic_string_bytes=" << snapshot.dynamicStringBytes
       << " config_db_entries=" << snapshot.configDbEntries
       << " config_db_bytes=" << snapshot.configDbBytes
       << " analysis_conn_ports=" << snapshot.analysisConnPorts
       << " analysis_conn_edges=" << snapshot.analysisConnEdges
       << " seq_fifo_maps=" << snapshot.seqFifoMaps
       << " seq_fifo_items=" << snapshot.seqFifoItems
       << " largest_process=" << snapshot.largestProcessId
       << " largest_process_bytes=" << snapshot.largestProcessBytes << "\n";

    if (memorySampleIntervalSteps > 0) {
      const MemoryStateSnapshot &peakSnapshot =
          (memorySampleCount > 0) ? memoryPeakSnapshot : snapshot;
      uint64_t peakStep = (memorySampleCount > 0) ? memorySamplePeakStep : 0;
      uint64_t peakTotalBytes = (memorySampleCount > 0)
                                    ? memorySamplePeakTotalBytes
                                    : snapshot.totalTrackedBytes();
      llvm::StringRef peakLargestFunc =
          (memorySampleCount > 0 && !memoryPeakLargestProcessFunc.empty())
              ? llvm::StringRef(memoryPeakLargestProcessFunc)
              : llvm::StringRef("-");
      os << "[circt-sim] Memory peak: samples=" << memorySampleCount
         << " sample_interval_steps=" << memorySampleIntervalSteps
         << " peak_step=" << peakStep
         << " peak_total_bytes=" << peakTotalBytes
         << " global_bytes=" << peakSnapshot.globalBytes
         << " malloc_bytes=" << peakSnapshot.mallocBytes
         << " native_bytes=" << peakSnapshot.nativeBytes
         << " process_bytes=" << peakSnapshot.processBytes
         << " dynamic_string_bytes=" << peakSnapshot.dynamicStringBytes
         << " config_db_bytes=" << peakSnapshot.configDbBytes
         << " analysis_conn_edges=" << peakSnapshot.analysisConnEdges
         << " seq_fifo_items=" << peakSnapshot.seqFifoItems
         << " largest_process=" << peakSnapshot.largestProcessId
         << " largest_process_bytes=" << peakSnapshot.largestProcessBytes
         << " largest_process_func=" << peakLargestFunc << "\n";
    }

    if (memorySampleHistory.size() >= 2) {
      const auto &start = memorySampleHistory.front();
      const auto &end = memorySampleHistory.back();
      auto delta = [](uint64_t from, uint64_t to) -> int64_t {
        if (to >= from)
          return static_cast<int64_t>(to - from);
        return -static_cast<int64_t>(from - to);
      };

      os << "[circt-sim] Memory delta window: samples="
         << memorySampleHistory.size()
         << " configured_window=" << memoryDeltaWindowSamples
         << " start_step=" << start.step
         << " end_step=" << end.step
         << " delta_total_bytes="
         << delta(start.snapshot.totalTrackedBytes(),
                  end.snapshot.totalTrackedBytes())
         << " delta_malloc_bytes="
         << delta(start.snapshot.mallocBytes, end.snapshot.mallocBytes)
         << " delta_native_bytes="
         << delta(start.snapshot.nativeBytes, end.snapshot.nativeBytes)
         << " delta_process_bytes="
         << delta(start.snapshot.processBytes, end.snapshot.processBytes)
         << " delta_dynamic_string_bytes="
         << delta(start.snapshot.dynamicStringBytes,
                  end.snapshot.dynamicStringBytes)
         << " delta_config_db_bytes="
         << delta(start.snapshot.configDbBytes, end.snapshot.configDbBytes)
         << " delta_analysis_conn_edges="
         << delta(start.snapshot.analysisConnEdges,
                  end.snapshot.analysisConnEdges)
         << " delta_seq_fifo_items="
         << delta(start.snapshot.seqFifoItems, end.snapshot.seqFifoItems)
         << " delta_global_blocks="
         << delta(start.snapshot.globalBlocks, end.snapshot.globalBlocks)
         << " delta_malloc_blocks="
         << delta(start.snapshot.mallocBlocks, end.snapshot.mallocBlocks)
         << " delta_native_blocks="
         << delta(start.snapshot.nativeBlocks, end.snapshot.nativeBlocks)
         << " delta_process_blocks="
         << delta(start.snapshot.processBlocks, end.snapshot.processBlocks)
         << " delta_dynamic_strings="
         << delta(start.snapshot.dynamicStrings, end.snapshot.dynamicStrings)
         << " delta_config_db_entries="
         << delta(start.snapshot.configDbEntries, end.snapshot.configDbEntries)
         << " delta_analysis_conn_ports="
         << delta(start.snapshot.analysisConnPorts,
                  end.snapshot.analysisConnPorts)
         << " delta_seq_fifo_maps="
         << delta(start.snapshot.seqFifoMaps, end.snapshot.seqFifoMaps)
         << "\n";
    }

    if (memorySummaryTopProcesses > 0 && !processStates.empty()) {
      llvm::SmallVector<std::pair<ProcessId, uint64_t>, 16> ranked;
      ranked.reserve(processStates.size());
      for (const auto &entry : processStates) {
        uint64_t bytes = 0;
        for (const auto &block : entry.second.memoryBlocks)
          bytes += block.second.size;
        ranked.push_back({entry.first, bytes});
      }
      llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
        if (lhs.second != rhs.second)
          return lhs.second > rhs.second;
        return lhs.first < rhs.first;
      });
      size_t limit = std::min<size_t>(memorySummaryTopProcesses, ranked.size());
      for (size_t i = 0; i < limit; ++i) {
        ProcessId pid = ranked[i].first;
        uint64_t bytes = ranked[i].second;
        llvm::StringRef procName = "-";
        if (const Process *proc = scheduler.getProcess(pid)) {
          if (!proc->getName().empty())
            procName = proc->getName();
        }
        llvm::StringRef funcName = "-";
        auto procIt = processStates.find(pid);
        if (procIt != processStates.end() &&
            !procIt->second.currentFuncName.empty())
          funcName = procIt->second.currentFuncName;

        os << "[circt-sim] Memory process top[" << i
           << "]: proc=" << pid
           << " bytes=" << bytes
           << " name=" << procName
           << " func=" << funcName << "\n";
      }
    }
  }

  os.flush();
}

void LLHDProcessInterpreter::dumpOpStats(llvm::raw_ostream &os,
                                         size_t topN) const {
  if (opStats.empty())
    return;

  llvm::SmallVector<std::pair<llvm::StringRef, uint64_t>, 16> entries;
  entries.reserve(opStats.size());
  for (const auto &entry : opStats)
    entries.push_back({entry.getKey(), entry.getValue()});

  llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
    if (lhs.second != rhs.second)
      return lhs.second > rhs.second;
    return lhs.first < rhs.first;
  });

  os << "\n=== Op Stats (top " << topN << ") ===\n";
  size_t limit = std::min(topN, entries.size());
  for (size_t i = 0; i < limit; ++i) {
    os << entries[i].first << ": " << entries[i].second << "\n";
  }
  os << "========================\n";
}

void LLHDProcessInterpreter::dumpProcessStats(llvm::raw_ostream &os,
                                              size_t topN) const {
  if (processStates.empty())
    return;

  struct ProcEntry {
    ProcessId id;
    uint64_t steps;
    uint64_t funcBodySteps;
    size_t opCount;
    uint64_t cacheSkips;
    uint64_t sensCacheHits;
    llvm::StringRef name;
  };

  llvm::SmallVector<ProcEntry, 16> entries;
  entries.reserve(processStates.size());
  for (const auto &entry : processStates) {
    ProcessId procId = entry.first;
    const ProcessExecutionState &state = entry.second;
    llvm::StringRef name;
    if (const Process *proc = scheduler.getProcess(procId))
      name = proc->getName();
    size_t opCount = state.opCount;
    if (opCount == 0) {
      if (auto processOp = state.getProcessOp()) {
        opCount = countRegionOps(processOp.getBody());
      } else if (auto initialOp = state.getInitialOp()) {
        opCount = countRegionOps(initialOp.getBody());
      }
    }
    entries.push_back({procId, state.totalSteps, state.funcBodySteps, opCount,
                       state.cacheSkips, state.waitSensitivityCacheHits, name});
  }

  llvm::sort(entries, [](const ProcEntry &lhs, const ProcEntry &rhs) {
    if (lhs.steps != rhs.steps)
      return lhs.steps > rhs.steps;
    return lhs.id < rhs.id;
  });

  os << "\n=== Process Stats (top " << topN << ") ===\n";
  size_t limit = std::min(topN, entries.size());
  for (size_t i = 0; i < limit; ++i) {
    os << "proc " << entries[i].id;
    if (!entries[i].name.empty())
      os << " '" << entries[i].name << "'";
    os << " steps=" << entries[i].steps
       << " funcSteps=" << entries[i].funcBodySteps
       << " ops=" << entries[i].opCount
       << " skips=" << entries[i].cacheSkips
       << " sens_cache=" << entries[i].sensCacheHits << "\n";
  }
  // Callback dispatch summary.
  if (callbackDispatchCount > 0 || callbackFastResuspendCount > 0) {
    size_t numCallbackProcs = 0;
    size_t numStaticObserved = 0;
    for (const auto &[pid, model] : processExecModels) {
      if (model != ExecModel::Coroutine) {
        ++numCallbackProcs;
        if (model == ExecModel::CallbackStaticObserved)
          ++numStaticObserved;
      }
    }
    os << "callback_procs=" << numCallbackProcs
       << " static_observed=" << numStaticObserved
       << " dispatches=" << callbackDispatchCount
       << " fast_resuspends=" << callbackFastResuspendCount << "\n";
  }
  os << "===========================\n";
}

void LLHDProcessInterpreter::maybeTraceJoinNoneCheck(
    ProcessId procId, ForkId forkId, ProcessId childProcId,
    unsigned pollCount) const {
  if (!traceForkJoinEnabled || pollCount != 0)
    return;
  auto childStateIt = processStates.find(childProcId);
  if (childStateIt == processStates.end())
    return;
  const ProcessExecutionState &childState = childStateIt->second;
  const Process *childProc = scheduler.getProcess(childProcId);
  llvm::errs() << "[JOIN-NONE-CHECK] parent=" << procId
               << " fork=" << forkId << " child=" << childProcId
               << " sched="
               << (childProc ? getProcessStateName(childProc->getState())
                             : "<none>")
               << " waiting=" << (childState.waiting ? 1 : 0)
               << " halted=" << (childState.halted ? 1 : 0)
               << " steps=" << childState.totalSteps << "\n";
}

void LLHDProcessInterpreter::maybeTraceJoinNoneWait(
    ProcessId procId, ForkId forkId, ProcessId childProcId,
    unsigned pollCount) const {
  if (!traceForkJoinEnabled || pollCount > 4)
    return;
  llvm::errs() << "[JOIN-NONE-WAIT] parent=" << procId << " fork=" << forkId
               << " poll=" << pollCount << " child=" << childProcId << "\n";
}

void LLHDProcessInterpreter::maybeTraceJoinNoneResume(
    ProcessId procId, ForkId forkId, unsigned pollCount,
    bool waitingForChildReady) const {
  if (!traceForkJoinEnabled)
    return;
  llvm::errs() << "[JOIN-NONE-RESUME] parent=" << procId
               << " fork=" << forkId << " polls=" << pollCount
               << " waitingForChildReady=" << (waitingForChildReady ? 1 : 0)
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceForkIntercept(
    ProcessId procId, llvm::StringRef joinKind, size_t branchCount,
    uint64_t phaseAddr, bool hadExplicitBlockingMap, bool shapeMatch) const {
  if (!traceForkJoinEnabled)
    return;
  llvm::errs() << "[FORK-INTERCEPT] proc=" << procId
               << " join=" << joinKind
               << " branches=" << branchCount
               << " phase=0x" << llvm::format_hex(phaseAddr, 16)
               << " explicitMap=" << (hadExplicitBlockingMap ? 1 : 0)
               << " shapeMatch=" << (shapeMatch ? 1 : 0) << "\n";
}

void LLHDProcessInterpreter::maybeTraceForkInterceptObjectionWait(
    ProcessId procId, int64_t objectionHandle, int64_t objectionCount,
    uint64_t phaseAddr) const {
  if (!traceForkJoinEnabled)
    return;
  llvm::errs() << "[FORK-INTERCEPT] proc=" << procId
               << " wait_mode=objection_zero handle=" << objectionHandle
               << " count=" << objectionCount
               << " phase=0x" << llvm::format_hex(phaseAddr, 16) << "\n";
}

void LLHDProcessInterpreter::maybeTraceForkCreate(
    ProcessId procId, ForkId forkId, llvm::StringRef joinKind,
    size_t branchCount, uint64_t phaseAddr, bool shapeExec,
    llvm::StringRef parentFunc, llvm::StringRef forkName) const {
  if (!traceForkJoinEnabled)
    return;
  llvm::errs() << "[FORK-CREATE] parent=" << procId
               << " fork=" << forkId
               << " join=" << joinKind
               << " branches=" << branchCount
               << " phase=0x" << llvm::format_hex(phaseAddr, 16)
               << " shapeExec=" << (shapeExec ? 1 : 0)
               << " func=" << parentFunc;
  if (!forkName.empty())
    llvm::errs() << " name=\"" << forkName << "\"";
  llvm::errs() << "\n";
}

void LLHDProcessInterpreter::maybeTraceJoinNoneYield(ProcessId procId,
                                                     ForkId forkId) const {
  if (!traceForkJoinEnabled)
    return;
  llvm::errs() << "[JOIN-NONE-YIELD] parent=" << procId
               << " fork=" << forkId
               << " mode=wait-child-start\n";
}

void LLHDProcessInterpreter::maybeTraceProcessFinalize(ProcessId procId,
                                                       bool killed) const {
  static bool traceFinalize = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_FINALIZE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (!traceFinalize)
    return;

  const Process *traceProc = scheduler.getProcess(procId);
  std::string procName = traceProc ? traceProc->getName() : "<null>";
  const char *schedState =
      traceProc ? getProcessStateName(traceProc->getState()) : "<none>";
  llvm::StringRef curFunc = "-";
  auto procStateItForTrace = processStates.find(procId);
  if (procStateItForTrace != processStates.end())
    curFunc = procStateItForTrace->second.currentFuncName;

  llvm::errs() << "[PROC-FINALIZE] proc=" << procId
               << " name=" << procName
               << " killed=" << (killed ? 1 : 0)
               << " sched_state=" << schedState
               << " func=" << curFunc << "\n";
}

void LLHDProcessInterpreter::maybeTraceWaitSensitivityList(
    ProcessId procId, llvm::StringRef tag,
    const SensitivityList &waitList) const {
  static bool traceWaitSensitivity = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_WAIT_SENS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (!traceWaitSensitivity)
    return;

  llvm::errs() << "[WAIT-SENS] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name='" << proc->getName() << "'";
  llvm::errs() << " tag=" << tag << " entries=" << waitList.size() << "\n";
  for (const auto &entry : waitList.getEntries()) {
    llvm::StringRef sigName = "<unknown>";
    auto nameIt = signalIdToName.find(entry.signalId);
    if (nameIt != signalIdToName.end())
      sigName = nameIt->second;
    llvm::errs() << "[WAIT-SENS]   sig=" << entry.signalId << " (" << sigName
                 << ") edge=" << getEdgeTypeName(entry.edge) << "\n";
  }
}

void LLHDProcessInterpreter::maybeTraceWaitEventCache(
    ProcessId procId, llvm::StringRef action, mlir::Operation *waitEventOp,
    const SensitivityList &waitList) const {
  if (!isWaitEventCacheTraceEnabled())
    return;

  llvm::SmallString<256> listText;
  listText += "[";
  unsigned count = 0;
  for (const auto &entry : waitList.getEntries()) {
    if (count)
      listText += ",";
    ++count;
    listText += "sig=";
    listText += llvm::utostr(entry.signalId);
    listText += ":";
    listText += getEdgeTypeName(entry.edge);
    auto nameIt = signalIdToName.find(entry.signalId);
    if (nameIt != signalIdToName.end()) {
      listText += ":";
      listText += nameIt->second;
    }
    if (count >= 4 && waitList.size() > count) {
      listText += ",...";
      break;
    }
  }
  listText += "]";

  SimTime now = scheduler.getCurrentTime();
  llvm::errs() << "[WAIT-EVENT-CACHE] " << action << " proc=" << procId
               << " t=" << now.realTime << " d=" << now.deltaStep
               << " entries=" << waitList.size()
               << " op=0x"
               << llvm::format_hex(reinterpret_cast<uintptr_t>(waitEventOp), 18)
               << " list=" << listText << "\n";
}

void LLHDProcessInterpreter::maybeTraceWaitEventNoop(
    ProcessId procId, llvm::StringRef funcName,
    mlir::Operation *waitEventOp) const {
  if (!isWaitEventNoopTraceEnabled())
    return;

  llvm::errs() << "[WAIT-EVENT-NOOP] proc=" << procId;
  if (const Process *proc = scheduler.getProcess(procId))
    llvm::errs() << " name=" << proc->getName();
  llvm::errs() << " func=" << funcName
               << " op=0x"
               << llvm::format_hex(reinterpret_cast<uintptr_t>(waitEventOp), 18)
               << "\n";
}

void LLHDProcessInterpreter::maybeTraceForkTerminator(ProcessId procId) const {
  if (!traceForkJoinEnabled)
    return;
  auto stIt = processStates.find(procId);
  if (stIt == processStates.end())
    return;
  const ProcessExecutionState &ps = stIt->second;
  if (ps.parentProcessId == InvalidProcessId || procId <= 100)
    return;

  static unsigned forkTermDiagCount = 0;
  if (forkTermDiagCount >= 100)
    return;
  ++forkTermDiagCount;

  const Process *proc = scheduler.getProcess(procId);
  std::string procName = proc ? proc->getName() : "??";

  std::string parentFunc;
  auto parentIt = processStates.find(ps.parentProcessId);
  if (parentIt != processStates.end())
    parentFunc = parentIt->second.currentFuncName;

  std::string parentName;
  if (const Process *parentProc = scheduler.getProcess(ps.parentProcessId))
    parentName = parentProc->getName();

  llvm::errs() << "[FORK-TERM] proc=" << procId
               << " name=" << procName
               << " steps=" << ps.totalSteps
               << " parent=" << ps.parentProcessId
               << " parentName=" << parentName
               << " parentFunc=" << parentFunc << "\n";
}

void LLHDProcessInterpreter::maybeTraceJoinAnyImmediate(ProcessId procId,
                                                        ForkId forkId) const {
  if (!traceForkJoinEnabled || procId <= 100)
    return;

  static unsigned joinAnyImmDiagCount = 0;
  if (joinAnyImmDiagCount >= 30)
    return;
  ++joinAnyImmDiagCount;

  const auto *group = forkJoinManager.getForkGroup(forkId);
  unsigned childCount = group ? group->childProcesses.size() : 0;
  unsigned completedCount = group ? group->completedCount : 0;
  const Process *proc = scheduler.getProcess(procId);
  std::string procName = proc ? proc->getName() : "??";
  llvm::errs() << "[JOIN-ANY-IMM] proc=" << procId
               << " name=" << procName
               << " forkId=" << forkId
               << " children=" << childCount
               << " completed=" << completedCount << "\n";

  if (!group)
    return;
  for (ProcessId childId : group->childProcesses) {
    const Process *childProc = scheduler.getProcess(childId);
    bool terminated =
        childProc && childProc->getState() == ProcessState::Terminated;
    auto childStateIt = processStates.find(childId);
    size_t childSteps =
        childStateIt != processStates.end() ? childStateIt->second.totalSteps
                                            : 0;
    std::string childName = childProc ? childProc->getName() : "??";
    llvm::errs() << "  child=" << childId
                 << " name=" << childName
                 << " terminated=" << terminated
                 << " steps=" << childSteps << "\n";
  }
}

void LLHDProcessInterpreter::maybeTraceDisableForkBegin(
    ProcessId procId, size_t forkCount, bool deferredFire) const {
  if (!isDisableForkTraceEnabled())
    return;
  if (deferredFire) {
    llvm::errs() << "[DISABLE-FORK-DEFER-FIRE] parent=" << procId
                 << " fork_count=" << forkCount << "\n";
    return;
  }
  llvm::errs() << "[DISABLE-FORK] parent=" << procId
               << " fork_count=" << forkCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceDisableForkDeferredPoll(
    ProcessId procId, ForkId forkId, ProcessId childProcId,
    unsigned pollCount) const {
  if (!isDisableForkTraceEnabled())
    return;
  llvm::errs() << "[DISABLE-FORK-DEFER-POLL] parent=" << procId
               << " fork=" << forkId << " child=" << childProcId
               << " poll=" << pollCount << "\n";
}

void LLHDProcessInterpreter::maybeTraceDisableForkDeferredArm(
    ProcessId procId, ForkId forkId, ProcessId childProcId,
    ProcessState childSchedState, uint64_t token) const {
  if (!isDisableForkTraceEnabled())
    return;
  llvm::errs() << "[DISABLE-FORK-DEFER] parent=" << procId
               << " fork=" << forkId << " child=" << childProcId
               << " child_state=" << getProcessStateName(childSchedState)
               << " token=" << token << "\n";
}

void LLHDProcessInterpreter::maybeTraceDisableForkChild(
    ProcessId procId, ForkId forkId, ProcessId childProcId,
    llvm::StringRef mode) const {
  if (!isDisableForkTraceEnabled())
    return;
  auto childStateIt = processStates.find(childProcId);
  if (childStateIt == processStates.end())
    return;

  const ProcessExecutionState &childState = childStateIt->second;
  const Process *childProc = scheduler.getProcess(childProcId);
  llvm::StringRef childFunc = childState.currentFuncName;
  llvm::errs() << "  [DISABLE-FORK-CHILD] parent=" << procId
               << " fork=" << forkId << " child=" << childProcId
               << " state="
               << (childProc ? getProcessStateName(childProc->getState())
                             : "<none>")
               << " waiting=" << (childState.waiting ? 1 : 0)
               << " halted=" << (childState.halted ? 1 : 0)
               << " steps=" << childState.totalSteps
               << " mode=" << mode
               << " func=" << childFunc << "\n";
}

void LLHDProcessInterpreter::traceI3CForkRuntimeEvent(
    llvm::StringRef tag, ProcessId parentProcId, ProcessId childProcId,
    ForkId forkId, llvm::StringRef mode) {
  if (!traceI3CForkRuntimeEnabled)
    return;

  auto emitProc = [&](llvm::StringRef label, ProcessId pid) {
    llvm::errs() << " " << label << "=" << pid;
    auto stateIt = processStates.find(pid);
    if (stateIt == processStates.end()) {
      llvm::errs() << " " << label << "_state=<missing>";
      return;
    }
    const auto &st = stateIt->second;
    const Process *proc = scheduler.getProcess(pid);
    llvm::errs() << " " << label
                 << "_sched="
                 << (proc ? getProcessStateName(proc->getState()) : "<none>")
                 << " " << label << "_halted=" << (st.halted ? 1 : 0)
                 << " " << label << "_waiting=" << (st.waiting ? 1 : 0)
                 << " " << label << "_call_stack=" << st.callStack.size()
                 << " " << label << "_func=" << st.currentFuncName;
    if (st.currentBlock) {
      llvm::errs() << " " << label
                   << "_current_block_ops="
                   << st.currentBlock->getOperations().size();
      if (st.currentOp == st.currentBlock->end())
        llvm::errs() << " " << label << "_current_op=<end>";
      else
        llvm::errs() << " " << label
                     << "_current_op=" << st.currentOp->getName().getStringRef();
    } else {
      llvm::errs() << " " << label << "_current_block=<null>";
    }
    if (st.destBlock) {
      llvm::errs() << " " << label
                   << "_dest_block_ops=" << st.destBlock->getOperations().size()
                   << " " << label << "_dest_args=" << st.destOperands.size()
                   << " " << label
                   << "_resume_at_current_op=" << (st.resumeAtCurrentOp ? 1 : 0);
    } else {
      llvm::errs() << " " << label << "_dest_block=<null>";
    }
  };

  SimTime now = scheduler.getCurrentTime();
  llvm::errs() << "[I3C-FORK-RUNTIME] tag=" << tag << " fork=" << forkId
               << " t=" << now.realTime << " d=" << now.deltaStep;
  if (!mode.empty())
    llvm::errs() << " mode=" << mode;
  emitProc("parent", parentProcId);
  if (childProcId != InvalidProcessId)
    emitProc("child", childProcId);
  llvm::errs() << "\n";
}
