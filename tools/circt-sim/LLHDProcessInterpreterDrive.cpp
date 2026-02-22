//===- LLHDProcessInterpreterDrive.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains LLHDProcessInterpreter drive/update handlers extracted
// from LLHDProcessInterpreter.cpp.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "LLHDProcessInterpreterStorePatterns.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

namespace {

// Distinct-driver resolution must differentiate the same DriveOp materialized
// across multiple hw.instance contexts. Using only DriveOp* aliases drivers
// from sibling instances and can collapse bidirectional inout behavior.
static uint64_t getDistinctContinuousDriverId(llhd::DriveOp driveOp,
                                              InstanceId instanceId) {
  return static_cast<uint64_t>(llvm::hash_combine(
      reinterpret_cast<uintptr_t>(driveOp.getOperation()),
      static_cast<uint64_t>(instanceId)));
}

static uint64_t getTriStateDriveSourceCacheKey(llhd::DriveOp driveOp,
                                               InstanceId instanceId) {
  return static_cast<uint64_t>(llvm::hash_combine(
      reinterpret_cast<uintptr_t>(driveOp.getOperation()),
      static_cast<uint64_t>(instanceId), 0x5452495354415445ULL));
}

// Continuous assignments lowered from tri-state behavior use drive enables.
// When disabled on strength-resolved nets, the driver must release (high-Z)
// instead of holding the last driven value.
static InterpretedValue
getDisabledContinuousDriveValue(const ProcessScheduler &scheduler,
                                SignalId signalId) {
  unsigned width = scheduler.getSignalValue(signalId).getWidth();
  if (scheduler.getSignalEncoding(signalId) == SignalEncoding::FourStateStruct)
    return InterpretedValue(llvm::APInt::getAllOnes(width));
  return InterpretedValue::makeX(width);
}

} // namespace

void LLHDProcessInterpreter::normalizeImplicitZDriveStrength(
    SignalId signalId, const InterpretedValue &driveVal,
    DriveStrength &strength0, DriveStrength &strength1) const {
  if (driveVal.isX())
    return;
  if (scheduler.getSignalEncoding(signalId) != SignalEncoding::FourStateStruct)
    return;

  APInt bits = driveVal.getAPInt();
  unsigned width = bits.getBitWidth();
  if (width < 2 || (width % 2) != 0)
    return;

  unsigned logicalWidth = width / 2;
  APInt unknownBits = bits.extractBits(logicalWidth, 0);
  APInt valueBits = bits.extractBits(logicalWidth, logicalWidth);
  if (unknownBits.isAllOnes() && valueBits.isAllOnes()) {
    strength0 = DriveStrength::HighZ;
    strength1 = DriveStrength::HighZ;
  }
}

SignalId LLHDProcessInterpreter::resolveTriStateDriveSourceFieldSignal(
    llhd::DriveOp driveOp, InstanceId instanceId) {
  uint64_t cacheKey = getTriStateDriveSourceCacheKey(driveOp, instanceId);
  auto cacheIt = triStateDriveSourceFieldCache.find(cacheKey);
  if (cacheIt != triStateDriveSourceFieldCache.end())
    return cacheIt->second;

  ProcessExecutionState tempState;
  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId) || tempProcId == InvalidProcessId)
    tempProcId = nextTempProcId++;
  processStates[tempProcId] = std::move(tempState);

  auto cleanup = llvm::make_scope_exit([&]() { processStates.erase(tempProcId); });
  auto &tmpState = processStates[tempProcId];
  tmpState.valueMap = moduleInitValueMap;
  if (activeProcessId != InvalidProcessId) {
    auto activeIt = processStates.find(activeProcessId);
    if (activeIt != processStates.end()) {
      for (const auto &entry : activeIt->second.valueMap)
        tmpState.valueMap.try_emplace(entry.first, entry.second);
    }
  }

  auto resolveAddr = [&](Value addrValue) -> uint64_t {
    InterpretedValue addrVal = getValue(tempProcId, addrValue);
    if (addrVal.isX())
      return 0;
    return addrVal.getUInt64();
  };

  SignalId sourceFieldSigId = 0;
  uint64_t sourceAddr = 0;
  if (matchFourStateStructCreateLoad(driveOp.getValue(), resolveAddr,
                                     sourceAddr) &&
      sourceAddr != 0) {
    auto fieldIt = interfaceFieldSignals.find(sourceAddr);
    if (fieldIt != interfaceFieldSignals.end())
      sourceFieldSigId = fieldIt->second;
  }

  triStateDriveSourceFieldCache[cacheKey] = sourceFieldSigId;
  return sourceFieldSigId;
}

bool LLHDProcessInterpreter::tryEvaluateTriStateDestDriveValue(
    llhd::DriveOp driveOp, SignalId targetSigId, InterpretedValue &driveVal) {
  if (interfaceTriStateRules.empty())
    return false;

  SignalId sourceFieldSigId =
      resolveTriStateDriveSourceFieldSignal(driveOp, activeInstanceId);
  if (sourceFieldSigId == 0)
    return false;

  const InterfaceTriStateRule *rule = nullptr;
  for (const auto &candidate : interfaceTriStateRules) {
    if (candidate.destSigId == sourceFieldSigId) {
      rule = &candidate;
      break;
    }
  }
  if (!rule)
    return false;

  auto getSignalValue = [&](SignalId sigId) -> InterpretedValue {
    auto pendingIt = pendingEpsilonDrives.find(sigId);
    if (pendingIt != pendingEpsilonDrives.end())
      return pendingIt->second;
    return InterpretedValue::fromSignalValue(scheduler.getSignalValue(sigId));
  };

  InterpretedValue condVal = getSignalValue(rule->condSigId);
  bool condTrue = false;
  if (!condVal.isX() && rule->condBitIndex < condVal.getWidth())
    condTrue = condVal.getAPInt()[rule->condBitIndex];

  InterpretedValue selectedVal =
      condTrue ? getSignalValue(rule->srcSigId) : rule->elseValue;

  unsigned targetWidth = scheduler.getSignalValue(targetSigId).getWidth();
  if (selectedVal.isX()) {
    driveVal = InterpretedValue::makeX(targetWidth);
    return true;
  }

  APInt bits = selectedVal.getAPInt();
  if (bits.getBitWidth() < targetWidth)
    bits = bits.zext(targetWidth);
  else if (bits.getBitWidth() > targetWidth)
    bits = bits.trunc(targetWidth);
  driveVal = InterpretedValue(bits);
  return true;
}

void LLHDProcessInterpreter::executeContinuousAssignment(
    llhd::DriveOp driveOp) {
  // Get the signal being driven
  SignalId targetSigId = getSignalId(driveOp.getSignal());
  if (targetSigId == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Error: Unknown signal in continuous assignment\n");
    return;
  }

  // Suppress continuous assignments to forced signals (IEEE 1800-2017 §10.6.2)
  // Evaluate and save the would-be value so release restores it.
  if (forcedSignals.contains(targetSigId)) {
    InterpretedValue suppressedVal = evaluateContinuousValue(driveOp.getValue());
    forcedSignalSavedValues[targetSigId] = suppressedVal;
    LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment suppressed: signal "
                            << targetSigId << " is forced\n");
    return;
  }

  // Evaluate the drive value by interpreting the defining operation chain
  // We use process ID 0 as a dummy since continuous assignments don't have
  // their own process state - they evaluate values directly from signal state
  bool releaseDisabledDrive = false;
  if (driveOp.getEnable()) {
    InterpretedValue enableVal = evaluateContinuousValue(driveOp.getEnable());
    if (enableVal.isX() || enableVal.getUInt64() == 0) {
      if (!distinctContinuousDriverSignals.contains(targetSigId) ||
          scheduler.getSignalEncoding(targetSigId) !=
              SignalEncoding::FourStateStruct) {
        LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment disabled\n");
        return;
      }
      releaseDisabledDrive = true;
      LLVM_DEBUG(
          llvm::dbgs() << "  Continuous assignment disabled, releasing driver\n");
    }
  }
  InterpretedValue driveVal =
      releaseDisabledDrive
          ? getDisabledContinuousDriveValue(scheduler, targetSigId)
          : [&]() {
              InterpretedValue triStateDriveVal;
              if (tryEvaluateTriStateDestDriveValue(driveOp, targetSigId,
                                                    triStateDriveVal))
                return triStateDriveVal;
              return evaluateContinuousValue(driveOp.getValue());
            }();

  {
    static bool traceContExec = []() {
      const char *env = std::getenv("CIRCT_SIM_TRACE_CONT_EXEC");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    if (traceContExec) {
      auto nameIt = scheduler.getSignalNames().find(targetSigId);
      llvm::StringRef sigName = nameIt != scheduler.getSignalNames().end()
                                    ? llvm::StringRef(nameIt->second)
                                    : llvm::StringRef("<unknown>");
      llvm::SmallString<64> bits;
      if (driveVal.isX())
        bits = "X";
      else
        driveVal.getAPInt().toString(bits, 16, false);
      llvm::errs() << "[CONT-EXEC] sig=" << targetSigId
                   << " name=" << sigName
                   << " val=0x" << bits
                   << " t=" << scheduler.getCurrentTime().realTime
                   << " d=" << scheduler.getCurrentTime().deltaStep
                   << "\n";
    }
  }

  // Get the delay time
  SimTime delay;
  if (auto timeOp = driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>()) {
    delay = convertTime(timeOp.getValueAttr());
  } else {
    // Default to epsilon delay
    delay = SimTime(0, 0, 1);
  }

  // Calculate the target time
  SimTime currentTime = scheduler.getCurrentTime();
  SimTime targetTime = currentTime.advanceTime(delay.realTime);
  if (delay.deltaStep > 0) {
    targetTime.deltaStep = currentTime.deltaStep + delay.deltaStep;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment: scheduling update to signal "
                          << targetSigId << " value="
                          << (driveVal.isX() ? "X"
                                             : std::to_string(driveVal.getUInt64()))
                          << " at time " << targetTime.realTime << " fs\n");

  {
    static bool traceI3CDrives = []() {
      const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_DRIVES");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    if (traceI3CDrives) {
      auto nameIt = scheduler.getSignalNames().find(targetSigId);
      if (nameIt != scheduler.getSignalNames().end()) {
        llvm::StringRef sigName = nameIt->second;
        if (sigName.contains("I3C_SCL") || sigName.contains("I3C_SDA")) {
          llvm::SmallString<64> bits;
          if (driveVal.isX())
            bits = "X";
          else
            driveVal.getAPInt().toString(bits, 2, false);
          llvm::StringRef funcName = "-";
          if (activeProcessState && !activeProcessState->currentFuncName.empty())
            funcName = activeProcessState->currentFuncName;
          ProcessId traceProcId = activeProcessId;
          llvm::errs() << "[I3C-DRV] t=" << currentTime.realTime << " d="
                       << currentTime.deltaStep << " sig=" << targetSigId
                       << " (" << sigName << ") val=" << bits
                       << " proc=" << traceProcId << " func=" << funcName
                       << " inst=" << activeInstanceId
                       << "\n";
        }
      }
    }
  }

  // Schedule the signal update.
  // Use updateSignalWithStrength to support multi-driver resolution (wand/wor).
  // Strength-sensitive drives use IDs keyed by DriveOp and active instance.
  SignalValue newVal = driveVal.toSignalValue();

  // Extract drive strength from the drive operation.
  DriveStrength strength0 = DriveStrength::Strong;
  DriveStrength strength1 = DriveStrength::Strong;
  if (auto s0Attr = driveOp.getStrength0Attr())
    strength0 = static_cast<DriveStrength>(
        static_cast<uint8_t>(s0Attr.getValue()));
  if (auto s1Attr = driveOp.getStrength1Attr())
    strength1 = static_cast<DriveStrength>(
        static_cast<uint8_t>(s1Attr.getValue()));

  if (releaseDisabledDrive) {
    strength0 = DriveStrength::HighZ;
    strength1 = DriveStrength::HighZ;
  } else {
    normalizeImplicitZDriveStrength(targetSigId, driveVal, strength0,
                                    strength1);
  }

  // Strength-sensitive nets (e.g. pullups/open-drain) need per-drive IDs so
  // multiple continuous assignments resolve correctly. Keep legacy
  // last-write-wins IDs for non-strength-sensitive paths.
  uint64_t driverId =
      distinctContinuousDriverSignals.contains(targetSigId)
          ? getDistinctContinuousDriverId(driveOp, activeInstanceId)
          : static_cast<uint64_t>(targetSigId);

  {
    static bool traceI3CDrives = []() {
      const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_DRIVES");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    if (traceI3CDrives) {
      auto nameIt = scheduler.getSignalNames().find(targetSigId);
      if (nameIt != scheduler.getSignalNames().end()) {
        llvm::StringRef sigName = nameIt->second;
        if (sigName.contains("I3C_SCL") || sigName.contains("I3C_SDA")) {
          llvm::SmallString<64> bits;
          if (driveVal.isX())
            bits = "X";
          else
            driveVal.getAPInt().toString(bits, 2, false);
          llvm::errs() << "[I3C-DRV-ID] t=" << currentTime.realTime << " d="
                       << currentTime.deltaStep << " sig=" << targetSigId
                       << " (" << sigName << ") val=" << bits
                       << " driver=" << driverId
                       << " inst=" << activeInstanceId
                       << " loc=" << driveOp.getLoc() << "\n";
        }
      }
    }
  }

  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::Active,
      Event([this, targetSigId, driverId, newVal, strength0, strength1]() {
        if (forcedSignals.contains(targetSigId)) {
          forcedSignalSavedValues[targetSigId] =
              InterpretedValue::fromSignalValue(newVal);
          pendingEpsilonDrives.erase(targetSigId);
          return;
        }
        scheduler.updateSignalWithStrength(targetSigId, driverId, newVal,
                                           strength0, strength1);
        // Clear any stale pending epsilon drive so future probes see
        // the scheduler's committed value rather than a stale pending.
        pendingEpsilonDrives.erase(targetSigId);
      }));
}

std::optional<std::pair<int64_t, unsigned>>
LLHDProcessInterpreter::detectArrayElementDrive(llhd::DriveOp driveOp) {
  // Check cache first.
  auto *cacheKey = driveOp.getOperation();
  auto cacheIt = arrayElementDriveCache.find(cacheKey);
  if (cacheIt != arrayElementDriveCache.end()) {
    if (cacheIt->second.elementIndex < 0)
      return std::nullopt;
    return std::make_pair(cacheIt->second.elementIndex,
                          cacheIt->second.elementBitWidth);
  }

  // Default: not an array element drive.
  auto negativeEntry = [&]() -> std::optional<std::pair<int64_t, unsigned>> {
    arrayElementDriveCache[cacheKey] = {-1, 0};
    return std::nullopt;
  };

  // The drive value must be a result of an llhd.process.
  mlir::Value driveValue = driveOp.getValue();
  auto processOp = driveValue.getDefiningOp<llhd::ProcessOp>();
  if (!processOp)
    return negativeEntry();

  // Get the result index.
  auto result = cast<mlir::OpResult>(driveValue);
  unsigned resultIndex = result.getResultNumber();

  // Find the WaitOp in the process body that yields values.
  llhd::WaitOp waitOp = nullptr;
  processOp.walk([&](llhd::WaitOp w) {
    if (!w.getYieldOperands().empty())
      waitOp = w;
  });
  if (!waitOp || waitOp.getYieldOperands().size() <= resultIndex)
    return negativeEntry();

  // Check if the yield operand at the matching index is hw.array_inject.
  mlir::Value yieldVal = waitOp.getYieldOperands()[resultIndex];
  auto injectOp = yieldVal.getDefiningOp<hw::ArrayInjectOp>();
  if (!injectOp)
    return negativeEntry();

  // The index must be a constant.
  auto constIdxOp = injectOp.getIndex().getDefiningOp<hw::ConstantOp>();
  if (!constIdxOp)
    return negativeEntry();

  // Get element index and bit width.
  int64_t elemIndex = constIdxOp.getValue().getZExtValue();
  auto arrayType = cast<hw::ArrayType>(injectOp.getType());
  int64_t elemBitWidthSigned = hw::getBitWidth(arrayType.getElementType());
  if (elemBitWidthSigned <= 0)
    return negativeEntry();
  unsigned elemBitWidth = static_cast<unsigned>(elemBitWidthSigned);

  // Store in cache and return.
  arrayElementDriveCache[cacheKey] = {elemIndex, elemBitWidth};
  LLVM_DEBUG(llvm::dbgs() << "  Detected array element drive: index="
                          << elemIndex << " elemWidth=" << elemBitWidth << "\n");
  return std::make_pair(elemIndex, elemBitWidth);
}

