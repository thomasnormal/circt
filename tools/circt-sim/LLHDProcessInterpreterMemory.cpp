//===- LLHDProcessInterpreterMemory.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains LLHDProcessInterpreter memory-model handlers extracted
// from LLHDProcessInterpreter.cpp.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "LLHDProcessInterpreterStorePatterns.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

namespace {

/// Safe insertBits that clamps instead of asserting when the sub-value
/// extends beyond the target's bit width.
static void safeInsertBits(llvm::APInt &target, const llvm::APInt &source,
                           unsigned bitPosition) {
  unsigned subWidth = source.getBitWidth();
  unsigned targetWidth = target.getBitWidth();
  if (bitPosition >= targetWidth) {
    LLVM_DEBUG(llvm::dbgs() << "insertBits clamp: bitPosition (" << bitPosition
                            << ") >= targetWidth (" << targetWidth
                            << "), skipping\n");
    return;
  }
  if (subWidth + bitPosition > targetWidth) {
    // Truncate the source to fit
    unsigned availBits = targetWidth - bitPosition;
    LLVM_DEBUG(llvm::dbgs() << "insertBits clamp: subWidth " << subWidth
                            << " -> " << availBits
                            << " (targetWidth=" << targetWidth
                            << " bitPos=" << bitPosition << ")\n");
    target.insertBits(source.trunc(availBits), bitPosition);
    return;
  }
  target.insertBits(source, bitPosition);
}

// When a {ptr, i64} aggregate is loaded from memory, treat it as a dynamic
// buffer descriptor and register the pointed-to native memory range. This
// allows subsequent GEP/load/store operations on that pointer to avoid
// spurious X propagation if the allocation happened outside interpreter-owned
// allocas/malloc blocks.
static void maybeRegisterNativeBlockFromPtrLenStruct(
    Type loadedType, const InterpretedValue &loadedValue,
    llvm::DenseMap<uint64_t, size_t> &nativeMemoryBlocks) {
  constexpr uint64_t kMinReasonablePtr = 0x10000ULL;
  constexpr uint64_t kMaxCanonicalUserPtr = 0x0000FFFFFFFFFFFFULL;
  constexpr uint64_t kMaxReasonableLen = 1ULL << 30; // 1 GiB hard cap.

  auto structTy = dyn_cast<LLVM::LLVMStructType>(loadedType);
  if (!structTy)
    return;
  auto body = structTy.getBody();
  if (body.size() != 2)
    return;
  if (!isa<LLVM::LLVMPointerType>(body[0]))
    return;
  auto lenTy = dyn_cast<IntegerType>(body[1]);
  if (!lenTy || lenTy.getWidth() != 64)
    return;
  if (loadedValue.isX() || loadedValue.getWidth() < 128)
    return;

  APInt bits = loadedValue.getAPInt();
  uint64_t ptr = bits.extractBits(64, 0).getZExtValue();
  uint64_t len = bits.extractBits(64, 64).getZExtValue();
  if (ptr == 0 || len == 0)
    return;
  // Ignore clearly invalid pointer/length pairs from uninitialized payloads.
  if (ptr < kMinReasonablePtr || ptr > kMaxCanonicalUserPtr)
    return;
  if (len > kMaxReasonableLen)
    return;
  if (ptr + len < ptr)
    return;

  auto it = nativeMemoryBlocks.find(ptr);
  if (it == nativeMemoryBlocks.end())
    nativeMemoryBlocks[ptr] = static_cast<size_t>(len);
  else
    it->second = std::max<size_t>(it->second, static_cast<size_t>(len));
}

} // namespace

LogicalResult LLHDProcessInterpreter::interpretLLVMAlloca(
    ProcessId procId, LLVM::AllocaOp allocaOp) {
  auto &state = processStates[procId];

  // Get the element type and array size
  Type elemType = allocaOp.getElemType();
  InterpretedValue arraySizeVal = getValue(procId, allocaOp.getArraySize());

  uint64_t arraySize = 1;
  if (!arraySizeVal.isX())
    arraySize = arraySizeVal.getUInt64();

  // Calculate total size in bytes
  unsigned elemSize = getLLVMTypeSize(elemType);
  size_t totalSize = elemSize * arraySize;

  // Create a memory block
  MemoryBlock block(totalSize, getTypeWidth(elemType));
  block.initialized = true;  // Alloca memory is zero-initialized and readable

  // Check if this alloca is at module level (not inside an llhd.process,
  // func.func, or llvm.func). Allocas inside functions should be process-local
  // even if the function is called from a global constructor, because they
  // need to be found via the process's valueMap when findMemoryBlockByAddress
  // is called.
  bool isModuleLevel = !allocaOp->getParentOfType<llhd::ProcessOp>() &&
                       !allocaOp->getParentOfType<mlir::func::FuncOp>() &&
                       !allocaOp->getParentOfType<LLVM::LLVMFuncOp>() &&
                       !allocaOp->getParentOfType<seq::InitialOp>();

  if (isModuleLevel) {
    // Store in module-level allocas (accessible by all processes)
    moduleLevelAllocas[allocaOp.getResult()] = std::move(block);
  } else {
    // Store in process-local memory
    state.memoryBlocks[allocaOp.getResult()] = std::move(block);
  }

  // Assign a unique address to this pointer (for tracking purposes)
  // Use globalNextAddress to ensure no overlap between module-level
  // and process-level allocas.
  uint64_t addr = globalNextAddress;
  globalNextAddress += totalSize;

  // Store the pointer value (the address)
  setValue(procId, allocaOp.getResult(), InterpretedValue(addr, 64));
  if (isModuleLevel)
    moduleLevelAllocaBaseAddr[allocaOp.getResult()] = addr;

  LLVM_DEBUG(llvm::dbgs() << "  llvm.alloca: allocated " << totalSize
                          << " bytes at address 0x" << llvm::format_hex(addr, 16)
                          << (isModuleLevel ? " (module level)" : "") << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMLoad(ProcessId procId,
                                                         LLVM::LoadOp loadOp) {
  auto convertFourStateToLLVMStructLayout =
      [&](InterpretedValue sourceSignalVal, SignalId sourceSigId,
          Type targetLLVMType,
          bool allowPlainInterfaceHeuristic = false) -> InterpretedValue {

    auto llvmStructTy = dyn_cast<LLVM::LLVMStructType>(targetLLVMType);
    if (!llvmStructTy)
      return sourceSignalVal;
    auto body = llvmStructTy.getBody();
    if (body.size() != 2)
      return sourceSignalVal;

    unsigned valueWidth = getTypeWidth(body[0]);
    unsigned unknownWidth = getTypeWidth(body[1]);
    if (valueWidth == 0 || valueWidth != unknownWidth)
      return sourceSignalVal;
    bool treatAsFourState =
        scheduler.getSignalEncoding(sourceSigId) ==
        SignalEncoding::FourStateStruct;
    if (!treatAsFourState && allowPlainInterfaceHeuristic) {
      // Some interface child shadow fields are represented as plain i2
      // signals even though the loaded LLVM type is struct<(i1,i1)>
      // ({value,unknown}). Treat those as four-state payloads so
      // extracting field #0 yields the value bit rather than raw packed bits.
      treatAsFourState =
          sourceSignalVal.getWidth() == valueWidth + unknownWidth;
    }
    if (!treatAsFourState)
      return sourceSignalVal;

    // Preserve explicit {value, unknown} semantics for fully-unknown signals
    // instead of collapsing to a scalar X sentinel.
    if (sourceSignalVal.isX()) {
      APInt llvmBits = APInt::getZero(valueWidth + unknownWidth);
      APInt unknownBits = APInt::getAllOnes(unknownWidth);
      safeInsertBits(llvmBits, unknownBits, valueWidth);
      return InterpretedValue(llvmBits);
    }
    if (valueWidth + unknownWidth != sourceSignalVal.getWidth())
      return sourceSignalVal;

    APInt bits = sourceSignalVal.getAPInt();
    APInt unknownBits = bits.extractBits(unknownWidth, 0);
    APInt valueBits = bits.extractBits(valueWidth, unknownWidth);
    APInt llvmBits = APInt::getZero(valueWidth + unknownWidth);
    safeInsertBits(llvmBits, valueBits, 0);
    safeInsertBits(llvmBits, unknownBits, valueWidth);
    return InterpretedValue(llvmBits);
  };

  // If this is a load from an llhd.ref converted to an LLVM pointer,
  // treat it as a signal probe instead of a memory read.
  if (SignalId sigId = resolveSignalId(loadOp.getAddr())) {
    InterpretedValue signalVal;
    auto pendingIt = pendingEpsilonDrives.find(sigId);
    if (pendingIt != pendingEpsilonDrives.end()) {
      signalVal = pendingIt->second;
    } else {
      const SignalValue &sv = scheduler.getSignalValue(sigId);
      signalVal = InterpretedValue::fromSignalValue(sv);
    }

    Type signalType = getSignalValueType(sigId);
    // If the inferred signal type is missing or width-mismatched, re-derive it
    // from the ref operand being loaded. This avoids accidental narrowing when
    // the same SignalId is reachable from both whole-aggregate refs and
    // sub-field refs (which share the ID but not the type width).
    if (!signalType || getTypeWidth(signalType) != signalVal.getWidth()) {
      if (auto castOp =
              loadOp.getAddr()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          if (auto refType =
                  dyn_cast<llhd::RefType>(castOp.getInputs()[0].getType()))
            signalType = refType.getNestedType();
        }
      }
    }
    Type llvmType = loadOp.getType();
    bool convertedFourStateStructLayout = false;
    if (!signalVal.isX() &&
        scheduler.getSignalEncoding(sigId) == SignalEncoding::FourStateStruct) {
      if (auto llvmStructTy = dyn_cast<LLVM::LLVMStructType>(llvmType)) {
        auto body = llvmStructTy.getBody();
        if (body.size() == 2) {
          unsigned valueWidth = getTypeWidth(body[0]);
          unsigned unknownWidth = getTypeWidth(body[1]);
          if (valueWidth != 0 && valueWidth == unknownWidth &&
              valueWidth + unknownWidth == signalVal.getWidth()) {
            APInt bits = signalVal.getAPInt();
            APInt unknownBits = bits.extractBits(unknownWidth, 0);
            APInt valueBits = bits.extractBits(valueWidth, unknownWidth);
            APInt llvmBits = APInt::getZero(valueWidth + unknownWidth);
            safeInsertBits(llvmBits, valueBits, 0);
            safeInsertBits(llvmBits, unknownBits, valueWidth);
            signalVal = InterpretedValue(llvmBits);
            convertedFourStateStructLayout = true;
          }
        }
      }
    }
    if (!convertedFourStateStructLayout && !signalVal.isX() && signalType &&
        (isa<hw::StructType, hw::ArrayType>(signalType)) &&
        (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmType))) {
      APInt converted =
          convertHWToLLVMLayout(signalVal.getAPInt(), signalType, llvmType);
      signalVal = InterpretedValue(converted);
    }

    unsigned targetWidth = getTypeWidth(loadOp.getType());
    if (signalVal.isX()) {
      signalVal = InterpretedValue::makeX(targetWidth);
    } else if (signalVal.getWidth() != targetWidth) {
      APInt apVal = signalVal.getAPInt();
      if (apVal.getBitWidth() < targetWidth)
        apVal = apVal.zext(targetWidth);
      else if (apVal.getBitWidth() > targetWidth)
        apVal = apVal.trunc(targetWidth);
      signalVal = InterpretedValue(apVal);
    }

    setValue(procId, loadOp.getResult(), signalVal);
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: read signal " << sigId
                            << " (width=" << targetWidth << ")\n");
    return success();
  }

  // Diagnostic: detect loads from unrealized_conversion_cast of !llhd.ref
  // that failed resolveSignalId — this means signal values won't be read.
  if (auto castOp =
          loadOp.getAddr()
              .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    (void)castOp; // checked above
  }

  // Get the pointer value
  InterpretedValue ptrVal = getValue(procId, loadOp.getAddr());
  Type resultType = loadOp.getType();
  unsigned bitWidth =
      isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(resultType)
          ? getMemoryLayoutBitWidth(resultType)
          : getTypeWidth(resultType);
  unsigned loadSize = getLLVMTypeSize(resultType);

  // Check if this load targets an interface field that has a shadow signal.
  // BFM tasks read interface ports via llvm.load on the struct, but the
  // signal value is driven by the DUT and only exists in the signal domain.
  // Return the signal value directly so BFMs see live hardware values.
  if (!ptrVal.isX() && !interfaceFieldSignals.empty()) {
    uint64_t loadAddr = ptrVal.getUInt64();
    auto fieldIt = interfaceFieldSignals.find(loadAddr);
    if (fieldIt != interfaceFieldSignals.end()) {
      SignalId fieldSigId = fieldIt->second;
      const SignalValue &sv = scheduler.getSignalValue(fieldSigId);
      InterpretedValue schedulerSignalVal = InterpretedValue::fromSignalValue(sv);
      InterpretedValue signalVal = schedulerSignalVal;
      bool usedPendingDrive = false;

      // Check for pending epsilon drives (not yet committed to scheduler).
      // Guard against stale pending values shadowing live scheduler state.
      auto pendingIt = pendingEpsilonDrives.find(fieldSigId);
      if (pendingIt != pendingEpsilonDrives.end()) {
        SignalValue pendingSignalVal = pendingIt->second.toSignalValue();
        if (pendingSignalVal == sv) {
          signalVal = pendingIt->second;
          usedPendingDrive = true;
        } else {
          // The scheduler already contains a newer value for this signal;
          // drop the stale pending entry to avoid repeated shadowing.
          pendingEpsilonDrives.erase(pendingIt);
        }
      }

      InterpretedValue rawSignalVal = signalVal;
      signalVal = convertFourStateToLLVMStructLayout(
          signalVal, fieldSigId, resultType,
          /*allowPlainInterfaceHeuristic=*/true);

      static bool traceI3CDetectEdgeValues = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_DETECTEDGE_VALUES");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      static bool traceI3CSampleLoads = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_SAMPLE_LOADS");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      static bool traceApbMonitorLoads = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_APB_MONITOR_LOADS");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      if (traceI3CDetectEdgeValues) {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          llvm::StringRef currentFunc = stIt->second.currentFuncName;
          if (currentFunc.contains("i3c_target_monitor_bfm::detectEdge_scl") ||
              currentFunc.contains("i3c_controller_monitor_bfm::detectEdge_scl") ||
              currentFunc.contains("i3c_target_driver_bfm::detectEdge_scl") ||
              currentFunc.contains("i3c_controller_driver_bfm::detectEdge_scl")) {
            static uint64_t printed = 0;
            if (printed < 1024) {
              ++printed;
              llvm::StringRef sigName = "<unknown>";
              auto nIt = signalIdToName.find(fieldSigId);
              if (nIt != signalIdToName.end())
                sigName = nIt->second;
              llvm::SmallString<64> bits;
              if (signalVal.isX())
                bits = "X";
              else
                signalVal.getAPInt().toString(bits, /*Radix=*/2,
                                              /*Signed=*/false);
              SimTime now = scheduler.getCurrentTime();
              llvm::errs() << "[I3C-DETECTEDGE-LOAD] proc=" << procId
                           << " t=" << now.realTime << " d=" << now.deltaStep
                           << " func=" << currentFunc << " sig=" << fieldSigId
                           << " name=" << sigName << " val=" << bits << "\n";
            }
          }
        }
      }
      if (traceI3CSampleLoads) {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          llvm::StringRef currentFunc = stIt->second.currentFuncName;
          if (currentFunc.contains("i3c_controller_monitor_bfm::sample_target_address") ||
              currentFunc.contains("i3c_target_monitor_bfm::sample_target_address") ||
              currentFunc.contains("i3c_target_driver_bfm::sample_target_address")) {
            static uint64_t printed = 0;
            if (printed < 2048) {
              ++printed;
              llvm::StringRef sigName = "<unknown>";
              auto nIt = signalIdToName.find(fieldSigId);
              if (nIt != signalIdToName.end())
                sigName = nIt->second;
              llvm::SmallString<64> bits;
              if (signalVal.isX())
                bits = "X";
              else
                signalVal.getAPInt().toString(bits, /*Radix=*/2,
                                              /*Signed=*/false);
              llvm::SmallString<64> rawBits;
              if (rawSignalVal.isX())
                rawBits = "X";
              else
                rawSignalVal.getAPInt().toString(rawBits, /*Radix=*/2,
                                                 /*Signed=*/false);
              llvm::SmallString<64> schedBits;
              if (schedulerSignalVal.isX())
                schedBits = "X";
              else
                schedulerSignalVal.getAPInt().toString(schedBits, /*Radix=*/2,
                                                       /*Signed=*/false);
              SimTime now = scheduler.getCurrentTime();
              llvm::errs() << "[I3C-SAMPLE-LOAD] proc=" << procId
                           << " t=" << now.realTime << " d=" << now.deltaStep
                           << " func=" << currentFunc << " addr=0x"
                           << llvm::format_hex(loadAddr, 16)
                           << " sig=" << fieldSigId << " name=" << sigName
                           << " enc="
                           << static_cast<int>(
                                  scheduler.getSignalEncoding(fieldSigId))
                           << " usedPending=" << (usedPendingDrive ? 1 : 0)
                           << " schedRaw=" << schedBits << " schedRawW="
                           << schedulerSignalVal.getWidth()
                           << " raw=" << rawBits << " rawW="
                           << rawSignalVal.getWidth() << " val=" << bits
                           << " valW=" << signalVal.getWidth() << "\n";
            }
          }
        }
      }
      if (traceApbMonitorLoads) {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          llvm::StringRef currentFunc = stIt->second.currentFuncName;
          if (currentFunc.contains("apb_master_monitor_bfm::sample_data") ||
              currentFunc.contains("apb_slave_monitor_bfm::sample_data")) {
            static uint64_t printed = 0;
            if (printed < 2048) {
              ++printed;
              llvm::StringRef sigName = "<unknown>";
              auto nIt = signalIdToName.find(fieldSigId);
              if (nIt != signalIdToName.end())
                sigName = nIt->second;
              llvm::SmallString<64> bits;
              if (signalVal.isX())
                bits = "X";
              else
                signalVal.getAPInt().toString(bits, /*Radix=*/2,
                                              /*Signed=*/false);
              SimTime now = scheduler.getCurrentTime();
              llvm::errs() << "[APB-MON-LOAD] proc=" << procId
                           << " t=" << now.realTime << " d=" << now.deltaStep
                           << " func=" << currentFunc << " addr=0x"
                           << llvm::format_hex(loadAddr, 16)
                           << " sig=" << fieldSigId << " name=" << sigName
                           << " w=" << signalVal.getWidth()
                           << " val=" << bits << "\n";
            }
          }
        }
      }

      static bool traceAhbMonitorLoads = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_AHB_MONITOR_LOADS");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      static uint64_t traceAhbMonitorMinTimeFs = []() -> uint64_t {
        const char *env = std::getenv("CIRCT_SIM_TRACE_AHB_MONITOR_MIN_TIME_FS");
        if (!env || env[0] == '\0')
          return 0;
        return strtoull(env, nullptr, 10);
      }();
      if (traceAhbMonitorLoads) {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          llvm::StringRef currentFunc = stIt->second.currentFuncName;
          if (isAhbMonitorSampleFunctionForTrace(currentFunc)) {
            SimTime now = scheduler.getCurrentTime();
            if (now.realTime >= traceAhbMonitorMinTimeFs) {
              static uint64_t printed = 0;
              if (printed < 4096) {
                ++printed;
                llvm::StringRef sigName = "<unknown>";
                auto nIt = signalIdToName.find(fieldSigId);
                if (nIt != signalIdToName.end())
                  sigName = nIt->second;
                llvm::SmallString<64> bits;
                if (signalVal.isX())
                  bits = "X";
                else
                  signalVal.getAPInt().toString(bits, /*Radix=*/2,
                                                /*Signed=*/false);
                llvm::errs() << "[AHB-MON-LOAD] proc=" << procId
                             << " t=" << now.realTime << " d=" << now.deltaStep
                             << " func=" << currentFunc << " addr="
                             << llvm::format_hex(loadAddr, 16)
                             << " sig=" << fieldSigId << " name=" << sigName
                             << " w=" << signalVal.getWidth()
                             << " val=" << bits << " usedPending="
                             << (usedPendingDrive ? 1 : 0) << "\n";
              }
            }
          }
        }
      }

      // If this signal is a struct pointer with known field signals,
      // reconstruct the struct value from individual field signal values.
      auto ptrFieldsIt = interfacePtrToFieldSignals.find(fieldSigId);

      // If the signal is X and NOT in interfacePtrToFieldSignals, it's likely
      // a sub-struct field of a parent interface. Try to reconstruct its value
      // from child field signals via the interfaceFieldPropagation chain.
      if (signalVal.isX() &&
          ptrFieldsIt == interfacePtrToFieldSignals.end()) {
        for (auto &[parentIfaceSigId, parentFieldSigIds] :
             interfacePtrToFieldSignals) {
          int fieldIdx = -1;
          for (size_t i = 0; i < parentFieldSigIds.size(); ++i) {
            if (parentFieldSigIds[i] == fieldSigId) {
              fieldIdx = static_cast<int>(i);
              break;
            }
          }
          if (fieldIdx < 0)
            continue;

          // Reconstruct by reading child field signal values.
          APInt result = APInt::getZero(bitWidth);
          unsigned bitOffset = 0;
          bool anyNonX = false;
          for (size_t fi = static_cast<size_t>(fieldIdx);
               fi < parentFieldSigIds.size() && bitOffset < bitWidth; ++fi) {
            SignalId parentFieldSig = parentFieldSigIds[fi];
            auto propIt2 = interfaceFieldPropagation.find(parentFieldSig);
            const SignalValue *childSV = nullptr;
            if (propIt2 != interfaceFieldPropagation.end()) {
              for (SignalId childSig : propIt2->second) {
                const SignalValue &csv = scheduler.getSignalValue(childSig);
                if (!csv.isUnknown()) {
                  childSV = &csv;
                  break;
                }
              }
            }
            if (!childSV) {
              const SignalValue &psv =
                  scheduler.getSignalValue(parentFieldSig);
              if (!psv.isUnknown())
                childSV = &psv;
            }
            if (childSV) {
              unsigned fw = childSV->getWidth();
              APInt fieldBits = childSV->getAPInt();
              if (fieldBits.getBitWidth() > fw)
                fieldBits = fieldBits.trunc(fw);
              for (unsigned b = 0;
                   b < fw && bitOffset + b < bitWidth; ++b) {
                if (fieldBits[b])
                  result.setBit(bitOffset + b);
              }
              bitOffset += fw;
              anyNonX = true;
            } else {
              unsigned fw = scheduler.getSignalValue(parentFieldSig).getWidth();
              bitOffset += fw;
            }
          }
          if (anyNonX) {
            signalVal = InterpretedValue(result);
            setValue(procId, loadOp.getResult(), signalVal);
            return success();
          }
          break;  // Only check first matching parent
        }
        goto normal_memory_load;
      }
      if (ptrFieldsIt != interfacePtrToFieldSignals.end() &&
          !ptrFieldsIt->second.empty()) {
        APInt result = APInt::getZero(bitWidth);
        unsigned bitOffset = 0;
        for (SignalId fid : ptrFieldsIt->second) {
          const SignalValue &fsv = scheduler.getSignalValue(fid);
          unsigned fw = fsv.getWidth();
          if (!fsv.isUnknown() && bitOffset + fw <= bitWidth) {
            APInt fieldBits = fsv.getAPInt();
            if (fieldBits.getBitWidth() < fw)
              fieldBits = fieldBits.zext(fw);
            else if (fieldBits.getBitWidth() > fw)
              fieldBits = fieldBits.trunc(fw);
            for (unsigned b = 0; b < fw && bitOffset + b < bitWidth; ++b) {
              if (fieldBits[b])
                result.setBit(bitOffset + b);
            }
          }
          bitOffset += fw;
        }
        signalVal = InterpretedValue(result);
        setValue(procId, loadOp.getResult(), signalVal);
        return success();
      }

      // Resize to match expected load width
      if (signalVal.isX()) {
        signalVal = InterpretedValue::makeX(bitWidth);
      } else if (signalVal.getWidth() != bitWidth) {
        APInt apVal = signalVal.getAPInt();
        if (apVal.getBitWidth() < bitWidth)
          apVal = apVal.zext(bitWidth);
        else if (apVal.getBitWidth() > bitWidth)
          apVal = apVal.trunc(bitWidth);
        signalVal = InterpretedValue(apVal);
      }

      setValue(procId, loadOp.getResult(), signalVal);
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.load: read interface field signal " << fieldSigId
                 << " at 0x" << llvm::format_hex(loadAddr, 16) << "\n");
      return success();
    }
  }

normal_memory_load:
  static bool traceNativeLoad = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_NATIVE_LOAD");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static bool disableNativeDeref = []() {
    const char *env = std::getenv("CIRCT_SIM_DISABLE_NATIVE_DEREF");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto traceNativeLoadEvent = [&](llvm::StringRef tag, uint64_t addr,
                                  uint64_t off, size_t size,
                                  unsigned accessSize) {
    if (!traceNativeLoad)
      return;
    static uint64_t printed = 0;
    if (printed >= 4096)
      return;
    ++printed;
    llvm::StringRef funcName = "<unknown>";
    auto stIt = processStates.find(procId);
    if (stIt != processStates.end() && !stIt->second.currentFuncName.empty())
      funcName = stIt->second.currentFuncName;
    llvm::errs() << "[NATIVE-LOAD] tag=" << tag << " proc=" << procId
                 << " func=" << funcName << " addr="
                 << llvm::format_hex(addr, 16) << " off=" << off
                 << " size=" << size << " access=" << accessSize << "\n";
  };

  // First try to find a local memory block (from alloca)
  MemoryBlock *block = findMemoryBlock(procId, loadOp.getAddr());
  uint64_t offset = 0;
  bool useNative = false;
  uint64_t nativeOffset = 0;
  size_t nativeSize = 0;
  if (block) {
    // Local alloca memory
    // Calculate the offset within the memory block
    if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
      InterpretedValue baseVal = getValue(procId, gepOp.getBase());
      if (!baseVal.isX() && !ptrVal.isX()) {
        offset = ptrVal.getUInt64() - baseVal.getUInt64();
      }
    }
  } else if (!ptrVal.isX()) {
    // Check if this is a global memory access
    uint64_t addr = ptrVal.getUInt64();

    // Find which global or malloc block this address belongs to.
    // Uses O(log n) binary search via the address range index instead of
    // O(n) linear scan through all 6000+ globals.
    block = findBlockByAddress(addr, offset);
    LLVM_DEBUG(if (block) llvm::dbgs()
               << "  llvm.load: found block at offset " << offset << "\n");
    if (!block) {
      if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize)) {
        useNative = true;
        offset = nativeOffset;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.load: found native block at 0x"
                                << llvm::format_hex(addr - nativeOffset, 16)
                                << " offset " << nativeOffset << "\n");
      }
    }
  }

  // Fallback: use comprehensive address-based search (also checks
  // process-local allocas by address, which findMemoryBlock's SSA
  // tracing may miss when the pointer was loaded from memory).
  // Also try this when useNative is set but the native block is too small
  // for the requested load size (e.g., an 8-byte assoc array slot matched
  // but we need to load a 24-byte struct from a larger malloc'd block).
  if (!block && !ptrVal.isX()) {
    unsigned loadSizeCheck = getLLVMTypeSize(loadOp.getResult().getType());
    if (!useNative || (offset + loadSizeCheck > nativeSize)) {
      uint64_t fbOffset = 0;
      auto *fbBlock = findMemoryBlockByAddress(ptrVal.getUInt64(), procId, &fbOffset);
      if (fbBlock) {
        block = fbBlock;
        offset = fbOffset;
        useNative = false;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.load: findMemoryBlockByAddress found "
                                   "block at offset " << offset << "\n");
      }
    }
  }

  // Native pointer access is only allowed for known blocks registered by the
  // runtime (e.g., associative array element refs). If a pointer is not in any
  // tracked block, return X (unknown value).

  if (!block && !useNative) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    setValue(procId, loadOp.getResult(),
             InterpretedValue::makeX(bitWidth));
    return success();
  }

  if (block) {
    if (offset + loadSize > block->size) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.load: out of bounds access (offset="
                              << offset << " size=" << loadSize
                              << " block_size=" << block->size << ")\n");
      setValue(procId, loadOp.getResult(),
               InterpretedValue::makeX(bitWidth));
      return success();
    }

    bool hasInterfaceMask = false;
    if (!ptrVal.isX()) {
      uint64_t baseAddr = ptrVal.getUInt64() - offset;
      auto maskIt = interfaceMemoryByteInitMask.find(baseAddr);
      if (maskIt != interfaceMemoryByteInitMask.end()) {
        hasInterfaceMask = true;
        const auto &byteInit = maskIt->second;
        bool rangeInitialized = offset + loadSize <= byteInit.size();
        if (rangeInitialized) {
          for (unsigned i = 0; i < loadSize; ++i) {
            if (byteInit[offset + i] == 0) {
              rangeInitialized = false;
              break;
            }
          }
        }
        if (!rangeInitialized) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  llvm.load: reading uninitialized interface bytes\n");
          setValue(procId, loadOp.getResult(),
                   InterpretedValue::makeX(bitWidth));
          return success();
        }
      }
    }

    // Check if memory has been initialized
    if (!hasInterfaceMask && !block->initialized) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.load: reading uninitialized memory\n");
      setValue(procId, loadOp.getResult(),
               InterpretedValue::makeX(bitWidth));
      return success();
    }
  } else {
    if (offset + loadSize > nativeSize) {
      traceNativeLoadEvent("native_oob", ptrVal.getUInt64(), offset, nativeSize,
                           loadSize);
      // Native memory ranges can point to runtime-managed allocations that may
      // no longer be valid. Do not blindly grow the range and dereference it.
      setValue(procId, loadOp.getResult(), InterpretedValue::makeX(bitWidth));
      return success();
    }
    if (disableNativeDeref) {
      traceNativeLoadEvent("native_deref_disabled", ptrVal.getUInt64(), offset,
                           nativeSize, loadSize);
      setValue(procId, loadOp.getResult(), InterpretedValue::makeX(bitWidth));
      return success();
    }
    traceNativeLoadEvent("native_read", ptrVal.getUInt64(), offset, nativeSize,
                         loadSize);
  }

  auto readByte = [&](unsigned i) -> uint8_t {
    if (block)
      return block->data[offset + i];
    auto *nativePtr = reinterpret_cast<const uint8_t *>(ptrVal.getUInt64());
    return nativePtr[i];
  };

  // Read bytes from memory and construct the value (little-endian).
  // Clamp to the number of bytes needed for the value width to avoid
  // shifting past the APInt width when loadSize includes padding.
  unsigned bytesForValue = std::min(loadSize, (bitWidth + 7) / 8);
  uint64_t value = 0;
  for (unsigned i = 0; i < bytesForValue && i < 8; ++i) {
    value |= static_cast<uint64_t>(readByte(i)) << (i * 8);
  }

  // For values larger than 64 bits, use APInt directly
  if (bitWidth > 64) {
    APInt apValue(bitWidth, 0);
    for (unsigned i = 0; i < bytesForValue; ++i) {
      APInt byteVal(bitWidth, readByte(i));
      apValue |= byteVal.shl(i * 8);
    }
    InterpretedValue loaded(apValue);
    setValue(procId, loadOp.getResult(), loaded);
    maybeRegisterNativeBlockFromPtrLenStruct(resultType, loaded,
                                             nativeMemoryBlocks);
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: loaded wide value ("
                            << loadSize << " bytes) from offset " << offset << "\n");
  } else {
    // Mask the loaded value to the exact bit width. Memory loads read
    // whole bytes, but sub-byte types (e.g., i1, i5) need only the
    // low bits. Without masking, a byte value like 0xFF for an i1 load
    // triggers an APInt assertion failure.
    if (bitWidth > 0 && bitWidth < 64)
      value &= (1ULL << bitWidth) - 1;
    InterpretedValue loaded(value, bitWidth);
    setValue(procId, loadOp.getResult(), loaded);
    maybeRegisterNativeBlockFromPtrLenStruct(resultType, loaded,
                                             nativeMemoryBlocks);
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: loaded 0x"
                            << llvm::format_hex(value, 16) << " ("
                            << loadSize << " bytes) from offset " << offset << "\n");
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMStore(
    ProcessId procId, LLVM::StoreOp storeOp) {
  auto isAllocaDerivedPtr = [&](Value ptr) {
    llvm::SmallVector<Value, 8> worklist;
    llvm::SmallDenseSet<Value, 8> visited;
    worklist.push_back(ptr);
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      if (current.getDefiningOp<LLVM::AllocaOp>())
        return true;
      if (auto gepOp = current.getDefiningOp<LLVM::GEPOp>()) {
        worklist.push_back(gepOp.getBase());
        continue;
      }
      if (auto bitcastOp = current.getDefiningOp<LLVM::BitcastOp>()) {
        worklist.push_back(bitcastOp.getArg());
        continue;
      }
      if (auto castOp =
              current.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        for (Value input : castOp.getInputs())
          worklist.push_back(input);
      }
    }
    return false;
  };

  // If this is a store to an llhd.ref converted to an LLVM pointer,
  // treat it as a signal drive instead of a memory write.
  if (SignalId sigId = resolveSignalId(storeOp.getAddr())) {
    InterpretedValue storeVal = getValue(procId, storeOp.getValue());
    const SignalValue &current = scheduler.getSignalValue(sigId);
    unsigned targetWidth = current.getWidth();

    Type signalType = getSignalValueType(sigId);
    if (!signalType) {
      if (auto castOp =
              storeOp.getAddr()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          if (auto refType =
                  dyn_cast<llhd::RefType>(castOp.getInputs()[0].getType()))
            signalType = refType.getNestedType();
        }
      }
    }
    Type llvmType = storeOp.getValue().getType();
    if (!storeVal.isX() && signalType &&
        (isa<hw::StructType, hw::ArrayType>(signalType)) &&
        (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmType))) {
      APInt converted =
          convertLLVMToHWLayout(storeVal.getAPInt(), llvmType, signalType);
      storeVal = InterpretedValue(converted);
    }

    if (storeVal.isX()) {
      storeVal = InterpretedValue::makeX(targetWidth);
    } else if (storeVal.getWidth() != targetWidth) {
      APInt apVal = storeVal.getAPInt();
      if (apVal.getBitWidth() < targetWidth)
        apVal = apVal.zext(targetWidth);
      else if (apVal.getBitWidth() > targetWidth)
        apVal = apVal.trunc(targetWidth);
      storeVal = InterpretedValue(apVal);
    }

    pendingEpsilonDrives[sigId] = storeVal;
    scheduler.updateSignal(sigId, storeVal.toSignalValue());
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: wrote signal " << sigId
                            << " (width=" << targetWidth << ")\n");
    return success();
  }

  // Signal not resolved - going to memory
  // Get the pointer value first
  InterpretedValue ptrVal = getValue(procId, storeOp.getAddr());
  bool addrIsAllocaDerived = isAllocaDerivedPtr(storeOp.getAddr());
  static bool traceNativeStore = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_NATIVE_STORE");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static bool disableNativeDeref = []() {
    const char *env = std::getenv("CIRCT_SIM_DISABLE_NATIVE_DEREF");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto traceNativeStoreEvent = [&](llvm::StringRef tag, uint64_t addr,
                                   uint64_t off, size_t size,
                                   unsigned accessSize) {
    if (!traceNativeStore)
      return;
    static uint64_t printed = 0;
    if (printed >= 4096)
      return;
    ++printed;
    llvm::StringRef funcName = "<unknown>";
    auto stIt = processStates.find(procId);
    if (stIt != processStates.end() && !stIt->second.currentFuncName.empty())
      funcName = stIt->second.currentFuncName;
    llvm::errs() << "[NATIVE-STORE] tag=" << tag << " proc=" << procId
                 << " func=" << funcName << " addr="
                 << llvm::format_hex(addr, 16) << " off=" << off
                 << " size=" << size << " access=" << accessSize << "\n";
  };

  // Find the memory block for this pointer
  MemoryBlock *block = findMemoryBlock(procId, storeOp.getAddr());
  uint64_t offset = 0;
  bool useNative = false;
  uint64_t nativeOffset = 0;
  size_t nativeSize = 0;

  if (block) {
    // Local alloca memory
    // Calculate the offset within the memory block
    if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
      InterpretedValue baseVal = getValue(procId, gepOp.getBase());
      if (!baseVal.isX() && !ptrVal.isX()) {
        offset = ptrVal.getUInt64() - baseVal.getUInt64();
      }
    }
  } else if (!ptrVal.isX()) {
    // Check if this is a global memory access
    uint64_t addr = ptrVal.getUInt64();

    // Find which global or malloc block this address belongs to.
    // Uses O(log n) binary search via the address range index.
    block = findBlockByAddress(addr, offset);
    LLVM_DEBUG(if (block) llvm::dbgs()
               << "  llvm.store: found block at offset " << offset << "\n");

    // Check module-level allocas by address. This is needed when the store
    // address is computed via a GEP on a pointer loaded from memory (e.g.,
    // class member access through a heap-allocated class instance).
    if (!block) {
      for (auto &[val, memBlock] : moduleLevelAllocas) {
        uint64_t blockAddr = getModuleLevelAllocaBaseAddress(val);
        if (blockAddr != 0 && addr >= blockAddr && addr < blockAddr + memBlock.size) {
          block = &memBlock;
          offset = addr - blockAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found module-level alloca at 0x"
                                  << llvm::format_hex(blockAddr, 16)
                                  << " offset " << offset << "\n");
          break;
        }
      }
    }

    if (!block) {
      if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize)) {
        useNative = true;
        offset = nativeOffset;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found native block at 0x"
                                << llvm::format_hex(addr - nativeOffset, 16)
                                << " offset " << nativeOffset << "\n");
      }
    }
  }

  // Get the value to store
  InterpretedValue storeVal = getValue(procId, storeOp.getValue());
  unsigned storeSize = getLLVMTypeSize(storeOp.getValue().getType());

  auto isTriStateDestSignal = [&](SignalId sigId) {
    return llvm::any_of(interfaceTriStateRules, [&](const auto &rule) {
      return rule.destSigId == sigId;
    });
  };

  // Mirror stores of a probed shared net back into an interface tri-state
  // destination field are observation-only. Writing them into field memory
  // turns the interface output path into a feedback latch (the mirrored value
  // is then re-driven onto the shared net).
  bool suppressTriStateMirrorStoreWrite = false;
  SignalId triStateMirrorFieldSigId = 0;
  SignalId triStateMirrorSrcSigId = 0;
  if (!ptrVal.isX() && !interfaceFieldSignals.empty() &&
      !addrIsAllocaDerived) {
    uint64_t storeAddr = ptrVal.getUInt64();
    auto fieldIt = interfaceFieldSignals.find(storeAddr);
    if (fieldIt != interfaceFieldSignals.end()) {
      triStateMirrorFieldSigId = fieldIt->second;
      auto resolveSignal = [&](Value signalRef) -> SignalId {
        if (SignalId sigId = resolveSignalId(signalRef))
          return sigId;
        return getSignalId(signalRef);
      };
      if (matchFourStateProbeCopyStore(storeOp.getValue(), resolveSignal,
                                       triStateMirrorSrcSigId) &&
          triStateMirrorSrcSigId != 0 &&
          triStateMirrorSrcSigId != triStateMirrorFieldSigId &&
          isTriStateDestSignal(triStateMirrorFieldSigId)) {
        // Only suppress mirrored probe-copy writes once the corresponding
        // tri-state rule can already drive the destination deterministically.
        // During early init some designs seed interface fields via probe-copy
        // before cond settles; suppressing too early can latch X and block
        // later bus activity.
        auto getRuleSignalValue = [&](SignalId sigId) -> InterpretedValue {
          auto pendingIt = pendingEpsilonDrives.find(sigId);
          if (pendingIt != pendingEpsilonDrives.end())
            return pendingIt->second;
          return InterpretedValue::fromSignalValue(
              scheduler.getSignalValue(sigId));
        };
        auto normalizeForDestSignal = [&](SignalId destSigId,
                                          InterpretedValue value)
            -> InterpretedValue {
          const SignalValue &destCurrent = scheduler.getSignalValue(destSigId);
          unsigned targetWidth = destCurrent.getWidth();
          if (value.isX())
            return InterpretedValue::makeX(targetWidth);
          if (value.getWidth() == targetWidth)
            return value;
          APInt bits = value.getAPInt();
          if (bits.getBitWidth() < targetWidth)
            bits = bits.zext(targetWidth);
          else if (bits.getBitWidth() > targetWidth)
            bits = bits.trunc(targetWidth);
          return InterpretedValue(bits);
        };
        auto isExplicitHighZForSignal = [&](SignalId sigId,
                                            InterpretedValue value) -> bool {
          if (value.isX())
            return false;
          if (scheduler.getSignalEncoding(sigId) !=
              SignalEncoding::FourStateStruct)
            return false;
          const SignalValue &destCurrent = scheduler.getSignalValue(sigId);
          unsigned width = destCurrent.getWidth();
          if (width < 2 || (width % 2) != 0)
            return false;
          value = normalizeForDestSignal(sigId, value);
          if (value.isX())
            return false;
          APInt bits = value.getAPInt();
          unsigned logicalWidth = width / 2;
          APInt unknownBits = bits.extractBits(logicalWidth, 0);
          APInt valueBits = bits.extractBits(logicalWidth, logicalWidth);
          return unknownBits.isAllOnes() && valueBits.isAllOnes();
        };
        bool inStartupTime = scheduler.getCurrentTime().realTime == 0;
        bool triStateRuleCanDriveDest = false;
        for (const auto &rule : interfaceTriStateRules) {
          if (rule.destSigId != triStateMirrorFieldSigId)
            continue;
          InterpretedValue condVal = getRuleSignalValue(rule.condSigId);
          if (condVal.isX() || rule.condBitIndex >= condVal.getWidth())
            continue;
          bool condTrue = condVal.getAPInt()[rule.condBitIndex];
          if (!condTrue) {
            InterpretedValue elseVal =
                normalizeForDestSignal(rule.destSigId, rule.elseValue);
            bool elseIsExplicitHighZ =
                isExplicitHighZForSignal(rule.destSigId, elseVal);
            // Keep startup behavior stable at t=0: high-Z mirror-store
            // suppression during initialization avoids pre-run phase churn.
            // After startup, explicit Z cannot deterministically represent the
            // resolved shared net value, so mirror observations must not be
            // suppressed.
            bool canSuppressForElse =
                !elseIsExplicitHighZ || inStartupTime;
            if (!elseVal.isX() && canSuppressForElse)
              triStateRuleCanDriveDest = true;
          } else {
            InterpretedValue srcVal = normalizeForDestSignal(
                rule.destSigId, getRuleSignalValue(rule.srcSigId));
            bool srcIsExplicitHighZ =
                isExplicitHighZForSignal(rule.destSigId, srcVal);
            bool canSuppressForSrc = !srcIsExplicitHighZ || inStartupTime;
            if (!srcVal.isX() && canSuppressForSrc)
              triStateRuleCanDriveDest = true;
          }
          if (triStateRuleCanDriveDest)
            break;
        }
        suppressTriStateMirrorStoreWrite = triStateRuleCanDriveDest;
      }
    }
  }

  // Fallback: use comprehensive address-based search (also checks
  // process-local allocas by address, which findMemoryBlock's SSA
  // tracing may miss when the pointer was loaded from memory).
  // Also try this when useNative is set but the native block is too small
  // for the requested store size (e.g., an 8-byte assoc array slot matched
  // but we need to store a 24-byte struct into a larger malloc'd block).
  if (!block && !ptrVal.isX()) {
    if (!useNative || (offset + storeSize > nativeSize)) {
      uint64_t fbOffset = 0;
      auto *fbBlock = findMemoryBlockByAddress(ptrVal.getUInt64(), procId, &fbOffset);
      if (fbBlock) {
        block = fbBlock;
        offset = fbOffset;
        useNative = false;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.store: findMemoryBlockByAddress found "
                                   "block at offset " << offset << "\n");
      }
    }
  }

  // Native pointer access is only allowed for known blocks registered by the
  // runtime. If pointer is not tracked, the store is silently skipped (stores
  // to X are no-ops anyway).

  if (!block && !useNative) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    return success(); // Don't fail, just skip the store
  }

  if (block) {
    if (offset + storeSize > block->size) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.store: out of bounds access\n");
      return success();
    }
  } else {
    if (offset + storeSize > nativeSize) {
      traceNativeStoreEvent("native_oob", ptrVal.getUInt64(), offset,
                            nativeSize, storeSize);
      return success();
    }
    if (disableNativeDeref) {
      traceNativeStoreEvent("native_deref_disabled", ptrVal.getUInt64(), offset,
                            nativeSize, storeSize);
      return success();
    }
    traceNativeStoreEvent("native_write", ptrVal.getUInt64(), offset,
                          nativeSize, storeSize);
  }

  static bool traceAhbMonitorStores = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_AHB_MONITOR_STORES");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (traceAhbMonitorStores) {
    auto stIt = processStates.find(procId);
    if (stIt != processStates.end()) {
      llvm::StringRef currentFunc = stIt->second.currentFuncName;
      if (isAhbMonitorSampleFunctionForTrace(currentFunc)) {
        static uint64_t printed = 0;
        if (printed < 8192) {
          ++printed;
          llvm::SmallString<128> rawBits;
          if (storeVal.isX())
            rawBits = "X";
          else
            storeVal.getAPInt().toString(rawBits, /*Radix=*/2, /*Signed=*/false);

          uint64_t storeAddr = ptrVal.isX() ? 0 : ptrVal.getUInt64();
          llvm::errs() << "[AHB-MON-STORE] proc=" << procId
                       << " func=" << currentFunc << " addr="
                       << llvm::format_hex(storeAddr, 16)
                       << " size=" << storeSize
                       << " valW=" << storeVal.getWidth()
                       << " val=" << rawBits
                       << " block=" << (block ? "managed" : (useNative ? "native" : "none"))
                       << " off=" << offset;

          if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
            auto rawIdx = gepOp.getRawConstantIndices();
            llvm::errs() << " gep=[";
            for (size_t i = 0; i < rawIdx.size(); ++i) {
              if (i)
                llvm::errs() << ",";
              if (rawIdx[i] == LLVM::GEPOp::kDynamicIndex)
                llvm::errs() << "?";
              else
                llvm::errs() << rawIdx[i];
            }
            llvm::errs() << "]";
          }

          if (!ptrVal.isX() && !interfaceFieldSignals.empty()) {
            auto fieldIt = interfaceFieldSignals.find(storeAddr);
            if (fieldIt != interfaceFieldSignals.end()) {
              llvm::StringRef sigName = "<unknown>";
              auto nameIt = signalIdToName.find(fieldIt->second);
              if (nameIt != signalIdToName.end())
                sigName = nameIt->second;
              llvm::errs() << " sig=" << fieldIt->second << "(" << sigName
                           << ")";
            }
          }
          llvm::errs() << "\n";
        }
      }
    }
  }

  // Write bytes to memory (little-endian)
  if (!suppressTriStateMirrorStoreWrite && !storeVal.isX()) {
    const APInt &apValue = storeVal.getAPInt();
    auto storeByte = [&](unsigned i, uint8_t value) {
      if (block)
        block->data[offset + i] = value;
      else
        reinterpret_cast<uint8_t *>(ptrVal.getUInt64())[i] = value;
    };
    if (apValue.getBitWidth() > 64) {
      // Handle wide values using APInt operations
      unsigned bitWidth = apValue.getBitWidth();
      for (unsigned i = 0; i < storeSize; ++i) {
        unsigned bitPos = i * 8;
        if (bitPos >= bitWidth) {
          // Beyond the value's bit width - store zero
          storeByte(i, 0);
        } else {
          // Extract available bits (up to 8), remaining bits are zero
          unsigned bitsAvailable = std::min(8u, bitWidth - bitPos);
          uint8_t byteVal = static_cast<uint8_t>(
              apValue.extractBits(bitsAvailable, bitPos).getZExtValue());
          storeByte(i, byteVal);
        }
      }
    } else {
      uint64_t value = storeVal.getUInt64();
      for (unsigned i = 0; i < storeSize && i < 8; ++i) {
        storeByte(i, static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
      }
    }
    if (block)
      block->initialized = true;
  }

  LLVM_DEBUG(llvm::dbgs() << "  llvm.store: stored "
                          << (storeVal.isX() ? "X" : std::to_string(storeVal.getUInt64()))
                          << " (" << storeSize << " bytes) at offset " << offset << "\n");

  // UVM fix: when m_inst is stored during global init, also mirror to uvm_top.
  // This prevents the re-entrant m_uvm_get_root() call from seeing
  // m_inst != uvm_top and triggering an infinite fatal loop.
  if (inGlobalInit && !storeVal.isX() && storeVal.getUInt64() != 0) {
    if (auto addrOfOp = storeOp.getAddr().getDefiningOp<LLVM::AddressOfOp>()) {
      if (addrOfOp.getGlobalName().contains("uvm_root::m_inst")) {
        StringRef uvmTopName = "uvm_pkg::uvm_top";
        auto topBlockIt = globalMemoryBlocks.find(uvmTopName);
        if (topBlockIt != globalMemoryBlocks.end()) {
          auto &topBlock = topBlockIt->second;
          uint64_t value = storeVal.getUInt64();
          for (unsigned i = 0; i < storeSize && i < 8 && i < topBlock.size; ++i)
            topBlock.data[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
          topBlock.initialized = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "  UVM fix: mirrored m_inst store to uvm_top ("
                     << llvm::format_hex(value, 18) << ")\n");
        }
      }
    }
  }
  // Sync signal backing memory: when a store writes to a memory block that
  // backs an LLHD signal (allocated in unrealized_conversion_cast for signals
  // without llhd.drv users), also update pendingEpsilonDrives and the
  // scheduler so that llhd.prb sees the new value.
  if (block && !signalBackingMemory.empty()) {
    for (auto &[backingSigId, backingInfo] : signalBackingMemory) {
      auto &backingSt = processStates[backingInfo.first];
      auto backingBlkIt = backingSt.memoryBlocks.find(backingInfo.second);
      if (backingBlkIt != backingSt.memoryBlocks.end() &&
          &backingBlkIt->second == block) {
        // This store wrote to signal backing memory — read back and sync
        const SignalValue &current = scheduler.getSignalValue(backingSigId);
        unsigned sigWidth = current.getWidth();
        unsigned sigBytes = (sigWidth + 7) / 8;
        APInt newBits = APInt::getZero(sigWidth);
        for (unsigned i = 0; i < sigBytes && i * 8 < sigWidth; ++i) {
          unsigned bitsToInsert = std::min(8u, sigWidth - i * 8);
          APInt byteVal(bitsToInsert,
                        block->data[i] & ((1u << bitsToInsert) - 1));
          safeInsertBits(newBits, byteVal, i * 8);
        }
        // Convert LLVM layout to HW layout if the signal has struct type
        Type sigType = getSignalValueType(backingSigId);
        if (sigType && isa<hw::StructType, hw::ArrayType>(sigType)) {
          Type llvmType = storeOp.getValue().getType();
          if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmType))
            newBits = convertLLVMToHWLayout(newBits, llvmType, sigType);
        }
        InterpretedValue syncVal(newBits);
        pendingEpsilonDrives[backingSigId] = syncVal;
        scheduler.updateSignal(backingSigId, syncVal.toSignalValue());
        break;
      }
    }
  }

  // Drive interface field shadow signals. When a store targets an address
  // within a known interface struct, also drive the corresponding shadow
  // signal so that processes sensitive to the interface field wake up.
  if (!ptrVal.isX() && !interfaceFieldSignals.empty() &&
      !addrIsAllocaDerived) {
    static bool traceInterfaceStore = []() {
      const char *env = std::getenv("CIRCT_SIM_TRACE_IFACE_STORE");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    uint64_t storeAddr = ptrVal.getUInt64();
    auto fieldIt = interfaceFieldSignals.find(storeAddr);
    if (fieldIt != interfaceFieldSignals.end()) {
      SignalId fieldSigId = fieldIt->second;
      if (!suppressTriStateMirrorStoreWrite && block &&
          offset + storeSize <= block->size) {
        uint64_t baseAddr = storeAddr - offset;
        auto &byteInit = interfaceMemoryByteInitMask[baseAddr];
        if (byteInit.size() != block->size)
          byteInit.assign(block->size, 0);
        std::fill(byteInit.begin() + offset,
                  byteInit.begin() + offset + storeSize,
                  storeVal.isX() ? uint8_t{0} : uint8_t{1});
      }

      bool addedDynamicCopyLink = false;
      SignalId copySrcSigId = 0;
      uint64_t copySrcAddr = 0;
      bool copySrcFromInterfaceField = false;
      bool copySrcCrossBlock = false;
      auto resolveSignal = [&](Value signalRef) -> SignalId {
        if (SignalId sigId = resolveSignalId(signalRef))
          return sigId;
        return getSignalId(signalRef);
      };
      auto resolveAddr = [&](Value addrValue) -> uint64_t {
        InterpretedValue addrVal = getValue(procId, addrValue);
        if (!addrVal.isX() && addrVal.getUInt64() != 0)
          return addrVal.getUInt64();
        auto initIt = moduleInitValueMap.find(addrValue);
        if (initIt != moduleInitValueMap.end() && !initIt->second.isX() &&
            initIt->second.getUInt64() != 0)
          return initIt->second.getUInt64();
        return 0;
      };
      auto areCrossBlockInterfaceFields = [&](SignalId srcSigId,
                                              SignalId destSigId) -> bool {
        auto srcAddrIt = fieldSignalToAddr.find(srcSigId);
        auto dstAddrIt = fieldSignalToAddr.find(destSigId);
        if (srcAddrIt == fieldSignalToAddr.end() ||
            dstAddrIt == fieldSignalToAddr.end())
          return false;
        uint64_t srcOff = 0, dstOff = 0;
        MemoryBlock *srcBlk = findBlockByAddress(srcAddrIt->second, srcOff);
        MemoryBlock *dstBlk = findBlockByAddress(dstAddrIt->second, dstOff);
        return srcBlk && dstBlk && srcBlk != dstBlk;
      };
      auto hasMatchingWidth = [&](SignalId srcSigId, SignalId destSigId) -> bool {
        return scheduler.getSignalValue(srcSigId).getWidth() ==
               scheduler.getSignalValue(destSigId).getWidth();
      };
      bool suppressTriStateMirrorStoreForField =
          suppressTriStateMirrorStoreWrite &&
          fieldSigId == triStateMirrorFieldSigId &&
          triStateMirrorSrcSigId != 0;
      bool useAggressiveTriStateMirrorSuppression =
          !hasInstanceScopedInterfaceFieldSignals;
      if (matchFourStateCopyStore(storeOp.getValue(), resolveAddr,
                                  copySrcAddr) ||
          matchFourStateStructCreateLoad(storeOp.getValue(), resolveAddr,
                                         copySrcAddr)) {
        auto srcFieldIt = interfaceFieldSignals.find(copySrcAddr);
        if (srcFieldIt != interfaceFieldSignals.end()) {
          copySrcSigId = srcFieldIt->second;
          copySrcFromInterfaceField = true;
          copySrcCrossBlock =
              areCrossBlockInterfaceFields(copySrcSigId, fieldSigId);
        }
      }
      SignalId probeCopySrcSigId = 0;
      bool hasProbeCopySrc =
          matchFourStateProbeCopyStore(storeOp.getValue(), resolveSignal,
                                       probeCopySrcSigId) &&
          probeCopySrcSigId != 0 && probeCopySrcSigId != fieldSigId;
      bool hasInterfaceFieldCopySrc =
          copySrcFromInterfaceField && copySrcSigId != 0 &&
          copySrcSigId != fieldSigId;
      if (hasProbeCopySrc || hasInterfaceFieldCopySrc) {
        if (hasProbeCopySrc) {
          copySrcSigId = probeCopySrcSigId;
          copySrcAddr = 0;
          copySrcFromInterfaceField = false;
          copySrcCrossBlock = false;
        }
        auto addDynamicLink = [&](SignalId destSigId) {
          if (copySrcFromInterfaceField) {
            if (!copySrcCrossBlock)
              return;
            if (!hasMatchingWidth(copySrcSigId, destSigId))
              return;
          }
          auto &children = interfaceFieldPropagation[copySrcSigId];
          if (std::find(children.begin(), children.end(), destSigId) ==
              children.end()) {
            children.push_back(destSigId);
            addedDynamicCopyLink = true;
            if (copySrcFromInterfaceField) {
              auto srcNameIt = signalIdToName.find(copySrcSigId);
              auto dstNameIt = signalIdToName.find(destSigId);
              bool srcTopLevel =
                  srcNameIt != signalIdToName.end() &&
                  llvm::StringRef(srcNameIt->second).starts_with("sig_");
              bool dstTopLevel =
                  dstNameIt != signalIdToName.end() &&
                  llvm::StringRef(dstNameIt->second).starts_with("sig_");
              if (srcTopLevel && !dstTopLevel)
                childToParentFieldAddr[storeAddr] = copySrcAddr;
            }
          }
        };
        if (suppressTriStateMirrorStoreForField) {
          // Keep existing source->dest links intact. The suppression logic
          // already skips mirrored writes and derives retained values from
          // tri-state rule state; destructively removing links can leave
          // source fanout empty in long-running monitor loops.
          if (useAggressiveTriStateMirrorSuppression) {
            auto mirrorIt = interfaceFieldPropagation.find(fieldSigId);
            if (mirrorIt != interfaceFieldPropagation.end()) {
              for (SignalId mirrorSigId : mirrorIt->second)
                addDynamicLink(mirrorSigId);
            }
          }
        } else {
          addDynamicLink(fieldSigId);
        }
      }

      Type storeLLVMType = storeOp.getValue().getType();
      if (traceInterfaceStore) {
        llvm::StringRef sigName = "<unknown>";
        auto nameIt = signalIdToName.find(fieldSigId);
        if (nameIt != signalIdToName.end())
          sigName = nameIt->second;
        llvm::SmallString<64> rawBits;
        if (storeVal.isX())
          rawBits = "X";
        else
          storeVal.getAPInt().toString(rawBits, /*Radix=*/2, /*Signed=*/false);
        llvm::errs() << "[IFACE-STORE] proc=" << procId << " addr=0x"
                     << llvm::format_hex(storeAddr, 16) << " sig=" << fieldSigId
                     << " (" << sigName << ") rawWidth="
                     << getTypeWidth(storeLLVMType) << " raw=" << rawBits;
        if (copySrcSigId != 0) {
          llvm::errs() << " copySrc=" << copySrcSigId;
          if (copySrcAddr != 0)
            llvm::errs() << "@0x" << llvm::format_hex(copySrcAddr, 16);
          if (copySrcFromInterfaceField)
            llvm::errs() << " fieldCopy=1";
          if (addedDynamicCopyLink)
            llvm::errs() << " linked=1";
        }
        if (suppressTriStateMirrorStoreForField)
          llvm::errs() << " suppressed=1";
        llvm::errs() << "\n";
      }

      auto normalizeInterfaceStoreValue = [&](SignalId targetSigId,
                                              InterpretedValue rawVal,
                                              bool sourceAlreadySignalEncoded =
                                                  false)
          -> InterpretedValue {
        const SignalValue &targetCurrent = scheduler.getSignalValue(targetSigId);
        unsigned targetWidth = targetCurrent.getWidth();
        if (rawVal.isX())
          return InterpretedValue::makeX(targetWidth);

        APInt bits = rawVal.getAPInt();

        // LLVM stores for 4-state fields commonly use struct<(value, unknown)>
        // layout, while FourStateStruct signals encode bits as
        // {value(high half), unknown(low half)}. Normalize to signal encoding.
        if (!sourceAlreadySignalEncoded &&
            scheduler.getSignalEncoding(targetSigId) ==
                SignalEncoding::FourStateStruct) {
          bool converted = false;
          if (auto llvmStructTy = dyn_cast<LLVM::LLVMStructType>(storeLLVMType)) {
            auto body = llvmStructTy.getBody();
            if (body.size() == 2) {
              unsigned valueWidth = getTypeWidth(body[0]);
              unsigned unknownWidth = getTypeWidth(body[1]);
              if (valueWidth == unknownWidth && valueWidth * 2 == targetWidth) {
                APInt valueBits = bits.extractBits(valueWidth, 0);
                APInt unknownBits = bits.extractBits(unknownWidth, valueWidth);
                APInt encoded = APInt::getZero(targetWidth);
                safeInsertBits(encoded, unknownBits, 0);
                safeInsertBits(encoded, valueBits, unknownWidth);
                bits = encoded;
                converted = true;
              }
            }
          }
          if (!converted) {
            unsigned logicalWidth = targetWidth / 2;
            if (bits.getBitWidth() <= logicalWidth) {
              // Scalar known value -> four-state struct encoding:
              // low half = unknown (0), high half = value bits.
              APInt valueBits = bits;
              if (valueBits.getBitWidth() < logicalWidth)
                valueBits = valueBits.zext(logicalWidth);
              else if (valueBits.getBitWidth() > logicalWidth)
                valueBits = valueBits.trunc(logicalWidth);
              APInt encoded = APInt::getZero(targetWidth);
              safeInsertBits(encoded, valueBits, logicalWidth);
              bits = encoded;
            }
          }
        }
        if (bits.getBitWidth() < targetWidth)
          bits = bits.zext(targetWidth);
        else if (bits.getBitWidth() > targetWidth)
          bits = bits.trunc(targetWidth);
        return InterpretedValue(bits);
      };

      // Debug: track stores to 64-bit interface fields (PWDATA, PADDR, etc.)
      LLVM_DEBUG({
        const SignalValue &dbgCur = scheduler.getSignalValue(fieldSigId);
        if (dbgCur.getWidth() == 64 && !storeVal.isX() &&
            storeVal.getUInt64() != 0) {
          llvm::dbgs() << "[IFACE-STORE] sig=" << fieldSigId
                       << " addr=0x" << llvm::format_hex(storeAddr, 10)
                       << " val=" << storeVal.getUInt64();
          auto revIt = childToParentFieldAddr.find(storeAddr);
          if (revIt != childToParentFieldAddr.end())
            llvm::dbgs() << " (reverse→parent@0x"
                         << llvm::format_hex(revIt->second, 10) << ")";
          auto propIt = interfaceFieldPropagation.find(fieldSigId);
          if (propIt != interfaceFieldPropagation.end())
            llvm::dbgs() << " (fwd→" << propIt->second.size() << " children)";
          llvm::dbgs() << "\n";
        }
      });
      const SignalValue &current = scheduler.getSignalValue(fieldSigId);
      InterpretedValue currentVal = InterpretedValue::fromSignalValue(current);

      InterpretedValue driveVal = normalizeInterfaceStoreValue(
          fieldSigId, storeVal, /*sourceAlreadySignalEncoded=*/false);
      if (suppressTriStateMirrorStoreForField) {
        driveVal = currentVal;
        bool shouldDeriveFromRule = currentVal.isX();
        // Derive the retained destination value directly from tri-state
        // cond/src state when possible to avoid stale bus-mirror latching.
        // Apply rule-derivation when the destination is unknown, or when the
        // tri-state condition just transitioned from driving (1) to released
        // (0). Re-deriving on every suppressed mirror store while cond stays 0
        // can clobber externally observed bus values.
        auto getSignalValueForRule = [&](SignalId sigId) -> InterpretedValue {
          auto pendingIt = pendingEpsilonDrives.find(sigId);
          if (pendingIt != pendingEpsilonDrives.end())
            return pendingIt->second;
          return InterpretedValue::fromSignalValue(
              scheduler.getSignalValue(sigId));
        };
        auto normalizeRuleValue = [&](InterpretedValue value)
            -> InterpretedValue {
          unsigned targetWidth = current.getWidth();
          if (value.isX())
            return InterpretedValue::makeX(targetWidth);
          if (value.getWidth() == targetWidth)
            return value;
          APInt bits = value.getAPInt();
          if (bits.getBitWidth() < targetWidth)
            bits = bits.zext(targetWidth);
          else if (bits.getBitWidth() > targetWidth)
            bits = bits.trunc(targetWidth);
          return InterpretedValue(bits);
        };

        bool derivedFromRule = false;
        bool requestRuleDerivation = shouldDeriveFromRule;
        for (const auto &rule : interfaceTriStateRules) {
          if (rule.destSigId != fieldSigId)
            continue;
          InterpretedValue condVal = getSignalValueForRule(rule.condSigId);
          if (condVal.isX() || rule.condBitIndex >= condVal.getWidth())
            continue;
          bool condTrue = condVal.getAPInt()[rule.condBitIndex];
          bool hadPrevCond = interfaceTriStateCondSeen.count(fieldSigId) != 0;
          bool prevCondTrue =
              hadPrevCond ? interfaceTriStateCondLastValue.lookup(fieldSigId)
                          : true;
          bool condTransitionToFalse =
              !condTrue && (!hadPrevCond || prevCondTrue);
          interfaceTriStateCondLastValue[fieldSigId] = condTrue;
          interfaceTriStateCondSeen.insert(fieldSigId);
          if (condTransitionToFalse)
            requestRuleDerivation = true;

          if (!requestRuleDerivation)
            break;

          InterpretedValue selectedVal =
              condTrue ? getSignalValueForRule(rule.srcSigId) : rule.elseValue;
          driveVal = normalizeRuleValue(selectedVal);
          derivedFromRule = true;
          break;
        }

        if (requestRuleDerivation && !derivedFromRule &&
            !useAggressiveTriStateMirrorSuppression &&
            driveVal.isX() && block) {
          // Preserve legacy behavior for instance-scoped topologies.
          unsigned targetWidth = current.getWidth();
          unsigned numBytes = (targetWidth + 7) / 8;
          if (numBytes > 0 && offset + numBytes <= block->size) {
            APInt bits(targetWidth, 0);
            for (unsigned i = 0; i < numBytes; ++i) {
              unsigned bitOffset = i * 8;
              if (bitOffset >= targetWidth)
                break;
              APInt byteBits(targetWidth,
                             static_cast<uint64_t>(block->data[offset + i]));
              bits |= byteBits.shl(bitOffset);
            }
            driveVal = InterpretedValue(bits);
          }
        }
      }

      // Only drive if the value actually changed — prevents zero-delta loops
      // in always_comb processes that copy interface fields bidirectionally.
      SignalValue newSigVal = driveVal.toSignalValue();
      static bool traceI3CIfaceProp = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_IFACE_PROP");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      auto shouldTraceI3CIfaceSig = [&](SignalId sigId) -> bool {
            auto nameIt = signalIdToName.find(sigId);
            if (nameIt == signalIdToName.end())
              return false;
            llvm::StringRef name = nameIt->second;
            return name.contains("i3c_controller_agent_bfm_0."
                             "i3c_controller_agent_bfm_h.sig_6.field_5") ||
               name.contains("i3c_target_agent_bfm_0."
                             "i3c_target_agent_bfm_h.sig_6.field_5") ||
               name.contains("i3c_controller_agent_bfm_0."
                             "i3c_controller_agent_bfm_h.sig_6.field_2") ||
               name.contains("i3c_target_agent_bfm_0."
                             "i3c_target_agent_bfm_h.sig_6.field_2") ||
               name.contains("sig_0.field_0") || name.contains("sig_0.field_1") ||
               name.contains("sig_0.field_2") || name.contains("sig_0.field_4") ||
               name.contains("sig_1.field_0") || name.contains("sig_1.field_1") ||
               name.contains("sig_1.field_2") || name.contains("sig_1.field_4");
          };
      auto traceI3CIfacePropUpdate =
          [&](llvm::StringRef tag, SignalId srcSigId, SignalId dstSigId,
              InterpretedValue oldVal, InterpretedValue newVal) {
            if (!traceI3CIfaceProp)
              return;
            if (!shouldTraceI3CIfaceSig(srcSigId) &&
                !shouldTraceI3CIfaceSig(dstSigId))
              return;
            static uint64_t printed = 0;
            if (printed >= 4096)
              return;
            ++printed;
            auto formatBits = [](InterpretedValue value,
                                 llvm::SmallString<64> &out) {
              if (value.isX()) {
                out = "X";
                return;
              }
              value.getAPInt().toString(out, /*Radix=*/2, /*Signed=*/false);
            };
            llvm::SmallString<64> oldBits;
            llvm::SmallString<64> newBits;
            formatBits(oldVal, oldBits);
            formatBits(newVal, newBits);
            auto srcNameIt = signalIdToName.find(srcSigId);
            auto dstNameIt = signalIdToName.find(dstSigId);
            llvm::StringRef srcName =
                srcNameIt != signalIdToName.end() ? srcNameIt->second : "?";
            llvm::StringRef dstName =
                dstNameIt != signalIdToName.end() ? dstNameIt->second : "?";
            SimTime now = scheduler.getCurrentTime();
            llvm::errs() << "[I3C-IFACE-PROP] " << tag << " proc=" << procId
                         << " t=" << now.realTime << " d=" << now.deltaStep
                         << " src=" << srcSigId << " (" << srcName << ")"
                         << " dst=" << dstSigId << " (" << dstName << ")"
                         << " old=" << oldBits << " new=" << newBits << "\n";
          };
      if (current == newSigVal) {
        if (traceInterfaceStore) {
          llvm::StringRef sigName = "<unknown>";
          auto nameIt = signalIdToName.find(fieldSigId);
          if (nameIt != signalIdToName.end())
            sigName = nameIt->second;
          llvm::SmallString<64> curBits;
          llvm::SmallString<64> normBits;
          InterpretedValue curVal = InterpretedValue::fromSignalValue(current);
          if (curVal.isX())
            curBits = "X";
          else
            curVal.getAPInt().toString(curBits, /*Radix=*/2, /*Signed=*/false);
          if (driveVal.isX())
            normBits = "X";
          else
            driveVal.getAPInt().toString(normBits, /*Radix=*/2,
                                         /*Signed=*/false);
          llvm::errs() << "[IFACE-STORE] unchanged sig=" << fieldSigId << " ("
                       << sigName << ") cur=" << curBits
                       << " norm=" << normBits << "\n";
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.store: shadow signal " << fieldSigId
                   << " unchanged, skipping drive\n");
      } else {
        if (traceInterfaceStore) {
          llvm::StringRef sigName = "<unknown>";
          auto nameIt = signalIdToName.find(fieldSigId);
          if (nameIt != signalIdToName.end())
            sigName = nameIt->second;
          llvm::SmallString<64> curBits;
          llvm::SmallString<64> normBits;
          InterpretedValue curVal = InterpretedValue::fromSignalValue(current);
          if (curVal.isX())
            curBits = "X";
          else
            curVal.getAPInt().toString(curBits, /*Radix=*/2, /*Signed=*/false);
          if (driveVal.isX())
            normBits = "X";
          else
            driveVal.getAPInt().toString(normBits, /*Radix=*/2,
                                         /*Signed=*/false);
          llvm::errs() << "[IFACE-STORE] update sig=" << fieldSigId << " ("
                       << sigName << ") cur=" << curBits
                       << " norm=" << normBits << "\n";
        }
        pendingEpsilonDrives[fieldSigId] = driveVal;
        scheduler.updateSignal(fieldSigId, newSigVal);
        traceI3CIfacePropUpdate("direct", fieldSigId, fieldSigId,
                                InterpretedValue::fromSignalValue(current),
                                driveVal);
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.store: drove interface shadow signal "
                   << fieldSigId << " at 0x"
                   << llvm::format_hex(storeAddr, 16) << "\n");

        // Propagate to linked child BFM interface field signals.
        // Also update backing memory so that loads from the struct see
        // the new value (not just signals).
        auto isSameBlockParentRelay = [&](SignalId srcSigId,
                                          SignalId dstSigId) -> bool {
          auto srcAddrIt = fieldSignalToAddr.find(srcSigId);
          auto dstAddrIt = fieldSignalToAddr.find(dstSigId);
          if (srcAddrIt == fieldSignalToAddr.end() ||
              dstAddrIt == fieldSignalToAddr.end())
            return false;
          uint64_t srcOff = 0, dstOff = 0;
          MemoryBlock *srcBlk = findBlockByAddress(srcAddrIt->second, srcOff);
          MemoryBlock *dstBlk = findBlockByAddress(dstAddrIt->second, dstOff);
          if (!srcBlk || !dstBlk || srcBlk != dstBlk)
            return false;
          auto srcPropIt = interfaceFieldPropagation.find(srcSigId);
          auto dstPropIt = interfaceFieldPropagation.find(dstSigId);
          if (srcPropIt == interfaceFieldPropagation.end() ||
              dstPropIt == interfaceFieldPropagation.end() ||
              dstPropIt->second.empty())
            return false;
          return true;
        };

        auto hasUnknownPayload = [&](SignalId sourceSigId,
                                     InterpretedValue value) -> bool {
          if (value.isX())
            return false;
          if (scheduler.getSignalEncoding(sourceSigId) !=
              SignalEncoding::FourStateStruct)
            return false;
          if (value.getWidth() < 2 || (value.getWidth() % 2) != 0)
            return false;
          unsigned logicalWidth = value.getWidth() / 2;
          APInt unknownBits = value.getAPInt().extractBits(logicalWidth, 0);
          return unknownBits != 0;
        };

        auto propagateRawValueToSignal =
            [&](SignalId sourceSigId, SignalId targetSigId,
                InterpretedValue rawPropagationVal,
                bool sourceAlreadySignalEncoded) {
          const SignalValue &targetCurrent =
              scheduler.getSignalValue(targetSigId);
          unsigned targetW = targetCurrent.getWidth();
          InterpretedValue targetDriveVal = normalizeInterfaceStoreValue(
              targetSigId, rawPropagationVal, sourceAlreadySignalEncoded);
          SignalValue targetNewVal = targetDriveVal.toSignalValue();
          if (targetCurrent == targetNewVal)
            return;
          pendingEpsilonDrives[targetSigId] = targetDriveVal;

          // Synchronous update: immediately set the signal value and trigger
          // any processes that have already set up their event waits.
          scheduler.updateSignal(targetSigId, targetNewVal);
          traceI3CIfacePropUpdate("prop", sourceSigId, targetSigId,
                                  InterpretedValue::fromSignalValue(targetCurrent),
                                  targetDriveVal);

          // Write to backing memory (synchronous — needed for llvm.load).
          auto tgtAddrIt = fieldSignalToAddr.find(targetSigId);
          if (tgtAddrIt != fieldSignalToAddr.end()) {
            uint64_t tgtAddr = tgtAddrIt->second;
            uint64_t tgtOff = 0;
            MemoryBlock *tgtBlock = findBlockByAddress(tgtAddr, tgtOff);
            unsigned tgtStoreSize = (targetW + 7) / 8;
            if (tgtBlock && tgtOff + tgtStoreSize <= tgtBlock->size) {
              uint64_t tgtBaseAddr = tgtAddr - tgtOff;
              auto &byteInit = interfaceMemoryByteInitMask[tgtBaseAddr];
              if (byteInit.size() != tgtBlock->size)
                byteInit.assign(tgtBlock->size, 0);
              std::fill(byteInit.begin() + tgtOff,
                        byteInit.begin() + tgtOff + tgtStoreSize,
                        targetDriveVal.isX() ? uint8_t{0} : uint8_t{1});
              if (!targetDriveVal.isX()) {
                APInt bits = targetDriveVal.getAPInt();
                if (bits.getBitWidth() < tgtStoreSize * 8)
                  bits = bits.zext(tgtStoreSize * 8);
                for (unsigned i = 0; i < tgtStoreSize; ++i)
                  tgtBlock->data[tgtOff + i] =
                      bits.extractBitsAsZExtValue(8, i * 8);
                tgtBlock->initialized = true;
              }
            }
          }
        };

        auto propagateToSignal = [&](SignalId targetSigId) {
          InterpretedValue propagationVal = storeVal;
          bool sourceAlreadySignalEncoded = false;
          if (hasUnknownPayload(fieldSigId, driveVal) &&
              isSameBlockParentRelay(fieldSigId, targetSigId)) {
            propagationVal = driveVal;
            sourceAlreadySignalEncoded = true;
          }
          propagateRawValueToSignal(fieldSigId, targetSigId, propagationVal,
                                    sourceAlreadySignalEncoded);
        };

        auto propagateFromSignal = [&](SignalId sourceSigId,
                                       SignalId targetSigId) {
          InterpretedValue sourceVal;
          auto pendingIt = pendingEpsilonDrives.find(sourceSigId);
          if (pendingIt != pendingEpsilonDrives.end())
            sourceVal = pendingIt->second;
          else
            sourceVal = InterpretedValue::fromSignalValue(
                scheduler.getSignalValue(sourceSigId));
          propagateRawValueToSignal(
              sourceSigId, targetSigId, sourceVal,
              /*sourceAlreadySignalEncoded=*/true);
        };

        // Forward propagation only (no cross-sibling fan-out).
        // Cross-sibling propagation was removed because auto-linked
        // entries in interfaceFieldPropagation can contain cross-field
        // links (e.g. PWRITE → PADDR), causing field contamination.
        auto propIt = interfaceFieldPropagation.find(fieldSigId);
        if (propIt != interfaceFieldPropagation.end()) {
          for (SignalId childSigId : propIt->second)
            propagateToSignal(childSigId);
          // Cascade one level for intra-interface linked signals only.
          // Only signals that received intra-interface links need cascading.
          // Normal BFM fields already get sibling propagation via the
          // reverse propagation handler below, so cascading them here
          // would cause double-propagation and corrupt edge detection.
          for (SignalId childSigId : propIt->second) {
            bool parentRelayCascade =
                isSameBlockParentRelay(fieldSigId, childSigId);
            if (!intraLinkedSignals.count(childSigId) && !parentRelayCascade)
              continue;
            auto childPropIt = interfaceFieldPropagation.find(childSigId);
            if (childPropIt != interfaceFieldPropagation.end()) {
              for (SignalId grandchildId : childPropIt->second) {
                if (grandchildId != fieldSigId)
                  propagateFromSignal(childSigId, grandchildId);
              }
            }
          }
        }

        // Reverse propagation: when a CHILD interface field is written,
        // propagate UP to the parent field, then forward from the parent
        // to all OTHER children. This ensures sibling BFMs (e.g., monitor
        // and driver on the same bus) see each other's writes.
        // The value-changed check in propagateToSignal prevents loops.
        auto childAddrIt = fieldSignalToAddr.find(fieldSigId);
        if (childAddrIt != fieldSignalToAddr.end()) {
          auto parentAddrIt =
              childToParentFieldAddr.find(childAddrIt->second);
          if (parentAddrIt != childToParentFieldAddr.end()) {
            auto parentFieldIt =
                interfaceFieldSignals.find(parentAddrIt->second);
            if (parentFieldIt != interfaceFieldSignals.end()) {
              SignalId parentFieldSigId = parentFieldIt->second;
              // Update parent shadow signal (skips if unchanged).
              const SignalValue &parentCurrent =
                  scheduler.getSignalValue(parentFieldSigId);
              if (!(parentCurrent == newSigVal)) {
                propagateToSignal(parentFieldSigId);
                // Forward-propagate from parent to all children
                // (the originating child will be skipped by the
                // value-changed check since it already has the value).
                auto parentPropIt =
                    interfaceFieldPropagation.find(parentFieldSigId);
                if (parentPropIt != interfaceFieldPropagation.end()) {
                  for (SignalId siblingId : parentPropIt->second) {
                    if (siblingId != fieldSigId)
                      propagateToSignal(siblingId);
                  }
                  // Cascade for intra-linked siblings only.
                  for (SignalId siblingId : parentPropIt->second) {
                    if (siblingId == fieldSigId)
                      continue;
                    if (!intraLinkedSignals.count(siblingId))
                      continue;
                    auto sibPropIt =
                        interfaceFieldPropagation.find(siblingId);
                    if (sibPropIt != interfaceFieldPropagation.end()) {
                      for (SignalId grandchildId : sibPropIt->second) {
                        if (grandchildId != fieldSigId &&
                            grandchildId != parentFieldSigId)
                          propagateFromSignal(siblingId, grandchildId);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        // Re-evaluate synthetic tri-state field rules (e.g. scl/sda output
        // enable muxing) that depend on the updated source signal.
        applyInterfaceTriStateRules(fieldSigId);
      }
    }
  }

  // Check if any processes are waiting on memory events at this location.
  // If the stored value changed, wake those processes.
  checkMemoryEventWaiters();

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMGEP(ProcessId procId,
                                                        LLVM::GEPOp gepOp) {
  // Get the base pointer value
  InterpretedValue baseVal = getValue(procId, gepOp.getBase());
  if (baseVal.isX()) {
    setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
    return success();
  }

  uint64_t baseAddr = baseVal.getUInt64();

  // Detect null pointer dereference: GEP on NULL (address 0) or near-null
  // addresses (< 0x1000) indicates a null object handle dereference.
  if (baseAddr < 0x1000 && baseAddr != 0) {
    // Non-zero but very small address - likely result of GEP on null
    LLVM_DEBUG(llvm::dbgs() << "  llvm.getelementptr: near-null base address 0x"
                            << llvm::format_hex(baseAddr, 16) << " -> X\n");
    setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
    return success();
  }
  uint64_t offset = 0;

  // Get the element type
  Type elemType = gepOp.getElemType();

  // Process indices using the GEPIndicesAdaptor
  auto indices = gepOp.getIndices();
  Type currentType = elemType;

  size_t idx = 0;
  for (auto indexValue : indices) {
    int64_t indexVal = 0;

    // Check if this is a constant index (IntegerAttr) or dynamic (Value)
    if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(indexValue)) {
      indexVal = intAttr.getInt();
    } else if (auto dynamicIdx = llvm::dyn_cast_if_present<Value>(indexValue)) {
      InterpretedValue dynVal = getValue(procId, dynamicIdx);
      if (dynVal.isX()) {
        setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
        return success();
      }
      indexVal = static_cast<int64_t>(dynVal.getUInt64());
    }

    if (idx == 0) {
      // First index: scales by the size of the pointed-to type
      offset += indexVal * getLLVMTypeSizeForGEP(elemType);
    } else if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
      // Struct indexing: accumulate offsets of previous fields.
      // Use GEP-aligned sizes so sub-byte struct fields (e.g. struct<(i3,i3)>)
      // occupy their correct byte span, keeping subsequent field offsets right.
      auto body = structType.getBody();
      for (int64_t i = 0; i < indexVal && static_cast<size_t>(i) < body.size(); ++i) {
        offset += getLLVMTypeSizeForGEP(body[i]);
      }
      if (static_cast<size_t>(indexVal) < body.size()) {
        currentType = body[indexVal];
      }
    } else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
      // Array indexing: multiply by element size
      offset += indexVal * getLLVMTypeSizeForGEP(arrayType.getElementType());
      currentType = arrayType.getElementType();
    } else {
      // For other types, treat as array of the current type
      offset += indexVal * getLLVMTypeSizeForGEP(currentType);
    }
    ++idx;
  }

  uint64_t resultAddr = baseAddr + offset;
  setValue(procId, gepOp.getResult(), InterpretedValue(resultAddr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.getelementptr: base=0x"
                          << llvm::format_hex(baseAddr, 16) << " offset="
                          << offset << " result=0x"
                          << llvm::format_hex(resultAddr, 16) << "\n");

  return success();
}
