//===- LLHDProcessInterpreterCallIndirect.cpp - call_indirect handling -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file isolates func.call_indirect execution logic from the main
// LLHDProcessInterpreter.cpp translation unit to improve maintainability.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "circt/Runtime/CirctSimABI.h"
#include "circt/Runtime/MooreRuntime.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

static bool traceUvmFactoryByTypeEnabled() {
  const char *env = std::getenv("CIRCT_SIM_TRACE_UVM_FACTORY_BY_TYPE");
  return env && env[0] != '\0' && env[0] != '0';
}

static bool disableUvmFactoryByTypeFastPath() {
  const char *env = std::getenv("CIRCT_SIM_ENABLE_UVM_FACTORY_BYTYPE_FASTPATH");
  bool enabled = env && env[0] != '\0' && env[0] != '0';
  // Disabled by default until object-init correctness is proven across
  // full UVM startup; callers can opt in explicitly for performance testing.
  return !enabled;
}

static bool disableUvmFactoryFastPaths() {
  const char *env = std::getenv("CIRCT_SIM_ENABLE_UVM_FACTORY_FASTPATH");
  bool enabled = env && env[0] != '\0' && env[0] != '0';
  // Disabled by default until full UVM startup semantics are proven
  // equivalent to the canonical MLIR implementation.
  return !enabled;
}

// Optional function-call tracing for focused runtime diagnosis.
// Enable with CIRCT_SIM_TRACE_CALL_FILTER. Example:
//   CIRCT_SIM_TRACE_CALL_FILTER=uvm_tlm_analysis_fifo::write,uvm_tlm_fifo::get
// If the env var is set to "1", all function calls are traced.
static void maybeTraceFilteredCall(ProcessId procId, llvm::StringRef callKind,
                                   llvm::StringRef calleeName, int64_t nowFs,
                                   uint64_t deltaStep) {
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

static bool isNativeAnalysisWriteCalleeCI(llvm::StringRef calleeName) {
  if (!calleeName.contains("::write") || calleeName.contains("write_m_"))
    return false;
  return calleeName.contains("analysis_port") ||
         calleeName.contains("uvm_tlm_if_base");
}

static uint64_t sequencerProcKey(ProcessId procId) {
  return 0xF1F1000000000000ULL | static_cast<uint64_t>(procId);
}

static void decrementRecursionDepthEntry(ProcessExecutionState &state,
                                         Operation *funcKey,
                                         uint64_t arg0Val) {
  auto funcIt = state.recursionVisited.find(funcKey);
  if (funcIt == state.recursionVisited.end())
    return;
  auto &depthMap = funcIt->second;
  auto argIt = depthMap.find(arg0Val);
  if (argIt == depthMap.end())
    return;
  if (argIt->second > 1) {
    --argIt->second;
    return;
  }
  depthMap.erase(argIt);
  if (depthMap.empty())
    state.recursionVisited.erase(funcIt);
}

static unsigned writeConfigDbBytesToNativeMemory(
    uint64_t addr, uint64_t nativeOffset, size_t nativeSize,
    const std::vector<uint8_t> &valueData, unsigned requestedBytes,
    bool zeroFillMissing) {
  if (requestedBytes == 0 || nativeOffset >= nativeSize)
    return 0;

  size_t availableBytes = nativeSize - static_cast<size_t>(nativeOffset);
  unsigned maxWritable =
      static_cast<unsigned>(std::min<size_t>(requestedBytes, availableBytes));
  unsigned copyBytes =
      std::min(maxWritable, static_cast<unsigned>(valueData.size()));

  if (copyBytes > 0)
    std::memcpy(reinterpret_cast<void *>(addr), valueData.data(), copyBytes);
  if (zeroFillMissing && maxWritable > copyBytes)
    std::memset(reinterpret_cast<void *>(addr + copyBytes), 0,
                maxWritable - copyBytes);
  return maxWritable;
}

static unsigned writeConfigDbBytesToMemoryBlock(
    MemoryBlock *block, uint64_t offset, const std::vector<uint8_t> &valueData,
    unsigned requestedBytes, bool zeroFillMissing) {
  if (!block || requestedBytes == 0)
    return 0;

  size_t start = static_cast<size_t>(offset);
  if (start >= block->size)
    return 0;

  size_t availableBytes = block->size - start;
  unsigned maxWritable =
      static_cast<unsigned>(std::min<size_t>(requestedBytes, availableBytes));
  unsigned copyBytes =
      std::min(maxWritable, static_cast<unsigned>(valueData.size()));

  if (copyBytes > 0)
    std::memcpy(block->bytes() + start, valueData.data(), copyBytes);
  if (zeroFillMissing && maxWritable > copyBytes)
    std::memset(block->bytes() + start + copyBytes, 0,
                maxWritable - copyBytes);
  if (maxWritable > 0)
    block->initialized = true;
  return maxWritable;
}

// Compute a native UVM port connection count by traversing the interpreter's
// port connection graph and counting terminal providers.
static void collectNativeUvmPortTerminals(
    const llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 2>>
        &analysisPortConnections,
    uint64_t portAddr, llvm::SmallVectorImpl<uint64_t> &terminals) {
  terminals.clear();
  if (portAddr == 0)
    return;

  auto seedIt = analysisPortConnections.find(portAddr);
  if (seedIt == analysisPortConnections.end() || seedIt->second.empty())
    return;

  llvm::SmallVector<uint64_t, 8> worklist(seedIt->second.begin(),
                                          seedIt->second.end());
  llvm::DenseSet<uint64_t> visited;
  llvm::DenseSet<uint64_t> emittedTerminals;

  while (!worklist.empty()) {
    uint64_t addr = worklist.pop_back_val();
    if (addr == 0 || !visited.insert(addr).second)
      continue;

    auto it = analysisPortConnections.find(addr);
    if (it != analysisPortConnections.end() && !it->second.empty()) {
      for (uint64_t next : it->second)
        worklist.push_back(next);
      continue;
    }
    if (emittedTerminals.insert(addr).second)
      terminals.push_back(addr);
  }
}

static int32_t getNativeUvmPortSize(
    const llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 2>>
        &analysisPortConnections,
    uint64_t portAddr) {
  llvm::SmallVector<uint64_t, 4> terminalProviders;
  collectNativeUvmPortTerminals(analysisPortConnections, portAddr,
                                terminalProviders);
  if (!terminalProviders.empty())
    return static_cast<int32_t>(terminalProviders.size());
  // Fallback: preserve "connected means non-zero" even if graph is cyclic.
  auto seedIt = analysisPortConnections.find(portAddr);
  if (seedIt == analysisPortConnections.end())
    return 0;
  return static_cast<int32_t>(seedIt->second.size());
}

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

LogicalResult LLHDProcessInterpreter::interpretFuncCallIndirect(
    ProcessId procId, mlir::func::CallIndirectOp callIndirectOp) {
    // The callee is the first operand (function pointer)
    Value calleeValue = callIndirectOp.getCallee();
    ModuleOp callSiteModule = callIndirectOp->getParentOfType<ModuleOp>();
    InterpretedValue funcPtrVal = getValue(procId, calleeValue);
    bool traceSeq = traceSeqEnabled;
    static bool enableUvmAnalysisNativeInterceptors = []() {
      const char *env =
          std::getenv("CIRCT_SIM_ENABLE_UVM_ANALYSIS_NATIVE_INTERCEPTS");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    bool sawResolvedTarget = false;
    std::string resolvedTargetName;
    auto noteResolvedTarget = [&](llvm::StringRef name) {
      if (name.empty())
        return;
      sawResolvedTarget = true;
      resolvedTargetName = name.str();
    };
    auto noteRuntimeIndirectProfileOnExit = llvm::make_scope_exit([&]() {
      if (!jitRuntimeIndirectProfileEnabled)
        return;
      if (sawResolvedTarget)
        noteJitRuntimeIndirectResolvedTarget(procId, callIndirectOp,
                                            resolvedTargetName);
      else
        noteJitRuntimeIndirectUnresolved(procId, callIndirectOp);
    });

    // Install resolution cache entry on exit (after resolution completes).
    uint64_t ciCacheFuncAddr = 0;
    auto installResolutionCacheOnExit = llvm::make_scope_exit([&]() {
      if (sawResolvedTarget && ciCacheFuncAddr != 0 &&
          !callIndirectResolutionCache.count(ciCacheFuncAddr)) {
        callIndirectResolutionCache[ciCacheFuncAddr] = resolvedTargetName;
        ++ciResolutionCacheInstalls;
      }
    });

    // Early trace: log every call_indirect to detect analysis_port writes.
    if (traceAnalysisEnabled) {
      // Try to identify the callee from the SSA chain (GEP → vtable)
      auto castOp0 = calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      if (castOp0 && castOp0.getInputs().size() == 1) {
        auto loadOp0 = castOp0.getInputs()[0].getDefiningOp<LLVM::LoadOp>();
        if (loadOp0) {
          auto gepOp0 = loadOp0.getAddr().getDefiningOp<LLVM::GEPOp>();
          if (gepOp0) {
            auto baseLoad0 = gepOp0.getBase().getDefiningOp<LLVM::LoadOp>();
            if (baseLoad0) {
              auto objGep0 = baseLoad0.getAddr().getDefiningOp<LLVM::GEPOp>();
              if (objGep0) {
                if (auto structTy0 = dyn_cast<LLVM::LLVMStructType>(
                        objGep0.getElemType())) {
                  if (structTy0.isIdentified() &&
                      structTy0.getName().contains("analysis_port")) {
                    auto indices0 = gepOp0.getIndices();
                    int64_t slot0 = -1;
                    if (indices0.size() >= 2) {
                      if (auto ia = llvm::dyn_cast_if_present<IntegerAttr>(
                              indices0[indices0.size() - 1]))
                        slot0 = ia.getInt();
                    }
                    llvm::errs() << "[ANALYSIS-CI-ENTRY] struct="
                                 << structTy0.getName() << " slot=" << slot0
                                 << " funcPtrIsX=" << funcPtrVal.isX()
                                 << " funcAddr=0x"
                                 << (funcPtrVal.isX()
                                         ? std::string("X")
                                         : llvm::utohexstr(funcPtrVal.getUInt64()))
                                 << "\n";
                  }
                }
              }
            }
          }
        }
      }
    }

    // Throttle vtable dispatch warnings to prevent flooding stderr during
    // UVM initialization. Both X function pointers and unmapped addresses
    // share a single counter.
    auto emitVtableWarning = [&](StringRef reason) {
      static unsigned vtableWarnCount = 0;
      if (vtableWarnCount < 30) {
        ++vtableWarnCount;
        llvm::errs() << "[circt-sim] WARNING: virtual method call "
                     << "(func.call_indirect) failed: " << reason
                     << ". Callee operand: ";
        calleeValue.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
        llvm::errs() << " (type: " << calleeValue.getType() << ")\n";
      } else if (vtableWarnCount == 30) {
        ++vtableWarnCount;
        llvm::errs() << "[circt-sim] (suppressing further vtable warnings)\n";
      }
    };

    // Try to recover a direct callee symbol from:
    //   func.call_indirect (unrealized_conversion_cast (llvm.mlir.addressof @sym))
    auto tryGetDirectAddressOfCalleeName = [&]() -> std::optional<std::string> {
      auto castOp =
          calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      if (!castOp || castOp.getInputs().size() != 1)
        return std::nullopt;
      auto addrOfOp = castOp.getInputs().front().getDefiningOp<LLVM::AddressOfOp>();
      if (!addrOfOp)
        return std::nullopt;
      return addrOfOp.getGlobalName().str();
    };
    auto findDirectCoverageRuntimeCalleeInSymbol =
        [&](llvm::StringRef symbolName) -> std::optional<std::string> {
      ModuleOp moduleOp = callSiteModule ? callSiteModule : rootModule;
      if (!moduleOp)
        return std::nullopt;

      auto tryGetDirectSymbolFromCalleeValue =
          [&](Value calleeValue) -> std::optional<std::string> {
        Value base = calleeValue;
        if (auto castOp = base.getDefiningOp<UnrealizedConversionCastOp>()) {
          if (castOp.getNumOperands() == 1)
            base = castOp.getOperand(0);
        }
        auto addrOfOp = base.getDefiningOp<LLVM::AddressOfOp>();
        if (!addrOfOp)
          return std::nullopt;
        return addrOfOp.getGlobalName().str();
      };

      constexpr size_t kMaxVisitedSymbols = 128;
      std::unordered_set<std::string> visitedSymbols;
      SmallVector<std::string, 8> worklist;
      worklist.push_back(symbolName.str());

      while (!worklist.empty() && visitedSymbols.size() < kMaxVisitedSymbols) {
        std::string currentSymbol = std::move(worklist.back());
        worklist.pop_back();
        if (!visitedSymbols.insert(currentSymbol).second)
          continue;

        if (isCoverageRuntimeCallee(currentSymbol))
          return currentSymbol;

        auto enqueueSymbol = [&](StringRef calleeName) {
          if (calleeName.empty() || isCoverageRuntimeCallee(calleeName))
            return;
          if (visitedSymbols.size() >= kMaxVisitedSymbols)
            return;
          if (visitedSymbols.count(calleeName.str()))
            return;
          worklist.push_back(calleeName.str());
        };

        if (auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(currentSymbol)) {
          std::optional<std::string> hit;
          funcOp.walk([&](Operation *nestedOp) {
            if (auto llvmCall = dyn_cast<LLVM::CallOp>(nestedOp)) {
              if (auto callee = llvmCall.getCallee()) {
                if (isCoverageRuntimeCallee(*callee)) {
                  hit = callee->str();
                  return WalkResult::interrupt();
                }
                enqueueSymbol(*callee);
              } else {
                auto calleeOperands = llvmCall.getCalleeOperands();
                if (!calleeOperands.empty()) {
                  if (auto directCallee = tryGetDirectSymbolFromCalleeValue(
                          calleeOperands.front())) {
                    if (isCoverageRuntimeCallee(*directCallee)) {
                      hit = *directCallee;
                      return WalkResult::interrupt();
                    }
                    enqueueSymbol(*directCallee);
                  }
                }
              }
            } else if (auto funcCall = dyn_cast<func::CallOp>(nestedOp)) {
              if (isCoverageRuntimeCallee(funcCall.getCallee())) {
                hit = funcCall.getCallee().str();
                return WalkResult::interrupt();
              }
              enqueueSymbol(funcCall.getCallee());
            } else if (auto callIndirect =
                           dyn_cast<func::CallIndirectOp>(nestedOp)) {
              if (auto directCallee =
                      tryGetDirectSymbolFromCalleeValue(callIndirect.getCallee())) {
                if (isCoverageRuntimeCallee(*directCallee)) {
                  hit = *directCallee;
                  return WalkResult::interrupt();
                }
                enqueueSymbol(*directCallee);
              }
            }
            return WalkResult::advance();
          });
          if (hit)
            return hit;
        }

        if (auto llvmFuncOp =
                moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(currentSymbol)) {
          std::optional<std::string> hit;
          llvmFuncOp.walk([&](LLVM::CallOp nestedCall) {
            if (auto callee = nestedCall.getCallee()) {
              if (isCoverageRuntimeCallee(*callee)) {
                hit = callee->str();
                return WalkResult::interrupt();
              }
              enqueueSymbol(*callee);
            } else {
              auto calleeOperands = nestedCall.getCalleeOperands();
              if (!calleeOperands.empty()) {
                if (auto directCallee = tryGetDirectSymbolFromCalleeValue(
                        calleeOperands.front())) {
                  if (isCoverageRuntimeCallee(*directCallee)) {
                    hit = *directCallee;
                    return WalkResult::interrupt();
                  }
                  enqueueSymbol(*directCallee);
                }
              }
            }
            return WalkResult::advance();
          });
          if (hit)
            return hit;
        }
      }

      return std::nullopt;
    };

    auto setCallIndirectResults =
        [&](llvm::ArrayRef<InterpretedValue> values) {
          auto opResults = callIndirectOp.getResults();
          unsigned i = 0;
          for (; i < opResults.size() && i < values.size(); ++i)
            setValue(procId, opResults[i], values[i]);
          // Mismatched call_indirect signatures can request more results than
          // the resolved callee produces (e.g. via unrealized casts). Zero-fill
          // the tail to avoid stale values and out-of-bounds reads.
          for (; i < opResults.size(); ++i) {
            unsigned width = getTypeWidth(opResults[i].getType());
            setValue(procId, opResults[i],
                     InterpretedValue(llvm::APInt(width, 0)));
        }
      };
    auto isVtableGlobalSymbol = [&](llvm::StringRef globalName) -> bool {
      if (globalName.empty())
        return false;
      if (globalName.ends_with("__vtable__"))
        return true;
      ModuleOp lookupModule = callSiteModule ? callSiteModule : rootModule;
      if (!lookupModule)
        return false;
      if (auto globalOp = lookupModule.lookupSymbol<LLVM::GlobalOp>(globalName))
        return globalOp->hasAttr("circt.vtable_entries");
      if (lookupModule != rootModule && rootModule) {
        if (auto globalOp = rootModule.lookupSymbol<LLVM::GlobalOp>(globalName))
          return globalOp->hasAttr("circt.vtable_entries");
      }
      return false;
    };

    auto isPlausibleNativePointerValue = [](uint64_t addr) -> bool {
      if (addr == 0)
        return true;
      constexpr uint64_t kVirtualLo = 0x10000000ULL;
      constexpr uint64_t kVirtualHi = 0x20000000ULL;
      constexpr uint64_t kTaggedLo = 0xF0000000ULL;
      constexpr uint64_t kTaggedHi = 0x100000000ULL;
      constexpr uint64_t kMaxCanonicalUserPtr = 0x0000FFFFFFFFFFFFULL;
      if (addr >= kVirtualLo && addr < kVirtualHi)
        return true;
      if (addr >= kTaggedLo && addr < kTaggedHi)
        return true;
      return addr <= kMaxCanonicalUserPtr;
    };
    auto isKnownPointerAddress = [&](uint64_t addr) -> bool {
      if (addr == 0)
        return true;
      uint64_t off = 0;
      if (findMemoryBlockByAddress(addr, procId, &off))
        return true;
      if (findBlockByAddress(addr, off))
        return true;
      size_t nativeSize = 0;
      return findNativeMemoryBlockByAddress(addr, &off, &nativeSize);
    };
    auto isFastFalseUnsafeUvmPredicateCallee =
        [](llvm::StringRef calleeName) -> bool {
      if (!calleeName.starts_with("uvm_pkg::"))
        return false;
      bool isTypedCallbacksPredicate =
          calleeName.contains("uvm_typed_callbacks_") &&
          calleeName.ends_with("::m_am_i_a");
      bool isPoolExistsPredicate =
          calleeName.contains("uvm_pool") && calleeName.ends_with("::exists");
      return isTypedCallbacksPredicate || isPoolExistsPredicate;
    };
    auto fillNativeCallArgs = [&](llvm::ArrayRef<InterpretedValue> callArgs,
                                  mlir::TypeRange argTypes,
                                  llvm::StringRef calleeNameForNormalization,
                                  unsigned numArgs, uint64_t (&packed)[8],
                                  bool normalizePointerArgs,
                                  bool &forcePredicateFalse) -> bool {
      forcePredicateFalse = false;
      for (unsigned i = 0; i < numArgs; ++i) {
        uint64_t argVal = (i < callArgs.size()) ? callArgs[i].getUInt64() : 0;
        bool hasPointerType =
            i < argTypes.size() && isa<mlir::LLVM::LLVMPointerType>(argTypes[i]);
        // Some UVM call_indirect lowers carry `this` as i64 instead of ptr.
        // Treat arg0 as pointer-like for UVM methods so native pointer guards
        // still run and can safely demote bad payloads to interpreted fallback.
        bool isLikelyI64ThisArg =
            i < argTypes.size() && isa<mlir::IntegerType>(argTypes[i]) &&
            cast<mlir::IntegerType>(argTypes[i]).getWidth() == 64;
        bool isUvmMethodThisArg =
            i == 0 && isLikelyI64ThisArg &&
            calleeNameForNormalization.contains("uvm_pkg::") &&
            calleeNameForNormalization.contains("::");
        if (normalizePointerArgs && (hasPointerType || isUvmMethodThisArg)) {
          argVal = normalizeNativeCallPointerArg(procId,
                                                 calleeNameForNormalization,
                                                 argVal);
          bool strictUvmThisPointerCheck =
              i == 0 && calleeNameForNormalization.contains("uvm_pkg::");
          if (strictUvmThisPointerCheck && argVal != 0 &&
              !isKnownPointerAddress(argVal)) {
            static bool traceNativeCalls =
                std::getenv("CIRCT_AOT_TRACE_NATIVE_CALLS") != nullptr;
            if (traceNativeCalls) {
              llvm::errs() << "[AOT TRACE] call_indirect skip unknown-pointer"
                           << " callee=" << calleeNameForNormalization
                           << " arg" << i << "="
                           << llvm::format_hex(argVal, 16)
                           << " active_proc=" << activeProcessId << "\n";
            }
            return false;
          }
          bool unsafeUvmPredicateArg1 =
              i == 1 &&
              isFastFalseUnsafeUvmPredicateCallee(calleeNameForNormalization);
          if (unsafeUvmPredicateArg1 && argVal != 0) {
            bool knownAddr = isKnownPointerAddress(argVal);
            bool badPointerShape = !isPlausibleNativePointerValue(argVal);
            // These UVM predicates dereference arg1 directly. If arg1 is
            // non-zero and we cannot resolve it to tracked interpreter/native
            // storage, fast-false instead of native dereference.
            bool unknownUnmappedPtr = !knownAddr;
            if (badPointerShape || unknownUnmappedPtr) {
              static bool traceNativeCalls =
                  std::getenv("CIRCT_AOT_TRACE_NATIVE_CALLS") != nullptr;
              if (traceNativeCalls) {
                llvm::errs()
                    << "[AOT TRACE] call_indirect fast-false unsafe-predicate"
                    << " callee=" << calleeNameForNormalization << " arg" << i
                    << "=" << llvm::format_hex(argVal, 16)
                    << " known=" << (knownAddr ? 1 : 0)
                    << " bad_shape=" << (badPointerShape ? 1 : 0)
                    << " active_proc=" << activeProcessId << "\n";
              }
              forcePredicateFalse = true;
              return true;
            }
          }
          if (!isPlausibleNativePointerValue(argVal)) {
            static bool traceNativeCalls =
                std::getenv("CIRCT_AOT_TRACE_NATIVE_CALLS") != nullptr;
            if (traceNativeCalls) {
              llvm::errs() << "[AOT TRACE] call_indirect skip bad-pointer"
                           << " callee=" << calleeNameForNormalization
                           << " arg" << i << "="
                           << llvm::format_hex(argVal, 16)
                           << " active_proc=" << activeProcessId << "\n";
            }
            return false;
          }
        }
        packed[i] = argVal;
      }
      return true;
    };
    auto maybeTraceIndirectNative = [&](uint32_t fid, llvm::StringRef callee,
                                        bool isNativeEntry, unsigned numArgs,
                                        unsigned numResults,
                                        const uint64_t (&packed)[8]) {
      static bool traceNativeCalls =
          std::getenv("CIRCT_AOT_TRACE_NATIVE_CALLS") != nullptr;
      static uint64_t traceNativeCallLimit = []() -> uint64_t {
        if (const char *s = std::getenv("CIRCT_AOT_TRACE_NATIVE_LIMIT"))
          return static_cast<uint64_t>(std::strtoull(s, nullptr, 10));
        return 200;
      }();
      static uint64_t traceNativeCallCount = 0;
      if (!traceNativeCalls || traceNativeCallCount >= traceNativeCallLimit)
        return;
      bool mayYield =
          compiledFuncFlags &&
          fid < numCompiledAllFuncs &&
          (compiledFuncFlags[fid] & CIRCT_FUNC_FLAG_MAY_YIELD);
      llvm::errs() << "[AOT TRACE] call_indirect dispatch="
                   << (isNativeEntry ? "native" : "trampoline")
                   << " fid=" << fid
                   << " callee=" << callee << " args=" << numArgs
                   << " rets=" << numResults
                   << " may_yield=" << (mayYield ? 1 : 0)
                   << " active_proc=" << activeProcessId << "\n";
      if (numArgs > 0)
        llvm::errs() << "            a0=" << llvm::format_hex(packed[0], 16)
                     << "\n";
      if (numArgs > 1)
        llvm::errs() << "            a1=" << llvm::format_hex(packed[1], 16)
                     << "\n";
      ++traceNativeCallCount;
    };
    auto shouldSkipMayYieldEntry = [&](uint32_t fid, bool isNativeEntry) {
      return shouldSkipMayYieldEntryDispatch(fid, isNativeEntry,
                                             activeProcessId);
    };
    auto shouldForceInterpretedFragileUvmCallee =
        [](llvm::StringRef calleeName) -> bool {
      // Keep known fragile UVM mutators/walkers interpreted even when
      // entry-table-eligible. These paths have shown startup/liveness
      // divergence under both native and trampoline entry dispatch.
      return calleeName.contains(
                 "uvm_pkg::uvm_phase::set_max_ready_to_end_iterations") ||
             calleeName.contains(
                 "uvm_pkg::uvm_phase_hopper::wait_for_waiters") ||
             calleeName.contains(
                 "uvm_pkg::uvm_sequence_base::clear_response_queue") ||
             calleeName.contains(
                 "uvm_pkg::uvm_component_proxy::get_immediate_children");
    };
    auto tryResolveAnalysisWriteTarget =
        [&](ModuleOp lookupModule, uint64_t selfAddr, llvm::StringRef traceTag)
        -> std::optional<std::pair<func::FuncOp, std::string>> {
      uint64_t vtableAddr = 0;
      bool haveVtableAddr =
          readObjectVTableAddress(selfAddr, vtableAddr, procId);

      if (!haveVtableAddr) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag << "] write-target lookup failed: "
                       << "object/self not found self=0x"
                       << llvm::format_hex(selfAddr, 0) << "\n";
        return std::nullopt;
      }

      auto globalIt = addressToGlobal.find(vtableAddr);
      if (globalIt == addressToGlobal.end()) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag << "] write-target lookup failed: "
                       << "vtable not in addressToGlobal addr=0x"
                       << llvm::format_hex(vtableAddr, 0) << "\n";
        return std::nullopt;
      }
      auto vtableBlockIt = globalMemoryBlocks.find(globalIt->second);
      if (vtableBlockIt == globalMemoryBlocks.end()) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag << "] write-target lookup failed: "
                       << "missing globalMemoryBlock for "
                       << globalIt->second << "\n";
        return std::nullopt;
      }

      auto &vtableBlock = vtableBlockIt->second;
      constexpr unsigned writeSlot = 11;
      unsigned slotOffset = writeSlot * 8;
      if (slotOffset + 8 > vtableBlock.size) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag << "] write-target lookup failed: "
                       << "slot11 OOB size=" << vtableBlock.size << "\n";
        return std::nullopt;
      }

      uint64_t writeFuncAddr = 0;
      for (unsigned i = 0; i < 8; ++i)
        writeFuncAddr |=
            static_cast<uint64_t>(vtableBlock[slotOffset + i]) << (i * 8);
      auto funcIt = addressToFunction.find(writeFuncAddr);
      if (funcIt == addressToFunction.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: " << traceTag
                   << " write func at vtable slot 11 (addr 0x"
                   << llvm::format_hex(writeFuncAddr, 16)
                   << ") not found\n");
        return std::nullopt;
      }

      auto writeFunc = lookupModule.lookupSymbol<func::FuncOp>(funcIt->second);
      if (!writeFunc) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag << "] write-target lookup failed: "
                       << "symbol missing for " << funcIt->second << "\n";
        return std::nullopt;
      }
      return std::make_optional(
          std::make_pair(writeFunc, funcIt->second));
    };
    auto dispatchAnalysisWriteTarget =
        [&](func::FuncOp targetFunc, llvm::StringRef targetName,
            llvm::StringRef traceTag, uint64_t selfAddr,
            const InterpretedValue &txArg) {
      if (traceAnalysisEnabled)
        llvm::errs() << "[" << traceTag << "] dispatching to " << targetName
                     << "\n";
      SmallVector<InterpretedValue, 2> impArgs;
      impArgs.push_back(InterpretedValue(llvm::APInt(64, selfAddr)));
      impArgs.push_back(txArg);
      SmallVector<InterpretedValue, 2> impResults;
      auto &callState = processStates[procId];
      ++callState.callDepth;
      (void)interpretFuncBody(procId, targetFunc, impArgs, impResults,
                              callIndirectOp);
      --callState.callDepth;
    };
    auto tryDispatchAnalysisWriteSelfFallback =
        [&](ModuleOp lookupModule, llvm::StringRef traceTag,
            llvm::StringRef baseWriteName, uint64_t selfAddr,
            const InterpretedValue &txArg) -> bool {
      if (selfAddr == 0)
        return false;
      auto writeTarget =
          tryResolveAnalysisWriteTarget(lookupModule, selfAddr, traceTag);
      if (!writeTarget)
        return false;
      if (writeTarget->second == baseWriteName) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[" << traceTag
                       << "] self fallback resolved base body "
                       << baseWriteName << ", keeping no-op\n";
        return false;
      }
      dispatchAnalysisWriteTarget(writeTarget->first, writeTarget->second,
                                  traceTag, selfAddr, txArg);
      return true;
    };

    if (funcPtrVal.isX()) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: callee is X "
                              << "(uninitialized vtable pointer)\n");
      if (auto directCallee = tryGetDirectAddressOfCalleeName()) {
        noteResolvedTarget(*directCallee);
        if (isCoverageRuntimeCallee(*directCallee)) {
          processStates[procId].sawUnhandledCoverageRuntimeCall = true;
          return callIndirectOp.emitError()
                 << "unhandled coverage runtime call in interpreter: "
                 << *directCallee;
        }
        if (auto wrappedCoverageCallee =
                findDirectCoverageRuntimeCalleeInSymbol(*directCallee)) {
          processStates[procId].sawUnhandledCoverageRuntimeCall = true;
          return callIndirectOp.emitError()
                 << "unhandled coverage runtime call in interpreter: "
                 << *wrappedCoverageCallee;
        }
      }

      // Fallback: try to resolve the virtual method statically by tracing
      // the SSA chain back to the vtable GEP pattern:
      //   calleeValue = unrealized_conversion_cast(llvm.load(llvm.getelementptr(
      //                     vtablePtr, [0, methodIndex])))
      // where vtablePtr = llvm.load(llvm.getelementptr(objPtr, [0, ..., 1]))
      // From the outer GEP's struct type we get the class name, construct
      // "ClassName::__vtable__", and read the function address at methodIndex.
      bool resolved = false;
      do {
        // Step 1: trace calleeValue -> unrealized_conversion_cast input
        auto castOp =
            calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          break;
        Value rawPtr = castOp.getInputs()[0];

        // Step 2: rawPtr should come from llvm.load (loads func ptr from vtable)
        auto funcPtrLoad = rawPtr.getDefiningOp<LLVM::LoadOp>();
        if (!funcPtrLoad)
          break;

        // Step 3: the load address comes from a GEP into the vtable array
        auto vtableGEP = funcPtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!vtableGEP)
          break;

        // Extract the method index from the GEP indices (last index)
        auto vtableIndices = vtableGEP.getIndices();
        if (vtableIndices.size() < 2)
          break;
        auto lastIdx = vtableIndices[vtableIndices.size() - 1];
        int64_t methodIndex = -1;
        if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(lastIdx))
          methodIndex = intAttr.getInt();
        else if (auto dynIdx = llvm::dyn_cast_if_present<Value>(lastIdx)) {
          InterpretedValue dynVal = getValue(procId, dynIdx);
          if (!dynVal.isX())
            methodIndex = static_cast<int64_t>(dynVal.getUInt64());
        }
        if (methodIndex < 0)
          break;

        // Step 4: vtableGEP base = vtable pointer from llvm.load
        auto vtablePtrLoad =
            vtableGEP.getBase().getDefiningOp<LLVM::LoadOp>();
        if (!vtablePtrLoad)
          break;

        // Step 5: the vtable pointer load address comes from GEP on the object
        auto objGEP =
            vtablePtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!objGEP)
          break;

        // Extract the struct type name from the outer GEP's element type
        std::string vtableGlobalName;
        if (auto structTy =
                dyn_cast<LLVM::LLVMStructType>(objGEP.getElemType())) {
          if (structTy.isIdentified()) {
            vtableGlobalName = structTy.getName().str() + "::__vtable__";
          }
        }
        if (vtableGlobalName.empty())
          break;

        // Runtime vtable override: read the actual vtable pointer from the
        // object's memory to find the correct derived-class vtable. The
        // compile-time struct type from the GEP may point to a base class
        // vtable with stubs, while the runtime object is a derived class
        // with real overrides.
        if (!callIndirectOp.getArgOperands().empty()) {
          InterpretedValue selfVal =
              getValue(procId, callIndirectOp.getArgOperands()[0]);
          if (!selfVal.isX()) {
            uint64_t objAddr = selfVal.getUInt64();
            uint64_t runtimeVtableAddr = 0;
            if (readObjectVTableAddress(objAddr, runtimeVtableAddr, procId)) {
              auto globalIt2 = addressToGlobal.find(runtimeVtableAddr);
              if (globalIt2 != addressToGlobal.end()) {
                std::string runtimeVtableName = globalIt2->second;
                if (runtimeVtableName != vtableGlobalName &&
                    globalMemoryBlocks.count(runtimeVtableName) &&
                    isVtableGlobalSymbol(runtimeVtableName)) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  call_indirect: runtime vtable override: "
                             << vtableGlobalName << " -> "
                             << runtimeVtableName << "\n");
                  vtableGlobalName = runtimeVtableName;
                }
              }
            }
          }
        }

        // Step 6: find the vtable global and read the function address
        auto globalIt = globalMemoryBlocks.find(vtableGlobalName);
        if (globalIt == globalMemoryBlocks.end())
          break;

        auto &vtableBlock = globalIt->second;
        unsigned slotOffset = methodIndex * 8;
        if (slotOffset + 8 > vtableBlock.size)
          break;

        // Read 8-byte function address (little-endian) from vtable memory
        uint64_t resolvedFuncAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          resolvedFuncAddr |=
              static_cast<uint64_t>(vtableBlock[slotOffset + i]) << (i * 8);

        if (resolvedFuncAddr == 0)
          break; // Slot is empty (no function registered)

        auto funcIt = addressToFunction.find(resolvedFuncAddr);
        if (funcIt == addressToFunction.end())
          break;

        StringRef resolvedName = funcIt->second;
        noteResolvedTarget(resolvedName);
        // [CI-XFALLBACK] diagnostic removed
        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: fallback vtable resolution: "
                   << vtableGlobalName << "[" << methodIndex << "] -> "
                   << resolvedName << "\n");

        // Look up the function
        auto &state = processStates[procId];
        Operation *parent = state.processOrInitialOp;
        while (parent && !isa<ModuleOp>(parent))
          parent = parent->getParentOp();
        ModuleOp moduleOp = parent ? cast<ModuleOp>(parent) : rootModule;
        auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(resolvedName);
        if (!funcOp)
          break;

        // Gather arguments
        SmallVector<InterpretedValue, 4> args;
        for (Value arg : callIndirectOp.getArgOperands())
          args.push_back(getValue(procId, arg));

        // Keep config_db interception behavior identical across call_indirect
        // dispatch paths, including funcPtrVal.isX() fallback.
        if (tryInterceptConfigDbCallIndirect(procId, callIndirectOp,
                                             resolvedName, args)) {
          resolved = true;
          break;
        }

        // Intercept UVM phase/objection methods in the X-fallback path.
        if ((resolvedName.contains("uvm_phase::raise_objection") ||
             resolvedName.contains("uvm_phase::drop_objection")) &&
            !resolvedName.contains("phase_hopper") &&
            !args.empty() && !args[0].isX()) {
          uint64_t phaseAddr = normalizeUvmObjectKey(procId, args[0].getUInt64());
          if (phaseAddr == 0)
            phaseAddr = args[0].getUInt64();
          InterpretedValue countVal =
              args.size() > 3 ? args[3] : InterpretedValue(llvm::APInt(32, 1));
          int64_t count = countVal.isX() ? 1 : static_cast<int64_t>(countVal.getUInt64());
          auto objIt = phaseObjectionHandles.find(phaseAddr);
          MooreObjectionHandle handle;
          if (objIt != phaseObjectionHandles.end()) {
            handle = objIt->second;
          } else {
            std::string phaseName = "phase_" + std::to_string(phaseAddr);
            handle = __moore_objection_create(
                phaseName.c_str(), static_cast<int64_t>(phaseName.size()));
            phaseObjectionHandles[phaseAddr] = handle;
          }
          int64_t beforeCount = __moore_objection_get_count(handle);
          if (resolvedName.contains("raise_objection")) {
            raisePhaseObjection(handle, count);
          } else {
            dropPhaseObjection(handle, count);
          }
          if (args.size() > 1 && !args[1].isX()) {
            InterpretedValue descVal =
                args.size() > 2 ? args[2] : InterpretedValue(llvm::APInt(128, 0));
            maybeDispatchUvmComponentObjectionCallback(
                procId, args[1].getUInt64(), handle,
                /*isRaise=*/resolvedName.contains("raise_objection"), descVal,
                countVal, callIndirectOp.getOperation());
          }
          int64_t afterCount = __moore_objection_get_count(handle);
          if (beforeCount > 0 || afterCount > 0)
            executePhasePhaseSawPositiveObjection[phaseAddr] = true;
          resolved = true;
          break;
        }

        // Record port connect() in X-fallback path.
        // Do not bypass UVM connect() bookkeeping; allow canonical behavior.
        auto isNativeConnectResolvedName = [&](llvm::StringRef name) {
          if (!name.contains("::connect"))
            return false;
          return name.contains("uvm_port_base") ||
                 name.contains("uvm_analysis_port") ||
                 name.contains("uvm_analysis_export") ||
                 name.contains("uvm_analysis_imp") ||
                 name.contains("uvm_seq_item_pull_") ||
                 (name.contains("uvm_tlm_") &&
                  (name.contains("_port") || name.contains("_export") ||
                   name.contains("_imp")));
        };
        if (isNativeConnectResolvedName(resolvedName) &&
            !resolvedName.contains("connect_phase") && args.size() >= 2) {
          uint64_t rawSelfAddr2 = args[0].isX() ? 0 : args[0].getUInt64();
          uint64_t rawProviderAddr2 = args[1].isX() ? 0 : args[1].getUInt64();
          recordUvmPortConnection(procId, rawSelfAddr2, rawProviderAddr2);
        }

        // Intercept analysis write entrypoints in X-fallback path.
        // The UVM write() body iterates m_imp_list via get_if(i), but
        // m_if is empty because we skip resolve_bindings. Use our native
        // analysisPortConnections map to dispatch to terminal imps.
        if (enableUvmAnalysisNativeInterceptors &&
            isNativeAnalysisWriteCalleeCI(resolvedName) && args.size() >= 2) {
          uint64_t rawPortAddr = args[0].isX() ? 0 : args[0].getUInt64();
          uint64_t portAddr = canonicalizeUvmObjectAddress(procId, rawPortAddr);
          if (traceAnalysisEnabled)
            llvm::errs() << "[ANALYSIS-WRITE-XFALLBACK] " << resolvedName
                         << " portAddr=0x" << llvm::format_hex(portAddr, 0)
                         << " inMap=" << analysisPortConnections.count(portAddr)
                         << "\n";
          // Flatten the connection chain to find all terminal imps.
          llvm::SmallVector<uint64_t, 4> terminals;
          llvm::SmallVector<uint64_t, 8> worklist;
          llvm::DenseSet<uint64_t> visited;
          seedAnalysisPortConnectionWorklist(procId, portAddr, worklist);
          while (!worklist.empty()) {
            uint64_t addr = worklist.pop_back_val();
            if (!visited.insert(addr).second)
              continue;
            auto chainIt = analysisPortConnections.find(addr);
            if (chainIt != analysisPortConnections.end() &&
                !chainIt->second.empty()) {
              for (uint64_t next : chainIt->second)
                worklist.push_back(next);
            } else {
              terminals.push_back(addr);
            }
          }
          if (!terminals.empty()) {
            if (traceAnalysisEnabled)
              llvm::errs() << "[ANALYSIS-WRITE-XFALLBACK] " << terminals.size()
                           << " terminal(s) found\n";
            for (uint64_t impAddr : terminals) {
              auto writeTarget = tryResolveAnalysisWriteTarget(
                  moduleOp, impAddr, "ANALYSIS-WRITE-XFALLBACK");
              if (!writeTarget)
                continue;
              dispatchAnalysisWriteTarget(
                  writeTarget->first, writeTarget->second,
                  "ANALYSIS-WRITE-XFALLBACK", impAddr, args[1]);
            }
            resolved = true;
            break;
          }
          if (resolvedName.contains("uvm_tlm_if_base")) {
            if (tryDispatchAnalysisWriteSelfFallback(
                    moduleOp, "ANALYSIS-WRITE-XFALLBACK", resolvedName,
                    portAddr, args[1])) {
              resolved = true;
              break;
            }
            if (traceAnalysisEnabled)
              llvm::errs() << "[ANALYSIS-WRITE-XFALLBACK] NO terminals for "
                           << "tlm_if_base self=0x"
                           << llvm::format_hex(portAddr, 0) << " (no-op)\n";
            resolved = true;
            break;
          }
          // If no native connections, fall through to normal UVM body dispatch.
        }

        // Dispatch the call
        // [SEQ-XFALLBACK] diagnostic removed
        // Entry-table dispatch: decode tagged FuncId from X-fallback vtable addr.
        // Dispatch both native and trampoline entries via the entry table.
        if (compiledFuncEntries && funcPtrVal.getUInt64() >= 0xF0000000ULL &&
            funcPtrVal.getUInt64() < 0x100000000ULL &&
            processStates[procId].callDepth < 2000) {
          uint32_t fid = static_cast<uint32_t>(funcPtrVal.getUInt64() - 0xF0000000ULL);
          noteAotFuncIdCall(fid);
          if (aotDepth != 0) {
            ++entryTableSkippedDepthCount;
          } else if (fid < numCompiledAllFuncs && compiledFuncEntries[fid]) {
            bool isNativeEntry =
                (fid < compiledFuncIsNative.size() && compiledFuncIsNative[fid]);
            bool hasTrampolineEntry =
                (fid < compiledFuncHasTrampoline.size() &&
                 compiledFuncHasTrampoline[fid]);
            // Deny/trap checks for call_indirect X-fallback path.
            if (isNativeEntry && aotDenyFids.count(fid))
              goto ci_xfallback_interpreted;
            if (isNativeEntry && static_cast<int32_t>(fid) == aotTrapFid) {
              llvm::errs() << "[AOT TRAP] ci-xfallback fid=" << fid;
              if (fid < aotFuncEntryNamesById.size())
                llvm::errs() << " name=" << aotFuncEntryNamesById[fid];
              llvm::errs() << "\n";
              __builtin_trap();
            }
            // Runtime interception policy may mark a FuncId as non-native even
            // when the compiled module still has a direct entry pointer. Only
            // call non-native entries through generated trampolines.
            if (!isNativeEntry && !hasTrampolineEntry)
              goto ci_xfallback_interpreted;
            // Skip native dispatch for yield-capable functions outside process
            // context.
            if (shouldSkipMayYieldEntry(fid, isNativeEntry)) {
              ++entryTableSkippedYieldCount;
              noteAotEntryYieldSkip(fid);
              goto ci_xfallback_interpreted;
            }
            void *entryPtr = const_cast<void *>(compiledFuncEntries[fid]);
            if (entryPtr) {
              unsigned numArgs = funcOp.getNumArguments();
              unsigned numResults = funcOp.getNumResults();
              bool eligible = (numArgs <= 8 && numResults <= 1);
              if (eligible) {
                for (unsigned i = 0; i < numArgs && eligible; ++i) {
                  auto ty = funcOp.getArgumentTypes()[i];
                  if (auto intTy = dyn_cast<mlir::IntegerType>(ty)) {
                    if (intTy.getWidth() > 64) eligible = false;
                  } else if (isa<mlir::IndexType>(ty) ||
                             isa<mlir::LLVM::LLVMPointerType>(ty)) {
                    // OK
                  } else {
                    eligible = false;
                  }
                }
                if (numResults == 1) {
                  auto resTy = funcOp.getResultTypes()[0];
                  if (auto intTy = dyn_cast<mlir::IntegerType>(resTy)) {
                    if (intTy.getWidth() > 64) eligible = false;
                  } else if (!isa<mlir::IndexType>(resTy) &&
                             !isa<mlir::LLVM::LLVMPointerType>(resTy)) {
                    eligible = false;
                  }
                }
              }
          if (eligible) {
            uint64_t a[8] = {};
            bool normalizePointerArgs = isNativeEntry;
                if (shouldForceInterpretedFragileUvmCallee(resolvedName))
                  goto ci_xfallback_interpreted;
                bool forcePredicateFalse = false;
                eligible = fillNativeCallArgs(args, funcOp.getArgumentTypes(),
                                              resolvedName, numArgs, a,
                                              normalizePointerArgs,
                                              forcePredicateFalse);
                if (forcePredicateFalse) {
                  setCallIndirectResults({});
                  resolved = true;
                  break;
                }
                if (!eligible)
                  goto ci_xfallback_interpreted;
                maybeTraceIndirectNative(fid, resolvedName, isNativeEntry,
                                         numArgs,
                                         numResults, a);

                if (eligible) {

                // Set TLS context so Moore runtime helpers can normalize ptrs.
                void *prevTls = __circt_sim_get_tls_ctx();
                __circt_sim_set_tls_ctx(static_cast<void *>(this));
                __circt_sim_set_tls_normalize(LLHDProcessInterpreter::normalizeVirtualPtr);
                uint64_t result = 0;
                using F0 = uint64_t (*)();
                using F1 = uint64_t (*)(uint64_t);
                using F2 = uint64_t (*)(uint64_t, uint64_t);
                using F3 = uint64_t (*)(uint64_t, uint64_t, uint64_t);
                using F4 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);
                using F5 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t);
                using F6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t);
                using F7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t, uint64_t);
                using F8 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t, uint64_t, uint64_t);
                switch (numArgs) {
                case 0: result = reinterpret_cast<F0>(entryPtr)(); break;
                case 1: result = reinterpret_cast<F1>(entryPtr)(a[0]); break;
                case 2: result = reinterpret_cast<F2>(entryPtr)(a[0], a[1]); break;
                case 3: result = reinterpret_cast<F3>(entryPtr)(a[0], a[1], a[2]); break;
                case 4: result = reinterpret_cast<F4>(entryPtr)(a[0], a[1], a[2], a[3]); break;
                case 5: result = reinterpret_cast<F5>(entryPtr)(a[0], a[1], a[2], a[3], a[4]); break;
                case 6: result = reinterpret_cast<F6>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5]); break;
                case 7: result = reinterpret_cast<F7>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
                case 8: result = reinterpret_cast<F8>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
                }
                __circt_sim_set_tls_ctx(prevTls);
                if (numResults == 1) {
                  SmallVector<InterpretedValue, 2> nativeResults;
                  auto resTy = funcOp.getResultTypes()[0];
                  unsigned bits = 64;
                  if (auto intTy = dyn_cast<mlir::IntegerType>(resTy))
                    bits = intTy.getWidth();
                  nativeResults.push_back(InterpretedValue(
                      llvm::APInt(64, result).zextOrTrunc(bits)));
                  setCallIndirectResults(nativeResults);
                }
                if (isNativeEntry)
                  ++nativeEntryCallCount;
                else
                  ++trampolineEntryCallCount;
                resolved = true;
                break;
                } // eligible (no fake addr)
              }
            }
          }
        }
      ci_xfallback_interpreted:
        auto &callState = processStates[procId];
        ++interpretedCallCounts[funcOp.getOperation()];
        ++callState.callDepth;
        SmallVector<InterpretedValue, 4> results;
        auto callResult = interpretFuncBody(procId, funcOp, args, results,
                                            callIndirectOp);
        --callState.callDepth;

        if (failed(callResult)) {
          if (callState.waiting)
            return success();
          if (shouldPropagateCoverageRuntimeFailure(procId))
            return failure();
          break;
        }

        // Set return values.
        setCallIndirectResults(results);

        resolved = true;
      } while (false);

      if (!resolved) {
        emitVtableWarning("function pointer is X (uninitialized)");
        // Return zero/null instead of X to prevent cascading X-propagation
        // in UVM code paths that check return values for null.
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
      }
      return success();
    }

    // Look up the function name from the vtable
    uint64_t funcAddr = funcPtrVal.getUInt64();

    ciCacheFuncAddr = funcAddr;

    // Check resolution cache first.
    auto rcIt = callIndirectResolutionCache.find(funcAddr);
    if (rcIt != callIndirectResolutionCache.end()) {
      ++ciResolutionCacheHits;
    } else {
      ++ciResolutionCacheMisses;
    }

    auto it = addressToFunction.find(funcAddr);
    if (it == addressToFunction.end()) {
      // Runtime vtable pointer is corrupt or unmapped. Try static resolution
      // by tracing the SSA chain back to the vtable global, which is always
      // correct regardless of runtime memory corruption.
      bool staticResolved = false;
      do {
        auto castOp =
            calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          break;
        Value rawPtr = castOp.getInputs()[0];
        auto funcPtrLoad = rawPtr.getDefiningOp<LLVM::LoadOp>();
        if (!funcPtrLoad)
          break;
        auto vtableGEP =
            funcPtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!vtableGEP)
          break;
        auto vtableIndices = vtableGEP.getIndices();
        if (vtableIndices.size() < 2)
          break;
        auto lastIdx = vtableIndices[vtableIndices.size() - 1];
        int64_t methodIndex = -1;
        if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(lastIdx))
          methodIndex = intAttr.getInt();
        else if (auto dynIdx = llvm::dyn_cast_if_present<Value>(lastIdx)) {
          InterpretedValue dynVal = getValue(procId, dynIdx);
          if (!dynVal.isX())
            methodIndex = static_cast<int64_t>(dynVal.getUInt64());
        }
        if (methodIndex < 0)
          break;

        // Alternative: vtable GEP base is llvm.mlir.addressof (direct vtable
        // reference, not loaded from object). Use circt.vtable_entries metadata
        // to resolve the function name without reading vtable memory.
        if (auto addrOfOp =
                vtableGEP.getBase().getDefiningOp<LLVM::AddressOfOp>()) {
          Operation *foundSymbol = mlir::SymbolTable::lookupNearestSymbolFrom(
              callIndirectOp.getOperation(), addrOfOp.getGlobalNameAttr());
          auto vtableGlobal = dyn_cast_or_null<LLVM::GlobalOp>(foundSymbol);
          if (vtableGlobal) {
            auto vtableEntriesAttr = dyn_cast_or_null<ArrayAttr>(
                vtableGlobal->getAttr("circt.vtable_entries"));
            if (vtableEntriesAttr) {
              for (Attribute entryAttr : vtableEntriesAttr) {
                auto entryArray = dyn_cast<ArrayAttr>(entryAttr);
                if (!entryArray || entryArray.size() < 2)
                  continue;
                auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
                auto symbolAttr = dyn_cast<FlatSymbolRefAttr>(entryArray[1]);
                if (!indexAttr || !symbolAttr)
                  continue;
                if (indexAttr.getInt() == methodIndex) {
                  StringRef resolvedName = symbolAttr.getValue();
                  noteResolvedTarget(resolvedName);
                  // Use a unique synthetic address per vtable+slot to avoid
                  // collisions in the addressToFunction map.
                  uint64_t syntheticAddr =
                      llvm::hash_combine(addrOfOp.getGlobalName(),
                                         methodIndex) |
                      0x8000000000000000ULL;
                  addressToFunction[syntheticAddr] = resolvedName;
                  it = addressToFunction.find(syntheticAddr);
                  staticResolved = true;
                  break;
                }
              }
            }
          }
          if (staticResolved)
            break;
          break;
        }

        auto vtablePtrLoad =
            vtableGEP.getBase().getDefiningOp<LLVM::LoadOp>();
        if (!vtablePtrLoad)
          break;
        auto objGEP =
            vtablePtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!objGEP)
          break;
        std::string vtableGlobalName;
        if (auto structTy =
                dyn_cast<LLVM::LLVMStructType>(objGEP.getElemType())) {
          if (structTy.isIdentified())
            vtableGlobalName = structTy.getName().str() + "::__vtable__";
        }
        if (vtableGlobalName.empty())
          break;

        // Runtime vtable override (same as X-fallback path above)
        if (!callIndirectOp.getArgOperands().empty()) {
          InterpretedValue selfVal2 =
              getValue(procId, callIndirectOp.getArgOperands()[0]);
          if (!selfVal2.isX()) {
            uint64_t objAddr2 = selfVal2.getUInt64();
            uint64_t runtimeVtableAddr2 = 0;
            if (readObjectVTableAddress(objAddr2, runtimeVtableAddr2, procId)) {
              auto globalIt3 = addressToGlobal.find(runtimeVtableAddr2);
              if (globalIt3 != addressToGlobal.end()) {
                std::string runtimeVtableName2 = globalIt3->second;
                if (runtimeVtableName2 != vtableGlobalName &&
                    globalMemoryBlocks.count(runtimeVtableName2) &&
                    isVtableGlobalSymbol(runtimeVtableName2)) {
                  vtableGlobalName = runtimeVtableName2;
                }
              }
            }
          }
        }

        auto globalIt = globalMemoryBlocks.find(vtableGlobalName);
        if (globalIt == globalMemoryBlocks.end())
          break;
        auto &vtableBlock = globalIt->second;
        unsigned slotOffset = methodIndex * 8;
        if (slotOffset + 8 > vtableBlock.size)
          break;
        uint64_t resolvedFuncAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          resolvedFuncAddr |=
              static_cast<uint64_t>(vtableBlock[slotOffset + i])
              << (i * 8);
        if (resolvedFuncAddr == 0)
          break;
        auto funcIt2 = addressToFunction.find(resolvedFuncAddr);
        if (funcIt2 == addressToFunction.end())
          break;
        StringRef resolvedName = funcIt2->second;
        noteResolvedTarget(resolvedName);

        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: static fallback: "
                   << vtableGlobalName << "[" << methodIndex << "] -> "
                   << resolvedName << "\n");
        if (traceConfigDbEnabled && procId == 1) {
          llvm::errs() << "[CFG-CI-STATIC-PROC1] "
                       << vtableGlobalName << "[" << methodIndex << "] -> "
                       << resolvedName << "\n";
        }
        auto &st2 = processStates[procId];
        Operation *par = st2.processOrInitialOp;
        while (par && !isa<ModuleOp>(par))
          par = par->getParentOp();
        ModuleOp modOp = par ? cast<ModuleOp>(par) : rootModule;
        auto fOp = modOp.lookupSymbol<func::FuncOp>(resolvedName);
        if (!fOp)
          break;
        SmallVector<InterpretedValue, 4> sArgs;
        for (Value arg : callIndirectOp.getArgOperands())
          sArgs.push_back(getValue(procId, arg));

        // Keep config_db interception behavior identical across call_indirect
        // dispatch paths, including static fallback resolution.
        if (tryInterceptConfigDbCallIndirect(procId, callIndirectOp,
                                             resolvedName, sArgs)) {
          staticResolved = true;
          break;
        }

        // Intercept UVM phase/objection methods in non-X static fallback.
        if ((resolvedName.contains("uvm_phase::raise_objection") ||
             resolvedName.contains("uvm_phase::drop_objection")) &&
            !resolvedName.contains("phase_hopper") &&
            !sArgs.empty() && !sArgs[0].isX()) {
          uint64_t phaseAddr = normalizeUvmObjectKey(procId, sArgs[0].getUInt64());
          if (phaseAddr == 0)
            phaseAddr = sArgs[0].getUInt64();
          InterpretedValue countVal =
              sArgs.size() > 3 ? sArgs[3] : InterpretedValue(llvm::APInt(32, 1));
          int64_t cnt = countVal.isX() ? 1 : static_cast<int64_t>(countVal.getUInt64());
          auto objIt = phaseObjectionHandles.find(phaseAddr);
          MooreObjectionHandle handle;
          if (objIt != phaseObjectionHandles.end()) {
            handle = objIt->second;
          } else {
            std::string phaseName = "phase_" + std::to_string(phaseAddr);
            handle = __moore_objection_create(
                phaseName.c_str(), static_cast<int64_t>(phaseName.size()));
            phaseObjectionHandles[phaseAddr] = handle;
          }
          int64_t beforeCount = __moore_objection_get_count(handle);
          if (resolvedName.contains("raise_objection")) {
            raisePhaseObjection(handle, cnt);
          } else {
            dropPhaseObjection(handle, cnt);
          }
          if (sArgs.size() > 1 && !sArgs[1].isX()) {
            InterpretedValue descVal =
                sArgs.size() > 2 ? sArgs[2] : InterpretedValue(llvm::APInt(128, 0));
            maybeDispatchUvmComponentObjectionCallback(
                procId, sArgs[1].getUInt64(), handle,
                /*isRaise=*/resolvedName.contains("raise_objection"), descVal,
                countVal, callIndirectOp.getOperation());
          }
          int64_t afterCount = __moore_objection_get_count(handle);
          if (beforeCount > 0 || afterCount > 0)
            executePhasePhaseSawPositiveObjection[phaseAddr] = true;
          staticResolved = true;
          break;
        }

        // Record port connect() in static fallback path.
        // Do not bypass UVM connect() bookkeeping; allow canonical behavior.
        auto isNativeConnectResolvedName = [&](llvm::StringRef name) {
          if (!name.contains("::connect"))
            return false;
          return name.contains("uvm_port_base") ||
                 name.contains("uvm_analysis_port") ||
                 name.contains("uvm_analysis_export") ||
                 name.contains("uvm_analysis_imp") ||
                 name.contains("uvm_seq_item_pull_") ||
                 (name.contains("uvm_tlm_") &&
                  (name.contains("_port") || name.contains("_export") ||
                   name.contains("_imp")));
        };
        if (isNativeConnectResolvedName(resolvedName) &&
            !resolvedName.contains("connect_phase") && sArgs.size() >= 2) {
          uint64_t rawSelfAddr3 = sArgs[0].isX() ? 0 : sArgs[0].getUInt64();
          uint64_t rawProviderAddr3 = sArgs[1].isX() ? 0 : sArgs[1].getUInt64();
          recordUvmPortConnection(procId, rawSelfAddr3, rawProviderAddr3);
        }

        // Intercept resource_db in static fallback path.
        if (resolvedName.contains("resource_db") &&
            resolvedName.contains("implementation") &&
            (resolvedName.contains("::set") ||
             resolvedName.contains("::read_by_name"))) {

          auto readStr2 = [&](unsigned argIdx) -> std::string {
            if (argIdx >= sArgs.size())
              return "";
            return readMooreStringStruct(procId, sArgs[argIdx]);
          };

          if (resolvedName.contains("::set") &&
              !resolvedName.contains("set_default") &&
              !resolvedName.contains("set_override") &&
              !resolvedName.contains("set_anonymous")) {
            if (sArgs.size() >= 4) {
              std::string scope = readStr2(1);
              std::string fieldName = readStr2(2);
              std::string key = scope + "." + fieldName;
              InterpretedValue &valueArg = sArgs[3];
              unsigned valueBits = valueArg.getWidth();
              bool truncatedValue = false;
              std::vector<uint8_t> valueData =
                  serializeInterpretedValueBytes(valueArg, /*maxBytes=*/1ULL << 20,
                                                 &truncatedValue);
              unsigned valueBytes = static_cast<unsigned>(valueData.size());
              configDbEntries[key] = std::move(valueData);
              if (traceConfigDbEnabled && truncatedValue) {
                llvm::errs() << "[RSRC-CI-STATIC-SET] truncated oversized value payload"
                             << " key=\"" << key << "\" bitWidth=" << valueBits
                             << "\n";
              }
            }
            staticResolved = true;
            break;
          }

          if (resolvedName.contains("::read_by_name") &&
              sArgs.size() >= 4 &&
              callIndirectOp.getNumResults() >= 1) {
            std::string scope = readStr2(1);
            std::string fieldName = readStr2(2);
            std::string key = scope + "." + fieldName;

            auto entryIt = configDbEntries.find(key);
            if (entryIt == configDbEntries.end()) {
              for (auto &[k, v] : configDbEntries) {
                size_t dotPos = k.rfind('.');
                if (dotPos != std::string::npos &&
                    k.substr(dotPos + 1) == fieldName) {
                  entryIt = configDbEntries.find(k);
                  break;
                }
              }
            }

            if (entryIt != configDbEntries.end()) {
              Value outputRef = callIndirectOp.getArgOperands()[3];
              const std::vector<uint8_t> &valueData = entryIt->second;
              Type refType = outputRef.getType();

              if (auto refT = dyn_cast<llhd::RefType>(refType)) {
                Type innerType = refT.getNestedType();
                unsigned innerBits = getTypeWidth(innerType);
                unsigned innerBytes = (innerBits + 7) / 8;
                llvm::APInt valueBits2(innerBits, 0);
                for (unsigned i = 0;
                     i < std::min(innerBytes, (unsigned)valueData.size()); ++i)
                  safeInsertBits(valueBits2,llvm::APInt(8, valueData[i]), i * 8);
                SignalId sigId3 = resolveSignalId(outputRef);
                if (sigId3 != 0)
                  pendingEpsilonDrives[sigId3] = InterpretedValue(valueBits2);
                // Also write to memory (same as direct-resolution path).
                InterpretedValue refAddr3 = getValue(procId, outputRef);
                if (!refAddr3.isX()) {
                  uint64_t addr3 = refAddr3.getUInt64();
                  uint64_t off4 = 0;
                  MemoryBlock *blk3 =
                      findMemoryBlockByAddress(addr3, procId, &off4);
                  if (!blk3)
                    blk3 = findBlockByAddress(addr3, off4);
                  if (blk3) {
                    writeConfigDbBytesToMemoryBlock(
                        blk3, off4, valueData, innerBytes,
                        /*zeroFillMissing=*/true);
                  } else {
                    uint64_t nativeOff = 0;
                    size_t nativeSize = 0;
                    if (findNativeMemoryBlockByAddress(addr3, &nativeOff,
                                                       &nativeSize)) {
                      writeConfigDbBytesToNativeMemory(
                          addr3, nativeOff, nativeSize, valueData, innerBytes,
                          /*zeroFillMissing=*/true);
                    }
                  }
                }
              } else if (isa<LLVM::LLVMPointerType>(refType)) {
                if (!sArgs[3].isX()) {
                  uint64_t outputAddr = sArgs[3].getUInt64();
                  uint64_t outOff2 = 0;
                  MemoryBlock *outBlock =
                      findMemoryBlockByAddress(outputAddr, procId, &outOff2);
                  if (!outBlock)
                    outBlock = findBlockByAddress(outputAddr, outOff2);
                  if (outBlock) {
                    writeConfigDbBytesToMemoryBlock(
                        outBlock, outOff2, valueData,
                        static_cast<unsigned>(valueData.size()),
                        /*zeroFillMissing=*/false);
                  } else {
                    uint64_t nativeOff = 0;
                    size_t nativeSize = 0;
                    if (findNativeMemoryBlockByAddress(outputAddr, &nativeOff,
                                                       &nativeSize)) {
                      writeConfigDbBytesToNativeMemory(
                          outputAddr, nativeOff, nativeSize, valueData,
                          static_cast<unsigned>(valueData.size()),
                          /*zeroFillMissing=*/false);
                    }
                  }
                }
              }

              setValue(procId, callIndirectOp.getResult(0),
                      InterpretedValue(llvm::APInt(1, 1)));
              staticResolved = true;
              break;
            }

            setValue(procId, callIndirectOp.getResult(0),
                    InterpretedValue(llvm::APInt(1, 0)));
            staticResolved = true;
            break;
          }
        }

        // Intercept analysis write entrypoints in non-X static fallback path.
        if (enableUvmAnalysisNativeInterceptors &&
            isNativeAnalysisWriteCalleeCI(resolvedName) && sArgs.size() >= 2) {
          uint64_t rawPortAddr3 = sArgs[0].isX() ? 0 : sArgs[0].getUInt64();
          uint64_t portAddr3 =
              canonicalizeUvmObjectAddress(procId, rawPortAddr3);
          if (traceAnalysisEnabled)
            llvm::errs() << "[ANALYSIS-WRITE-STATIC] " << resolvedName
                         << " portAddr=0x" << llvm::format_hex(portAddr3, 0)
                         << " inMap=" << analysisPortConnections.count(portAddr3)
                         << "\n";
          llvm::SmallVector<uint64_t, 4> terminals3;
          llvm::SmallVector<uint64_t, 8> worklist3;
          llvm::DenseSet<uint64_t> visited3;
          seedAnalysisPortConnectionWorklist(procId, portAddr3, worklist3);
          while (!worklist3.empty()) {
            uint64_t addr = worklist3.pop_back_val();
            if (!visited3.insert(addr).second)
              continue;
            auto chainIt = analysisPortConnections.find(addr);
            if (chainIt != analysisPortConnections.end() &&
                !chainIt->second.empty()) {
              for (uint64_t next : chainIt->second)
                worklist3.push_back(next);
            } else {
              terminals3.push_back(addr);
            }
          }
          if (!terminals3.empty()) {
            if (traceAnalysisEnabled)
              llvm::errs() << "[ANALYSIS-WRITE-STATIC] " << terminals3.size()
                           << " terminal(s) found\n";
            for (uint64_t impAddr : terminals3) {
              auto writeTarget = tryResolveAnalysisWriteTarget(
                  modOp, impAddr, "ANALYSIS-WRITE-STATIC");
              if (!writeTarget)
                continue;
              dispatchAnalysisWriteTarget(
                  writeTarget->first, writeTarget->second,
                  "ANALYSIS-WRITE-STATIC", impAddr, sArgs[1]);
            }
            staticResolved = true;
            break;
          }
          if (resolvedName.contains("uvm_tlm_if_base")) {
            if (tryDispatchAnalysisWriteSelfFallback(
                    modOp, "ANALYSIS-WRITE-STATIC", resolvedName, portAddr3,
                    sArgs[1])) {
              staticResolved = true;
              break;
            }
            if (traceAnalysisEnabled)
              llvm::errs() << "[ANALYSIS-WRITE-STATIC] NO terminals for "
                           << "tlm_if_base self=0x"
                           << llvm::format_hex(portAddr3, 0) << " (no-op)\n";
            staticResolved = true;
            break;
          }
          // If no native connections, fall through to normal UVM body dispatch.
        }

        // [SEQ-UNMAPPED] diagnostic removed
        // Entry-table dispatch for static fallback path.
        // Dispatch both native and trampoline entries via the entry table.
        if (compiledFuncEntries && funcAddr >= 0xF0000000ULL &&
            funcAddr < 0x100000000ULL &&
            processStates[procId].callDepth < 2000) {
          uint32_t fid = static_cast<uint32_t>(funcAddr - 0xF0000000ULL);
          noteAotFuncIdCall(fid);
        if (aotDepth != 0) {
          ++entryTableSkippedDepthCount;
        } else if (fid < numCompiledAllFuncs && compiledFuncEntries[fid]) {
            bool isNativeEntry =
                (fid < compiledFuncIsNative.size() && compiledFuncIsNative[fid]);
            bool hasTrampolineEntry =
                (fid < compiledFuncHasTrampoline.size() &&
                 compiledFuncHasTrampoline[fid]);
            // Deny/trap checks for call_indirect static fallback path.
            if (isNativeEntry && aotDenyFids.count(fid))
              goto ci_static_interpreted;
            if (isNativeEntry && static_cast<int32_t>(fid) == aotTrapFid) {
              llvm::errs() << "[AOT TRAP] ci-static fid=" << fid;
              if (fid < aotFuncEntryNamesById.size())
                llvm::errs() << " name=" << aotFuncEntryNamesById[fid];
              llvm::errs() << "\n";
              __builtin_trap();
            }
            // Runtime interception policy may mark a FuncId as non-native even
            // when the compiled module still has a direct entry pointer. Only
            // call non-native entries through generated trampolines.
            if (!isNativeEntry && !hasTrampolineEntry)
              goto ci_static_interpreted;
            // Keep MAY_YIELD entries on interpreted dispatch.
            if (shouldSkipMayYieldEntry(fid, isNativeEntry)) {
              ++entryTableSkippedYieldCount;
              noteAotEntryYieldSkip(fid);
              goto ci_static_interpreted;
            }
            void *entryPtr = const_cast<void *>(compiledFuncEntries[fid]);
            if (entryPtr) {
              unsigned numArgs = fOp.getNumArguments();
              unsigned numResults = fOp.getNumResults();
              bool eligible = (numArgs <= 8 && numResults <= 1);
              if (eligible) {
                for (unsigned i = 0; i < numArgs && eligible; ++i) {
                  auto ty = fOp.getArgumentTypes()[i];
                  if (auto intTy = dyn_cast<mlir::IntegerType>(ty)) {
                    if (intTy.getWidth() > 64) eligible = false;
                  } else if (isa<mlir::IndexType>(ty) ||
                             isa<mlir::LLVM::LLVMPointerType>(ty)) {
                    // OK
                  } else {
                    eligible = false;
                  }
                }
                if (numResults == 1) {
                  auto resTy = fOp.getResultTypes()[0];
                  if (auto intTy = dyn_cast<mlir::IntegerType>(resTy)) {
                    if (intTy.getWidth() > 64) eligible = false;
                  } else if (!isa<mlir::IndexType>(resTy) &&
                             !isa<mlir::LLVM::LLVMPointerType>(resTy)) {
                    eligible = false;
                  }
                }
              }
          if (eligible) {
            uint64_t a[8] = {};
            bool normalizePointerArgs = isNativeEntry;
                if (shouldForceInterpretedFragileUvmCallee(resolvedName))
                  goto ci_static_interpreted;
                bool forcePredicateFalse = false;
                eligible = fillNativeCallArgs(sArgs, fOp.getArgumentTypes(),
                                              resolvedName, numArgs, a,
                                              normalizePointerArgs,
                                              forcePredicateFalse);
                if (forcePredicateFalse) {
                  setCallIndirectResults({});
                  staticResolved = true;
                  break;
                }
                if (!eligible)
                  goto ci_static_interpreted;
                maybeTraceIndirectNative(fid, resolvedName, isNativeEntry,
                                         numArgs,
                                         numResults, a);

                if (eligible) {

                // Set TLS context so Moore runtime helpers can normalize ptrs.
                void *prevTls = __circt_sim_get_tls_ctx();
                __circt_sim_set_tls_ctx(static_cast<void *>(this));
                __circt_sim_set_tls_normalize(LLHDProcessInterpreter::normalizeVirtualPtr);
                uint64_t result = 0;
                using F0 = uint64_t (*)();
                using F1 = uint64_t (*)(uint64_t);
                using F2 = uint64_t (*)(uint64_t, uint64_t);
                using F3 = uint64_t (*)(uint64_t, uint64_t, uint64_t);
                using F4 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);
                using F5 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t);
                using F6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t);
                using F7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t, uint64_t);
                using F8 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t, uint64_t, uint64_t);
                switch (numArgs) {
                case 0: result = reinterpret_cast<F0>(entryPtr)(); break;
                case 1: result = reinterpret_cast<F1>(entryPtr)(a[0]); break;
                case 2: result = reinterpret_cast<F2>(entryPtr)(a[0], a[1]); break;
                case 3: result = reinterpret_cast<F3>(entryPtr)(a[0], a[1], a[2]); break;
                case 4: result = reinterpret_cast<F4>(entryPtr)(a[0], a[1], a[2], a[3]); break;
                case 5: result = reinterpret_cast<F5>(entryPtr)(a[0], a[1], a[2], a[3], a[4]); break;
                case 6: result = reinterpret_cast<F6>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5]); break;
                case 7: result = reinterpret_cast<F7>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
                case 8: result = reinterpret_cast<F8>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
                }
                __circt_sim_set_tls_ctx(prevTls);
                if (numResults == 1) {
                  SmallVector<InterpretedValue, 2> nativeResults;
                  auto resTy = fOp.getResultTypes()[0];
                  unsigned bits = 64;
                  if (auto intTy = dyn_cast<mlir::IntegerType>(resTy))
                    bits = intTy.getWidth();
                  nativeResults.push_back(InterpretedValue(
                      llvm::APInt(64, result).zextOrTrunc(bits)));
                  setCallIndirectResults(nativeResults);
                }
                if (isNativeEntry)
                  ++nativeEntryCallCount;
                else
                  ++trampolineEntryCallCount;
                staticResolved = true;
                break;
                } // eligible (no fake addr)
              }
            }
          }
        }
      ci_static_interpreted:
        auto &cs2 = processStates[procId];
        ++interpretedCallCounts[fOp.getOperation()];
        ++cs2.callDepth;
        SmallVector<InterpretedValue, 4> sResults;
        auto callRes = interpretFuncBody(procId, fOp, sArgs, sResults,
                                         callIndirectOp);
        --cs2.callDepth;
        if (failed(callRes)) {
          if (cs2.waiting)
            return success();
          if (shouldPropagateCoverageRuntimeFailure(procId))
            return failure();
          break;
        }
        setCallIndirectResults(sResults);
        staticResolved = true;
      } while (false);
      if (!staticResolved) {
        auto isMooreStringStructType = [](Type ty) -> bool {
          auto structTy = dyn_cast<LLVM::LLVMStructType>(ty);
          if (!structTy || structTy.isOpaque() || structTy.getBody().size() != 2)
            return false;
          Type t0 = structTy.getBody()[0];
          Type t1 = structTy.getBody()[1];
          return isa<LLVM::LLVMPointerType>(t0) &&
                 t1.isSignlessInteger(64);
        };

        auto looksLikeConfigDbSetSig = [&]() -> bool {
          if (callIndirectOp.getNumResults() != 0)
            return false;
          if (callIndirectOp.getNumOperands() < 5)
            return false;
          auto argOps = callIndirectOp.getArgOperands();
          return isa<LLVM::LLVMPointerType>(argOps[0].getType()) &&
                 isMooreStringStructType(argOps[1].getType()) &&
                 isMooreStringStructType(argOps[2].getType()) &&
                 isMooreStringStructType(argOps[3].getType());
        };
        auto looksLikeConfigDbGetSig = [&]() -> bool {
          if (callIndirectOp.getNumResults() != 1)
            return false;
          if (!callIndirectOp.getResult(0).getType().isSignlessInteger(1))
            return false;
          if (callIndirectOp.getNumOperands() < 5)
            return false;
          auto argOps = callIndirectOp.getArgOperands();
          return isa<LLVM::LLVMPointerType>(argOps[0].getType()) &&
                 isa<LLVM::LLVMPointerType>(argOps[1].getType()) &&
                 isMooreStringStructType(argOps[2].getType()) &&
                 isMooreStringStructType(argOps[3].getType());
        };

        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: address 0x"
                   << llvm::format_hex(funcAddr, 16)
                   << " not in vtable map\n");
        auto directCallee = tryGetDirectAddressOfCalleeName();
        if (directCallee &&
            llvm::StringRef(*directCallee).contains("config_db") &&
            llvm::StringRef(*directCallee).contains("implementation")) {
          return callIndirectOp.emitError()
                 << "CIRCTSIM-CFGDB-UNRESOLVED-DISPATCH: unresolved config_db "
                    "call_indirect target (direct callee: "
                 << *directCallee << ")";
        }
        if (looksLikeConfigDbSetSig() || looksLikeConfigDbGetSig()) {
          return callIndirectOp.emitError()
                 << "CIRCTSIM-CFGDB-UNRESOLVED-DISPATCH: unresolved config_db "
                    "call_indirect target (signature match)";
        }
        if (directCallee) {
          if (isCoverageRuntimeCallee(*directCallee)) {
            processStates[procId].sawUnhandledCoverageRuntimeCall = true;
            return callIndirectOp.emitError()
                   << "unhandled coverage runtime call in interpreter: "
                   << *directCallee;
          }
          if (auto wrappedCoverageCallee =
                  findDirectCoverageRuntimeCalleeInSymbol(*directCallee)) {
            processStates[procId].sawUnhandledCoverageRuntimeCall = true;
            return callIndirectOp.emitError()
                   << "unhandled coverage runtime call in interpreter: "
                   << *wrappedCoverageCallee;
          }
        }
        std::string reason = "address " +
            llvm::utohexstr(funcAddr) + " not found in vtable map";
        emitVtableWarning(reason);
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
      }
      return success();
    }

    StringRef calleeName = it->second;
    std::string overriddenCalleeName;
    if (!nativeFactoryOverridesConfigured &&
        isUvmFactoryOverrideSetter(calleeName))
      nativeFactoryOverridesConfigured = true;
    if (!nativeFactoryOverridesConfigured &&
        (calleeName.starts_with("set_type_override_") ||
         calleeName.starts_with("set_inst_override_")))
      nativeFactoryOverridesConfigured = true;
    if (!nativeFactoryInstanceOverridesConfigured &&
        isUvmFactoryInstanceOverrideSetter(calleeName))
      nativeFactoryInstanceOverridesConfigured = true;
    if (!nativeFactoryInstanceOverridesConfigured &&
        calleeName.starts_with("set_inst_override_"))
      nativeFactoryInstanceOverridesConfigured = true;

    // Intercept low-level sequencer handshake immediately after target
    // resolution, before any call-site caches or fast-dispatch paths.
    if ((calleeName.contains("uvm_sequencer") ||
         calleeName.contains("sqr_if_base")) &&
        (calleeName.ends_with("::wait_for_grant") ||
         calleeName.ends_with("::send_request") ||
         calleeName.ends_with("::wait_for_item_done"))) {
      SmallVector<InterpretedValue, 4> args;
      for (Value arg : callIndirectOp.getArgOperands())
        args.push_back(getValue(procId, arg));

      if (traceSeqEnabled) {
        uint64_t a0 = args.size() > 0 && !args[0].isX() ? args[0].getUInt64() : 0;
        uint64_t a1 = args.size() > 1 && !args[1].isX() ? args[1].getUInt64() : 0;
        llvm::errs() << "[SEQ-CI] " << calleeName << " a0=0x"
                     << llvm::format_hex(a0, 16) << " a1=0x"
                     << llvm::format_hex(a1, 16)
                     << " fifo_maps=" << sequencerItemFifo.size() << "\n";
      }

      if (calleeName.ends_with("::wait_for_grant")) {
        if (!args.empty() && !args[0].isX()) {
          uint64_t sqrAddr =
              normalizeUvmSequencerAddress(procId, args[0].getUInt64());
          if (sqrAddr != 0)
            itemToSequencer[sequencerProcKey(procId)] = sqrAddr;
        }
        return success();
      }

      if (calleeName.ends_with("::send_request") && args.size() >= 3) {
        uint64_t sqrAddr = args[0].isX()
                               ? 0
                               : normalizeUvmSequencerAddress(
                                     procId, args[0].getUInt64());
        uint64_t seqAddr = args[1].isX()
                               ? 0
                               : normalizeUvmObjectKey(procId,
                                                       args[1].getUInt64());
        uint64_t itemAddr = args[2].isX() ? 0 : args[2].getUInt64();
        uint64_t queueAddr = 0;
        if (itemAddr != 0) {
          if (auto ownerIt = itemToSequencer.find(itemAddr);
              ownerIt != itemToSequencer.end())
            queueAddr = ownerIt->second;
        }
        if (queueAddr == 0) {
          if (auto procIt = itemToSequencer.find(sequencerProcKey(procId));
              procIt != itemToSequencer.end())
            queueAddr = procIt->second;
        }
        if (queueAddr == 0)
          queueAddr = sqrAddr;
        queueAddr = normalizeUvmSequencerAddress(procId, queueAddr);
        if (itemAddr != 0 && seqAddr != 0)
          recordUvmSequenceItemOwner(itemAddr, seqAddr);
        if (itemAddr != 0 && queueAddr != 0) {
          sequencerItemFifo[queueAddr].push_back(itemAddr);
          recordUvmSequencerItemOwner(itemAddr, queueAddr);
          sequencePendingItemsByProc[procId].push_back(itemAddr);
          if (seqAddr != 0)
            sequencePendingItemsBySeq[seqAddr].push_back(itemAddr);
          wakeUvmSequencerGetWaiterForPush(queueAddr);
          if (traceSeqEnabled) {
            llvm::errs() << "[SEQ-CI] send_request item=0x"
                         << llvm::format_hex(itemAddr, 16) << " sqr=0x"
                         << llvm::format_hex(queueAddr, 16) << " depth="
                         << sequencerItemFifo[queueAddr].size() << "\n";
          }
        }
        return success();
      }

      if (calleeName.ends_with("::wait_for_item_done")) {
        uint64_t seqAddr = args.size() > 1 && !args[1].isX()
                               ? normalizeUvmObjectKey(procId,
                                                       args[1].getUInt64())
                               : 0;

        uint64_t itemAddr = 0;
        auto seqIt = sequencePendingItemsBySeq.end();
        if (seqAddr != 0) {
          seqIt = sequencePendingItemsBySeq.find(seqAddr);
          if (seqIt != sequencePendingItemsBySeq.end() &&
              !seqIt->second.empty())
            itemAddr = seqIt->second.front();
        }
        auto procIt = sequencePendingItemsByProc.find(procId);
        if (itemAddr == 0 && procIt != sequencePendingItemsByProc.end() &&
            !procIt->second.empty())
          itemAddr = procIt->second.front();
        if (itemAddr == 0) {
          if (traceSeqEnabled) {
            llvm::errs() << "[SEQ-CI] wait_for_item_done miss proc=" << procId
                         << " seq=0x" << llvm::format_hex(seqAddr, 16)
                         << "\n";
          }
          return success();
        }

        auto erasePendingItemByProc = [&](uint64_t item) {
          auto it = sequencePendingItemsByProc.find(procId);
          if (it == sequencePendingItemsByProc.end())
            return;
          auto &pending = it->second;
          auto match = std::find(pending.begin(), pending.end(), item);
          if (match != pending.end())
            pending.erase(match);
          if (pending.empty())
            sequencePendingItemsByProc.erase(it);
        };
        auto erasePendingItemBySeq = [&](uint64_t seqKey, uint64_t item) {
          if (seqKey == 0)
            return;
          auto it = sequencePendingItemsBySeq.find(seqKey);
          if (it == sequencePendingItemsBySeq.end())
            return;
          auto &pending = it->second;
          auto match = std::find(pending.begin(), pending.end(), item);
          if (match != pending.end())
            pending.erase(match);
          if (pending.empty())
            sequencePendingItemsBySeq.erase(it);
        };

        if (itemDoneReceived.count(itemAddr)) {
          erasePendingItemByProc(itemAddr);
          erasePendingItemBySeq(seqAddr, itemAddr);
          itemDoneReceived.erase(itemAddr);
          finishItemWaiters.erase(itemAddr);
          (void)takeUvmSequencerItemOwner(itemAddr);
          (void)takeUvmSequenceItemOwner(itemAddr);
          return success();
        }

        finishItemWaiters[itemAddr] = procId;
        auto &pState = processStates[procId];
        pState.waiting = true;
        pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
        if (traceSeqEnabled)
          llvm::errs() << "[SEQ-CI] wait_for_item_done item=0x"
                       << llvm::format_hex(itemAddr, 16) << "\n";
        return success();
      }
    }

    auto isSequencerHandshakeSensitive = [](StringRef name) {
      if (name.empty())
        return false;
      if (name.contains("::start_item") || name.contains("::finish_item") ||
          name.ends_with("::wait_for_grant") ||
          name.ends_with("::send_request") ||
          name.ends_with("::wait_for_item_done"))
        return true;
      bool isSequencerSurface =
          name.contains("seq_item_pull_port") ||
          name.contains("seq_item_pull_imp") || name.contains("sqr_if_base") ||
          name.contains("uvm_sequencer");
      if (!isSequencerSurface)
        return false;
      return name.ends_with("::get") || name.ends_with("::get_next_item") ||
             name.ends_with("::try_next_item") ||
             name.ends_with("::item_done");
    };

    // E5: Per-call-site fast-dispatch cache.
    if (!callIndirectDirectDispatchCacheDisabled) {
      auto siteIt = callIndirectSiteCache.find(callIndirectOp.getOperation());
      bool allowSiteCacheHit = false;
      if (siteIt != callIndirectSiteCache.end() && siteIt->second.valid &&
          siteIt->second.funcAddr == funcAddr &&
          !siteIt->second.hadVtableOverride && !siteIt->second.isIntercepted &&
          siteIt->second.funcOp) {
        StringRef cachedName = siteIt->second.funcOp.getSymName();
        // Keep runtime-vtable override active for sequence body dispatch.
        // Caching the base stub at this site can suppress the override path
        // and drop into "Body definition undefined".
        allowSiteCacheHit =
            cachedName != "uvm_pkg::uvm_sequence_base::body" &&
            !isSequencerHandshakeSensitive(cachedName);
      }
      if (allowSiteCacheHit) {
        ++ciSiteCacheHits;
        if (traceCallIndirectSiteCacheEnabled) {
          auto ovrIt = callIndirectRuntimeOverrideSiteInfo.find(
              callIndirectOp.getOperation());
          int64_t mi =
              (ovrIt != callIndirectRuntimeOverrideSiteInfo.end() &&
               ovrIt->second.hasStaticMethodIndex)
                  ? ovrIt->second.staticMethodIndex
                  : -1;
          maybeTraceCallIndirectSiteCacheHit(mi);
          if (siteIt->second.funcOp)
            llvm::errs() << "[CI-SITE-CACHE] hit-callee="
                         << siteIt->second.funcOp.getSymName() << "\n";
        }
        SmallVector<InterpretedValue, 4> fastArgs;
        for (Value arg : callIndirectOp.getArgOperands())
          fastArgs.push_back(getValue(procId, arg));
        // E5 site cache: entry-table dispatch via cached entry pointer.
        if (siteIt->second.cachedEntryPtr &&
            siteIt->second.funcAddr == funcAddr &&
            processStates[procId].callDepth < 2000 &&
            aotDepth == 0) {
          auto &entry = siteIt->second;
          bool isNativeEntry =
              (entry.cachedFid < compiledFuncIsNative.size() &&
               compiledFuncIsNative[entry.cachedFid]);
          bool hasTrampolineEntry =
              (entry.cachedFid < compiledFuncHasTrampoline.size() &&
               compiledFuncHasTrampoline[entry.cachedFid]);
          noteAotFuncIdCall(entry.cachedFid);
          // Deny/trap checks for call_indirect E5 cache path.
          if (isNativeEntry && aotDenyFids.count(entry.cachedFid))
            goto ci_cache_interpreted;
          if (isNativeEntry &&
              static_cast<int32_t>(entry.cachedFid) == aotTrapFid) {
            llvm::errs() << "[AOT TRAP] ci-cache fid=" << entry.cachedFid;
            if (entry.cachedFid < aotFuncEntryNamesById.size())
              llvm::errs() << " name=" << aotFuncEntryNamesById[entry.cachedFid];
            llvm::errs() << "\n";
            __builtin_trap();
          }
          // Runtime interception policy may mark a FuncId as non-native even
          // when the compiled module still has a direct entry pointer. Only
          // call non-native entries through generated trampolines.
          if (!isNativeEntry && !hasTrampolineEntry)
            goto ci_cache_interpreted;
          // Keep MAY_YIELD entries on interpreted dispatch.
          if (shouldSkipMayYieldEntry(entry.cachedFid, isNativeEntry)) {
            ++entryTableSkippedYieldCount;
            noteAotEntryYieldSkip(entry.cachedFid);
            goto ci_cache_interpreted;
          }
          auto cachedFuncOp = entry.funcOp;
          unsigned numArgs = cachedFuncOp.getNumArguments();
          unsigned numResults = cachedFuncOp.getNumResults();
          bool eligible = (numArgs <= 8 && numResults <= 1);
          if (eligible) {
            for (unsigned i = 0; i < numArgs && eligible; ++i) {
              auto ty = cachedFuncOp.getArgumentTypes()[i];
              if (auto intTy = dyn_cast<mlir::IntegerType>(ty)) {
                if (intTy.getWidth() > 64) eligible = false;
              } else if (isa<mlir::IndexType>(ty) ||
                         isa<mlir::LLVM::LLVMPointerType>(ty)) {
                // OK
              } else {
                eligible = false;
              }
            }
            if (numResults == 1) {
              auto resTy = cachedFuncOp.getResultTypes()[0];
              if (auto intTy = dyn_cast<mlir::IntegerType>(resTy)) {
                if (intTy.getWidth() > 64) eligible = false;
              } else if (!isa<mlir::IndexType>(resTy) &&
                         !isa<mlir::LLVM::LLVMPointerType>(resTy)) {
                eligible = false;
              }
            }
          }
          if (eligible) {
            uint64_t a[8] = {};
            bool normalizePointerArgs = isNativeEntry;
            if (shouldForceInterpretedFragileUvmCallee(
                    cachedFuncOp.getSymName()))
              goto ci_cache_interpreted;
            bool forcePredicateFalse = false;
            eligible = fillNativeCallArgs(fastArgs,
                                          cachedFuncOp.getArgumentTypes(),
                                          cachedFuncOp.getSymName(), numArgs, a,
                                          normalizePointerArgs,
                                          forcePredicateFalse);
            if (forcePredicateFalse) {
              setCallIndirectResults({});
              return success();
            }
            if (!eligible)
              goto ci_cache_interpreted;
            maybeTraceIndirectNative(entry.cachedFid,
                                     cachedFuncOp.getSymName(), isNativeEntry,
                                     numArgs, numResults, a);

            if (eligible) {

            void *fptr = entry.cachedEntryPtr;
            // Set TLS context so Moore runtime helpers can normalize ptrs.
            void *prevTls = __circt_sim_get_tls_ctx();
            __circt_sim_set_tls_ctx(static_cast<void *>(this));
            __circt_sim_set_tls_normalize(LLHDProcessInterpreter::normalizeVirtualPtr);
            uint64_t result = 0;
            using F0 = uint64_t (*)();
            using F1 = uint64_t (*)(uint64_t);
            using F2 = uint64_t (*)(uint64_t, uint64_t);
            using F3 = uint64_t (*)(uint64_t, uint64_t, uint64_t);
            using F4 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);
            using F5 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t);
            using F6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t);
            using F7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t);
            using F8 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t, uint64_t);
            ++processStates[procId].callDepth;
            switch (numArgs) {
            case 0: result = reinterpret_cast<F0>(fptr)(); break;
            case 1: result = reinterpret_cast<F1>(fptr)(a[0]); break;
            case 2: result = reinterpret_cast<F2>(fptr)(a[0], a[1]); break;
            case 3: result = reinterpret_cast<F3>(fptr)(a[0], a[1], a[2]); break;
            case 4: result = reinterpret_cast<F4>(fptr)(a[0], a[1], a[2], a[3]); break;
            case 5: result = reinterpret_cast<F5>(fptr)(a[0], a[1], a[2], a[3], a[4]); break;
            case 6: result = reinterpret_cast<F6>(fptr)(a[0], a[1], a[2], a[3], a[4], a[5]); break;
            case 7: result = reinterpret_cast<F7>(fptr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
            case 8: result = reinterpret_cast<F8>(fptr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
            }
            __circt_sim_set_tls_ctx(prevTls);
            --processStates[procId].callDepth;
            if (numResults == 1) {
              SmallVector<InterpretedValue, 2> nativeResults;
              auto resTy = cachedFuncOp.getResultTypes()[0];
              unsigned bits = 64;
              if (auto intTy = dyn_cast<mlir::IntegerType>(resTy))
                bits = intTy.getWidth();
              nativeResults.push_back(
                  InterpretedValue(llvm::APInt(64, result).zextOrTrunc(bits)));
              setCallIndirectResults(nativeResults);
            }
            if (isNativeEntry)
              ++nativeEntryCallCount;
            else
              ++trampolineEntryCallCount;
            return success();
            } // eligible (no fake addr)
          }
        } else if (siteIt->second.cachedEntryPtr &&
                   siteIt->second.funcAddr == funcAddr &&
                   processStates[procId].callDepth < 2000 &&
                   aotDepth != 0) {
          noteAotFuncIdCall(siteIt->second.cachedFid);
          ++entryTableSkippedDepthCount;
        }
      ci_cache_interpreted:
        // Fall through to interpretFuncBody for non-native-eligible calls.
        auto &fastState = processStates[procId];
        ++interpretedCallCounts[siteIt->second.funcOp.getOperation()];
        ++fastState.callDepth;
        SmallVector<InterpretedValue, 2> fastResults;
        LogicalResult fastCallResult =
            interpretFuncBody(procId, siteIt->second.funcOp, fastArgs,
                              fastResults, callIndirectOp);
        --fastState.callDepth;
        if (failed(fastCallResult)) {
          auto &failState = processStates[procId];
          if (failState.waiting)
            return success();
          if (shouldPropagateCoverageRuntimeFailure(procId))
            return failure();
          for (Value result : callIndirectOp.getResults()) {
            unsigned width = getTypeWidth(result.getType());
            setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
          }
          return success();
        }
        if (processStates[procId].waiting)
          return success();
        setCallIndirectResults(fastResults);
        return success();
      }
      ++ciSiteCacheMisses;
    }

    // [CI-DISPATCH] diagnostic removed
    LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: resolved 0x"
                            << llvm::format_hex(funcAddr, 16)
                            << " -> " << calleeName << "\n");

    if (profilingEnabled)
      ++funcCallProfile[calleeName];

    if (calleeName.contains("uvm_component::set_domain") ||
        calleeName.contains("uvm_phase::add")) {
      auto &cacheState = processStates[procId];
      if (!cacheState.funcResultCache.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: invalidating func result cache ("
                   << cacheState.funcResultCache.size()
                   << " functions cached) due to " << calleeName << "\n");
        cacheState.funcResultCache.clear();
      }
      if (!sharedFuncResultCache.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: invalidating shared func result cache ("
                   << sharedFuncResultCache.size()
                   << " functions cached) due to " << calleeName << "\n");
        sharedFuncResultCache.clear();
      }
    }

    // Runtime vtable override for the direct resolution path.
    // When the static function address maps to a base-class method, check
    // the self object's actual runtime vtable and resolve to the derived
    // class's override if it differs.
    do {
      int64_t methodIndex = -1;
      if (!getCachedCallIndirectStaticMethodIndex(callIndirectOp,
                                                  methodIndex)) {
        // Fallback for dynamic slot indices: resolve the method index
        // from the call-site SSA chain on this activation.
        auto castOp =
            calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          break;
        auto funcPtrLoad = castOp.getInputs()[0].getDefiningOp<LLVM::LoadOp>();
        if (!funcPtrLoad)
          break;
        auto vtableGEP = funcPtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!vtableGEP)
          break;
        auto vtableIndices = vtableGEP.getIndices();
        if (vtableIndices.size() >= 2) {
          auto lastIdx = vtableIndices[vtableIndices.size() - 1];
          if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(lastIdx))
            methodIndex = intAttr.getInt();
          else if (auto dynIdx = llvm::dyn_cast_if_present<Value>(lastIdx)) {
            InterpretedValue dynVal = getValue(procId, dynIdx);
            if (!dynVal.isX())
              methodIndex = static_cast<int64_t>(dynVal.getUInt64());
          }
        }
      }
      if (methodIndex < 0)
        break;

      // Read the self object's runtime vtable pointer
      if (callIndirectOp.getArgOperands().empty())
        break;
      InterpretedValue selfVal = getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (selfVal.isX())
        break;
      uint64_t objAddr = selfVal.getUInt64();
      uint64_t runtimeVtableAddr = 0;
      if (!readObjectVTableAddress(objAddr, runtimeVtableAddr, procId)) {
        // Sequence body can run in a forked child where the object header
        // vtable pointer is observed as zero. Recover from the vtable cached
        // at sequence start-time so virtual body override still resolves.
        if (methodIndex == 43 &&
            calleeName == "uvm_pkg::uvm_sequence_base::body") {
          if (!lookupCachedSequenceRuntimeVtable(procId, objAddr,
                                                 runtimeVtableAddr))
            break;
        } else {
          break;
        }
      }
      if (runtimeVtableAddr == 0)
        break;
      if (methodIndex == 38)
        cacheSequenceRuntimeVtableForObject(procId, objAddr, runtimeVtableAddr);
      uint64_t runtimeFuncAddr = 0;
      auto cacheKey = std::make_pair(runtimeVtableAddr, methodIndex);
      auto cacheIt = callIndirectRuntimeVtableSlotCache.find(cacheKey);
      if (cacheIt != callIndirectRuntimeVtableSlotCache.end()) {
        runtimeFuncAddr = cacheIt->second;
        if (traceCallIndirectSiteCacheEnabled) {
          llvm::StringRef funcName = "<unknown>";
          if (auto fnIt = addressToFunction.find(runtimeFuncAddr);
              fnIt != addressToFunction.end())
            funcName = fnIt->second;
          llvm::errs() << "[CI-SITE-CACHE] runtime-slot-hit vtable=0x"
                       << llvm::format_hex(runtimeVtableAddr, 16)
                       << " method_index=" << methodIndex << " func=0x"
                       << llvm::format_hex(runtimeFuncAddr, 16)
                       << " name=" << funcName << "\n";
        }
      } else {
        auto globalIt = addressToGlobal.find(runtimeVtableAddr);
        if (globalIt == addressToGlobal.end())
          break;
        std::string runtimeVtableName = globalIt->second;

        // Read the function pointer from the runtime vtable at the same slot.
        auto vtableBlockIt = globalMemoryBlocks.find(runtimeVtableName);
        if (vtableBlockIt == globalMemoryBlocks.end())
          break;
        auto &vtableBlock = vtableBlockIt->second;
        unsigned slotOffset = methodIndex * 8;
        if (slotOffset + 8 > vtableBlock.size)
          break;
        for (unsigned i = 0; i < 8; ++i)
          runtimeFuncAddr |=
              static_cast<uint64_t>(vtableBlock[slotOffset + i])
              << (i * 8);
        callIndirectRuntimeVtableSlotCache[cacheKey] = runtimeFuncAddr;
        if (traceCallIndirectSiteCacheEnabled) {
          llvm::StringRef funcName = "<unknown>";
          if (auto fnIt = addressToFunction.find(runtimeFuncAddr);
              fnIt != addressToFunction.end())
            funcName = fnIt->second;
          llvm::errs() << "[CI-SITE-CACHE] runtime-slot-store vtable=0x"
                       << llvm::format_hex(runtimeVtableAddr, 16)
                       << " method_index=" << methodIndex << " func=0x"
                       << llvm::format_hex(runtimeFuncAddr, 16)
                       << " name=" << funcName << "\n";
        }
      }
      if (runtimeFuncAddr == 0 || runtimeFuncAddr == funcAddr)
        break;

      // The runtime vtable has a different function at this slot — override
      auto runtimeFuncIt = addressToFunction.find(runtimeFuncAddr);
      if (runtimeFuncIt == addressToFunction.end())
        break;

      std::string runtimeVtableName = "<cached>";
      if (auto globalIt = addressToGlobal.find(runtimeVtableAddr);
          globalIt != addressToGlobal.end())
        runtimeVtableName = globalIt->second;
      overriddenCalleeName = runtimeFuncIt->second;
      LLVM_DEBUG(llvm::dbgs()
                 << "  func.call_indirect: runtime vtable override: "
                 << calleeName << " -> " << overriddenCalleeName
                 << " (vtable=" << runtimeVtableName
                 << " slot=" << methodIndex << ")\n");
      calleeName = overriddenCalleeName;
    } while (false);

    noteResolvedTarget(calleeName);
    if (!nativeFactoryOverridesConfigured &&
        isUvmFactoryOverrideSetter(calleeName))
      nativeFactoryOverridesConfigured = true;
    if (!nativeFactoryOverridesConfigured &&
        (calleeName.starts_with("set_type_override_") ||
         calleeName.starts_with("set_inst_override_")))
      nativeFactoryOverridesConfigured = true;
    if (!nativeFactoryInstanceOverridesConfigured &&
        isUvmFactoryInstanceOverrideSetter(calleeName))
      nativeFactoryInstanceOverridesConfigured = true;
    if (!nativeFactoryInstanceOverridesConfigured &&
        calleeName.starts_with("set_inst_override_"))
      nativeFactoryInstanceOverridesConfigured = true;
    SimTime now = scheduler.getCurrentTime();
    maybeTraceFilteredCall(procId, "func.call_indirect", calleeName,
                           now.realTime, now.deltaStep);

    if (traceConfigDbEnabled && calleeName.contains("config_db")) {
      llvm::errs() << "[CFG-CI-DISPATCH] callee=" << calleeName
                   << " nargs=" << callIndirectOp.getArgOperands().size()
                   << " nresults=" << callIndirectOp.getNumResults() << "\n";
    }
    if (traceConfigDbEnabled && procId == 1)
      llvm::errs() << "[CFG-CI-PROC1] callee=" << calleeName << "\n";

    // Intercept uvm_default_factory::register — fast-path native
    // registration. The original MLIR calls get_type_name 3-7 times via
    // vtable, does string comparisons, assoc array lookups, and override
    // scanning (~2300 steps each × 1078 types = 2.5M steps = ~80s).
    // We call get_type_name once (~6 ops) and store in a C++ map.
    if (!disableUvmFactoryFastPaths() &&
        (calleeName == "uvm_pkg::uvm_default_factory::register" ||
         calleeName == "uvm_pkg::uvm_factory::register")) {
      bool registered = false;
      do {
        if (callIndirectOp.getArgOperands().size() < 2)
          break;
        InterpretedValue wrapperVal =
            getValue(procId, callIndirectOp.getArgOperands()[1]);
        if (wrapperVal.isX() || wrapperVal.getUInt64() == 0)
          break;
        uint64_t wrapperAddr = wrapperVal.getUInt64();

        uint64_t vtableAddr = 0;
        if (!readObjectVTableAddress(wrapperAddr, vtableAddr, procId))
          break;

        // Read vtable entry [2] = get_type_name
        uint64_t off2 = 0;
        MemoryBlock *vtableBlk =
            findBlockByAddress(vtableAddr + 2 * 8, off2);
        if (!vtableBlk || !vtableBlk->initialized ||
            off2 + 8 > vtableBlk->size)
          break;
        uint64_t funcAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          funcAddr |=
              static_cast<uint64_t>(vtableBlk->bytes()[off2 + i]) << (i * 8);
        auto funcIt = addressToFunction.find(funcAddr);
        if (funcIt == addressToFunction.end())
          break;

        // Call get_type_name(wrapper) → struct<(ptr, i64)>
        auto funcOp =
            rootModule.lookupSymbol<mlir::func::FuncOp>(funcIt->second);
        if (!funcOp)
          break;
        SmallVector<InterpretedValue, 1> results;
        if (failed(interpretFuncBody(procId, funcOp, {wrapperVal}, results)) ||
            results.empty())
          break;

        // Extract (ptr, length) from 128-bit struct<(ptr, i64)>
        InterpretedValue nameStruct = results[0];
        uint64_t strAddr = 0;
        uint64_t strLen = 0;
        if (!decodePackedPtrLenPayload(nameStruct, strAddr, strLen))
          break;
        if (strLen == 0 || strLen > 1024 || strAddr == 0)
          break;

        // Read the string content from the packed string global
        uint64_t strOff = 0;
        MemoryBlock *strBlk = findBlockByAddress(strAddr, strOff);
        if (!strBlk || !strBlk->initialized ||
            strOff + strLen > strBlk->size)
          break;
        std::string typeName(
            reinterpret_cast<const char *>(strBlk->bytes() + strOff),
            strLen);

        // Store in native factory map
        nativeFactoryTypeNames[typeName] = wrapperAddr;
        registered = true;
      } while (false);
      if (registered) {
        return success();
      }
      // Fast-path failed, fall through to normal MLIR interpretation
      // so the type still gets registered. Critical for test classes.
      // Don't return — fall through to normal call handling so
      // the type still gets registered. Critical for test classes.
    }

    auto tryInvokeWrapperFactoryMethod =
        [&](uint64_t wrapperAddr, uint64_t slotIndex,
            llvm::ArrayRef<InterpretedValue> extraArgs,
            InterpretedValue &outResult) -> bool {
      uint64_t vtableAddr = 0;
      if (!readObjectVTableAddress(wrapperAddr, vtableAddr, procId))
        return false;

      uint64_t off2 = 0;
      MemoryBlock *vtBlk = findBlockByAddress(vtableAddr + slotIndex * 8, off2);
      if (!vtBlk || !vtBlk->initialized || off2 + 8 > vtBlk->size)
        return false;

      uint64_t funcAddr = 0;
      for (unsigned i = 0; i < 8; ++i)
        funcAddr |= static_cast<uint64_t>(vtBlk->bytes()[off2 + i]) << (i * 8);
      auto funcIt = addressToFunction.find(funcAddr);
      if (funcIt == addressToFunction.end())
        return false;

      auto funcOp = rootModule.lookupSymbol<mlir::func::FuncOp>(funcIt->second);
      if (!funcOp)
        return false;

      SmallVector<InterpretedValue, 4> invokeArgs;
      invokeArgs.push_back(InterpretedValue(wrapperAddr, 64));
      invokeArgs.append(extraArgs.begin(), extraArgs.end());
      SmallVector<InterpretedValue, 1> results;
      if (failed(interpretFuncBody(procId, funcOp, invokeArgs, results)) ||
          results.empty())
        return false;
      outResult = results.front();

      // Guard against partially initialized objects returned by aggressive
      // wrapper dispatch: a valid Moore class instance must have a non-zero
      // class-handle (field[0] in uvm_void). If the handle is still zero,
      // let the original MLIR path execute instead.
      if (!outResult.isX()) {
        uint64_t objAddr = outResult.getUInt64();
        if (objAddr != 0) {
          bool haveClassHandle = false;
          int32_t classHandle = 0;

          uint64_t objOff = 0;
          MemoryBlock *objBlk = findBlockByAddress(objAddr, objOff);
          if (!objBlk)
            objBlk = findMemoryBlockByAddress(objAddr, procId, &objOff);
          if (objBlk && objBlk->initialized && objOff + 4 <= objBlk->size) {
            uint32_t raw = 0;
            for (unsigned i = 0; i < 4; ++i)
              raw |= static_cast<uint32_t>(objBlk->bytes()[objOff + i])
                     << (i * 8);
            classHandle = static_cast<int32_t>(raw);
            haveClassHandle = true;
          } else {
            uint64_t nativeOff = 0;
            size_t nativeSize = 0;
            if (findNativeMemoryBlockByAddress(objAddr, &nativeOff, &nativeSize) &&
                nativeOff + 4 <= nativeSize) {
              std::memcpy(&classHandle,
                          reinterpret_cast<const void *>(objAddr),
                          sizeof(classHandle));
              haveClassHandle = true;
            }
          }

          if (haveClassHandle && classHandle == 0)
            return false;
        }
      }
      return true;
    };

    // Intercept create_component_by_type/object_by_type — when factory
    // register fast-path stores wrappers in nativeFactoryTypeNames, executing
    // the full MLIR path can still miss wrappers during early startup.
    // By-type calls already provide the wrapper pointer directly (arg1), so
    // dispatch straight to wrapper vtable create_* methods.
    if (!nativeFactoryOverridesConfigured &&
        !disableUvmFactoryByTypeFastPath() &&
        (calleeName ==
             "uvm_pkg::uvm_default_factory::create_component_by_type" ||
         calleeName == "uvm_pkg::uvm_factory::create_component_by_type") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 5) {
      InterpretedValue wrapperVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      if (traceUvmFactoryByTypeEnabled()) {
        llvm::errs() << "[UVM-BYTYPE] component callee=" << calleeName
                     << " proc=" << procId
                     << " wrapper=0x"
                     << llvm::format_hex(wrapperVal.isX() ? 0
                                                          : wrapperVal.getUInt64(),
                                         16)
                     << " wrapperX=" << (wrapperVal.isX() ? 1 : 0) << "\n";
      }
      if (!wrapperVal.isX() && wrapperVal.getUInt64() != 0) {
        InterpretedValue nameArg =
            getValue(procId, callIndirectOp.getArgOperands()[3]);
        InterpretedValue parentArg =
            getValue(procId, callIndirectOp.getArgOperands()[4]);
        InterpretedValue createdObj;
        if (tryInvokeWrapperFactoryMethod(
                wrapperVal.getUInt64(),
                /*slotIndex=*/1, {nameArg, parentArg}, createdObj)) {
          if (traceUvmFactoryByTypeEnabled()) {
            llvm::errs() << "[UVM-BYTYPE] component fastpath-hit wrapper=0x"
                         << llvm::format_hex(wrapperVal.getUInt64(), 16)
                         << " result=0x"
                         << llvm::format_hex(createdObj.isX()
                                                 ? 0
                                                 : createdObj.getUInt64(),
                                             16)
                         << " resultX=" << (createdObj.isX() ? 1 : 0) << "\n";
          }
          setValue(procId, callIndirectOp.getResults()[0], createdObj);
          return success();
        }
        if (traceUvmFactoryByTypeEnabled())
          llvm::errs() << "[UVM-BYTYPE] component fastpath-miss wrapper=0x"
                       << llvm::format_hex(wrapperVal.getUInt64(), 16) << "\n";
      }
      // Fall through to MLIR interpretation if fast-path fails.
    }

    if (!nativeFactoryOverridesConfigured &&
        !disableUvmFactoryByTypeFastPath() &&
        (calleeName ==
             "uvm_pkg::uvm_default_factory::create_object_by_type" ||
         calleeName == "uvm_pkg::uvm_factory::create_object_by_type") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 4) {
      InterpretedValue wrapperVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      if (traceUvmFactoryByTypeEnabled()) {
        llvm::errs() << "[UVM-BYTYPE] object callee=" << calleeName
                     << " proc=" << procId
                     << " wrapper=0x"
                     << llvm::format_hex(wrapperVal.isX() ? 0
                                                          : wrapperVal.getUInt64(),
                                         16)
                     << " wrapperX=" << (wrapperVal.isX() ? 1 : 0) << "\n";
      }
      if (!wrapperVal.isX() && wrapperVal.getUInt64() != 0) {
        InterpretedValue nameArg =
            getValue(procId, callIndirectOp.getArgOperands()[3]);
        InterpretedValue createdObj;
        if (tryInvokeWrapperFactoryMethod(
                wrapperVal.getUInt64(),
                /*slotIndex=*/0, {nameArg}, createdObj)) {
          if (traceUvmFactoryByTypeEnabled()) {
            llvm::errs() << "[UVM-BYTYPE] object fastpath-hit wrapper=0x"
                         << llvm::format_hex(wrapperVal.getUInt64(), 16)
                         << " result=0x"
                         << llvm::format_hex(createdObj.isX()
                                                 ? 0
                                                 : createdObj.getUInt64(),
                                             16)
                         << " resultX=" << (createdObj.isX() ? 1 : 0) << "\n";
          }
          setValue(procId, callIndirectOp.getResults()[0], createdObj);
          return success();
        }
        if (traceUvmFactoryByTypeEnabled())
          llvm::errs() << "[UVM-BYTYPE] object fastpath-miss wrapper=0x"
                       << llvm::format_hex(wrapperVal.getUInt64(), 16) << "\n";
      }
      // Fall through to MLIR interpretation if fast-path fails.
    }

    // Intercept create_component_by_name — since factory.register was
    // fast-pathed (skipping MLIR-side data population), the MLIR-side
    // create_component_by_name won't find registered types. This
    // intercept looks up the wrapper from the C++ map and calls
    // create_component via the wrapper's vtable slot 1.
    // Signature: (this, requested_type_name: struct<(ptr,i64)>,
    //             parent_inst_path: struct<(ptr,i64)>,
    //             name: struct<(ptr,i64)>, parent: ptr) -> ptr
    if (!disableUvmFactoryFastPaths() &&
        (calleeName ==
             "uvm_pkg::uvm_default_factory::create_component_by_name" ||
         calleeName == "uvm_pkg::uvm_factory::create_component_by_name") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 5) {
      // Extract the requested type name string (arg1).
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      std::string requestedName;
      bool nameExtracted = false;
      uint64_t strAddr = 0;
      uint64_t strLenBits = 0;
      if (decodePackedPtrLenPayload(nameVal, strAddr, strLenBits)) {
        int64_t strLen = static_cast<int64_t>(strLenBits);
        if (strLen > 0 && strLen <= 1024 && strAddr != 0) {
          nameExtracted =
              tryReadStringKey(procId, strAddr, strLen, requestedName);
        }
      }
      if (nameExtracted && !requestedName.empty()) {
        auto it = nativeFactoryTypeNames.find(requestedName);
        if (it != nativeFactoryTypeNames.end()) {
          uint64_t wrapperAddr = it->second;
          InterpretedValue nameArg =
              getValue(procId, callIndirectOp.getArgOperands()[3]);
          InterpretedValue parentArg =
              getValue(procId, callIndirectOp.getArgOperands()[4]);
          InterpretedValue createdObj;
          if (tryInvokeWrapperFactoryMethod(
                  wrapperAddr,
                  /*slotIndex=*/1, {nameArg, parentArg}, createdObj)) {
            setValue(procId, callIndirectOp.getResults()[0], createdObj);
            return success();
          }
        }
      }
      // Fall through to MLIR interpretation if fast-path fails.
    }

    // Intercept create_object_by_name — mirrors create_component_by_name but
    // dispatches wrapper slot 0 (create_object).
    if (!disableUvmFactoryFastPaths() &&
        (calleeName ==
             "uvm_pkg::uvm_default_factory::create_object_by_name" ||
         calleeName == "uvm_pkg::uvm_factory::create_object_by_name") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 4) {
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      std::string requestedName;
      bool nameExtracted = false;
      uint64_t strAddr = 0;
      uint64_t strLenBits = 0;
      if (decodePackedPtrLenPayload(nameVal, strAddr, strLenBits)) {
        int64_t strLen = static_cast<int64_t>(strLenBits);
        if (strLen > 0 && strLen <= 1024 && strAddr != 0) {
          nameExtracted =
              tryReadStringKey(procId, strAddr, strLen, requestedName);
        }
      }
      if (nameExtracted && !requestedName.empty()) {
        auto it = nativeFactoryTypeNames.find(requestedName);
        if (it != nativeFactoryTypeNames.end()) {
          InterpretedValue nameArg =
              getValue(procId, callIndirectOp.getArgOperands()[3]);
          InterpretedValue createdObj;
          if (tryInvokeWrapperFactoryMethod(
                  it->second, /*slotIndex=*/0, {nameArg}, createdObj)) {
            setValue(procId, callIndirectOp.getResults()[0], createdObj);
            return success();
          }
        }
      }
      // Fall through to MLIR interpretation if fast-path fails.
    }

    // Intercept find_wrapper_by_name — uses nativeFactoryTypeNames to look
    // up a type name registered by the fast-path factory.register above.
    // This is needed for +UVM_TESTNAME to find the test class wrapper.
    if (!disableUvmFactoryFastPaths() &&
        (calleeName == "uvm_pkg::uvm_default_factory::find_wrapper_by_name" ||
         calleeName == "uvm_pkg::uvm_factory::find_wrapper_by_name") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 2) {
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      uint64_t wrapperResult = 0;
      uint64_t strAddr = 0;
      uint64_t strLenBits = 0;
      if (!nativeFactoryTypeNames.empty() &&
          decodePackedPtrLenPayload(nameVal, strAddr, strLenBits)) {
        int64_t strLen = static_cast<int64_t>(strLenBits);
        std::string searchName;
        if (strLen > 0 && strLen <= 1024 && strAddr != 0 &&
            tryReadStringKey(procId, strAddr, strLen, searchName)) {
          auto it = nativeFactoryTypeNames.find(searchName);
          if (it != nativeFactoryTypeNames.end())
            wrapperResult = it->second;
        }
      }
      setValue(procId, callIndirectOp.getResults()[0],
               InterpretedValue(wrapperResult, 64));
      return success();
    }

    // Intercept is_type_name_registered — checks if a type name was
    // registered by the fast-path factory.register above.
    if (!disableUvmFactoryFastPaths() &&
        (calleeName == "uvm_pkg::uvm_default_factory::is_type_name_registered" ||
         calleeName == "uvm_pkg::uvm_factory::is_type_name_registered") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 2) {
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      bool found = false;
      uint64_t strAddr = 0;
      uint64_t strLenBits = 0;
      if (!nativeFactoryTypeNames.empty() &&
          decodePackedPtrLenPayload(nameVal, strAddr, strLenBits)) {
        int64_t strLen = static_cast<int64_t>(strLenBits);
        std::string searchName;
        if (strLen > 0 && strLen <= 1024 && strAddr != 0 &&
            tryReadStringKey(procId, strAddr, strLen, searchName)) {
          found = nativeFactoryTypeNames.count(searchName) > 0;
        }
      }
      setValue(procId, callIndirectOp.getResults()[0],
               InterpretedValue(llvm::APInt(1, found ? 1 : 0)));
      return success();
    }

    // Intercept is_type_registered — checks if a wrapper ptr was registered.
    if (!disableUvmFactoryFastPaths() &&
        (calleeName == "uvm_pkg::uvm_default_factory::is_type_registered" ||
         calleeName == "uvm_pkg::uvm_factory::is_type_registered") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 2) {
      InterpretedValue wrapperVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      bool found = false;
      if (!wrapperVal.isX() && wrapperVal.getUInt64() != 0) {
        uint64_t addr = wrapperVal.getUInt64();
        for (auto &[name, wAddr] : nativeFactoryTypeNames) {
          if (wAddr == addr) {
            found = true;
            break;
          }
        }
      }
      setValue(procId, callIndirectOp.getResults()[0],
               InterpretedValue(llvm::APInt(1, found ? 1 : 0)));
      return success();
    }

    if (handleUvmCallIndirectFastPath(procId, callIndirectOp, calleeName))
      return success();

    // Intercept end_of_elaboration_phase for any class whose EOE function
    // body calls uvm_driver::end_of_elaboration_phase (which checks
    // seq_item_port.size() and emits DRVCONNECT). We intercept at this
    // level for direct uvm_driver dispatches, and also suppress derived
    // class overrides that call super.end_of_elaboration_phase via func.call.
    if (calleeName.contains("end_of_elaboration_phase") &&
        calleeName.contains("uvm_driver") &&
        !calleeName.contains("driver_proxy")) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  call_indirect: end_of_elaboration_phase "
                 << "intercepted (no-op, suppresses DRVCONNECT): "
                 << calleeName << "\n");
      return success();
    }

    // Intercept get_uvm_seeding helper and virtual methods.
    if ((calleeName == "get_uvm_seeding" ||
         calleeName.ends_with("::get_uvm_seeding")) &&
        callIndirectOp.getNumResults() >= 1) {
      bool value = true;
      auto it =
          globalAddresses.find("uvm_pkg::uvm_pkg::uvm_object::use_uvm_seeding");
      if (it != globalAddresses.end()) {
        uint64_t globalAddr = it->getValue();
        uint64_t off = 0;
        if (MemoryBlock *blk = findBlockByAddress(globalAddr, off))
          if (blk->initialized && off < blk->size)
            value = (blk->bytes()[off] & 1) != 0;
      }
      unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
      if (width == 0)
        width = 1;
      setValue(procId, callIndirectOp.getResult(0),
               InterpretedValue(llvm::APInt(width, value ? 1 : 0)));
      return success();
    }

    // Fast-path cached get_type_name results (cached after first successful
    // interpretation of each callee symbol).
    const bool disableTypeNameCache =
        std::getenv("CIRCT_SIM_DISABLE_UVM_TYPENAME_CACHE") != nullptr;
    if (!disableTypeNameCache && calleeName.contains("get_type_name") &&
        !calleeName.contains("get_type_name_enabled") &&
        callIndirectOp.getNumResults() >= 1) {
      if (auto cachedIt = cachedTypeNameByCallee.find(calleeName);
          cachedIt != cachedTypeNameByCallee.end()) {
        setValue(procId, callIndirectOp.getResult(0), cachedIt->second);
        return success();
      }
    }

    // Intercept get_parent on uvm_component - returns field[9] at offset 87.
    if (calleeName.contains("get_parent") &&
        calleeName.contains("uvm_component") &&
        callIndirectOp.getNumResults() >= 1 &&
        !callIndirectOp.getArgOperands().empty()) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (!selfVal.isX() && selfVal.getUInt64() >= 0x1000) {
        constexpr uint64_t kParentOff2 = 87;
        uint64_t off = 0;
        MemoryBlock *blk = findBlockByAddress(selfVal.getUInt64() + kParentOff2, off);
        if (!blk)
          blk = findMemoryBlockByAddress(selfVal.getUInt64() + kParentOff2, procId, &off);
        if (blk && blk->initialized && off + 8 <= blk->size) {
          uint64_t parentAddr = 0;
          for (unsigned i = 0; i < 8; ++i)
            parentAddr |= static_cast<uint64_t>(blk->bytes()[off + i]) << (i * 8);
          setValue(procId, callIndirectOp.getResult(0),
                   InterpretedValue(parentAddr, 64));
          return success();
        }
      }
    }

    // Intercept get_factory on core-service objects. Read the factory pointer
    // directly and fall back to the global core-service singleton if `this`
    // is transiently invalid during startup.
    if ((calleeName == "uvm_pkg::uvm_default_coreservice_t::get_factory" ||
         calleeName == "uvm_pkg::uvm_coreservice_t::get_factory") &&
        callIndirectOp.getNumResults() >= 1 &&
        !callIndirectOp.getArgOperands().empty()) {
      const bool traceGetFactory =
          std::getenv("CIRCT_SIM_TRACE_UVM_GET_FACTORY") != nullptr;
      auto readU64FromAddress = [&](uint64_t addr, uint64_t &value) -> bool {
        value = 0;
        uint64_t off = 0;
        MemoryBlock *blk = findMemoryBlockByAddress(addr, procId, &off);
        if (!blk)
          blk = findBlockByAddress(addr, off);
        if (blk && blk->initialized && off + 8 <= blk->size) {
          for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<uint64_t>(blk->bytes()[off + i]) << (i * 8);
          return true;
        }

        uint64_t nativeOff = 0;
        size_t nativeSize = 0;
        if (findNativeMemoryBlockByAddress(addr, &nativeOff, &nativeSize) &&
            nativeOff + 8 <= nativeSize) {
          std::memcpy(&value, reinterpret_cast<const void *>(addr), sizeof(value));
          return true;
        }
        return false;
      };
      auto gatherCoreServiceCandidates =
          [&](uint64_t rawAddr, SmallVectorImpl<uint64_t> &candidates) {
            auto addCandidate = [&](uint64_t candidate) {
              if (candidate < 0x1000)
                return;
              if (std::find(candidates.begin(), candidates.end(), candidate) !=
                  candidates.end())
                return;
              candidates.push_back(candidate);
            };
            addCandidate(rawAddr);
            addCandidate(canonicalizeUvmObjectAddress(procId, rawAddr));
            const uint64_t maskedCandidates[] = {
                rawAddr & ~uint64_t(1), rawAddr & ~uint64_t(3),
                rawAddr & ~uint64_t(7)};
            for (uint64_t masked : maskedCandidates) {
              addCandidate(masked);
              addCandidate(canonicalizeUvmObjectAddress(procId, masked));
            }
          };
      auto isPlausibleRuntimePointer = [&](uint64_t addr) -> bool {
        // CIRCT-sim object addresses live in the canonical low virtual range.
        // Reject shifted/misaligned artifacts such as 0xXXXXXXXX00000000.
        return addr >= 0x1000 && addr < (uint64_t(1) << 47);
      };
      auto isLikelyFactoryObject = [&](uint64_t factoryAddr) -> bool {
        if (!isPlausibleRuntimePointer(factoryAddr))
          return false;
        uint64_t vtableAddr = 0;
        if (!readObjectVTableAddress(factoryAddr, vtableAddr, procId))
          return false;
        auto globalIt = addressToGlobal.find(vtableAddr);
        if (globalIt == addressToGlobal.end())
          return false;
        llvm::StringRef vtableName = globalIt->second;
        return vtableName.contains("uvm_default_factory::__vtable__") ||
               vtableName.contains("uvm_factory::__vtable__");
      };
      auto tryReadFactoryFromCoreService = [&](uint64_t rawCoreServiceAddr,
                                               uint64_t &factoryPtr,
                                               uint64_t &coreServiceAddr) -> bool {
        // ABI-aligned layout places `factory` at offset 16 on 64-bit targets.
        // Keep legacy packed offset 12 as a compatibility fallback.
        constexpr uint64_t kFactoryOffsets[] = {16, 12};
        SmallVector<uint64_t, 8> candidates;
        gatherCoreServiceCandidates(rawCoreServiceAddr, candidates);
        for (uint64_t candidate : candidates) {
          if (coreServiceAddr == 0)
            coreServiceAddr = candidate;

          uint64_t fallbackFactory = 0;
          for (uint64_t offset : kFactoryOffsets) {
            uint64_t candidateFactory = 0;
            if (!readU64FromAddress(candidate + offset, candidateFactory) ||
                candidateFactory == 0)
              continue;
            if (!isPlausibleRuntimePointer(candidateFactory))
              continue;
            if (fallbackFactory == 0)
              fallbackFactory = candidateFactory;
            if (isLikelyFactoryObject(candidateFactory)) {
              factoryPtr = candidateFactory;
              coreServiceAddr = candidate;
              return true;
            }
          }

          if (fallbackFactory != 0) {
            factoryPtr = fallbackFactory;
            coreServiceAddr = candidate;
            return true;
          }
        }
        return false;
      };

      uint64_t factoryPtr = 0;
      uint64_t coreServicePtrForFallback = 0;
      auto tryResolveFactoryFromCoreService = [&](uint64_t rawCoreServiceAddr,
                                                  llvm::StringRef source) {
        uint64_t resolvedCoreService = 0;
        bool resolved = tryReadFactoryFromCoreService(
            rawCoreServiceAddr, factoryPtr, resolvedCoreService);
        if (traceGetFactory) {
          llvm::errs() << "[UVM-GET-FACTORY] proc=" << procId
                       << " source=" << source
                       << " raw_core=0x"
                       << llvm::format_hex(rawCoreServiceAddr, 16)
                       << " resolved_core=0x"
                       << llvm::format_hex(resolvedCoreService, 16)
                       << " factory=0x" << llvm::format_hex(factoryPtr, 16)
                       << " resolved=" << (resolved ? 1 : 0) << "\n";
        }
        if (resolved) {
          if (resolvedCoreService != 0)
            coreServicePtrForFallback = resolvedCoreService;
          return true;
        }
        if (resolvedCoreService != 0)
          coreServicePtrForFallback = resolvedCoreService;
        return false;
      };
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (traceGetFactory) {
        llvm::errs() << "[UVM-GET-FACTORY] proc=" << procId
                     << " callee=" << calleeName << " self=0x"
                     << (selfVal.isX() ? llvm::format_hex(0, 16)
                                       : llvm::format_hex(selfVal.getUInt64(), 16))
                     << " self_is_x=" << (selfVal.isX() ? 1 : 0) << "\n";
      }
      if (!selfVal.isX())
        (void)tryResolveFactoryFromCoreService(selfVal.getUInt64(), "self");

      if (factoryPtr == 0) {
        auto instIt =
            globalAddresses.find("uvm_pkg::uvm_pkg::uvm_coreservice_t::inst");
        if (instIt != globalAddresses.end()) {
          uint64_t coreServicePtr = 0;
          if (readU64FromAddress(instIt->second, coreServicePtr))
            (void)tryResolveFactoryFromCoreService(coreServicePtr, "global");
        }
      }

      if (factoryPtr == 0 && rootModule) {
        auto get0Func = rootModule.lookupSymbol<func::FuncOp>("get_0");
        if (get0Func && !get0Func.getBody().empty()) {
          auto &state = processStates[procId];
          constexpr size_t maxCallDepth = 200;
          if (state.callDepth < maxCallDepth) {
            ++state.callDepth;
            SmallVector<InterpretedValue, 1> get0Results;
            LogicalResult get0Status = interpretFuncBody(
                procId, get0Func, {}, get0Results, callIndirectOp.getOperation());
            --state.callDepth;
            if (succeeded(get0Status) && !get0Results.empty() &&
                !get0Results.front().isX())
              (void)tryResolveFactoryFromCoreService(
                  get0Results.front().getUInt64(), "get_0");
          }
        }
      }

      if (factoryPtr != 0) {
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(factoryPtr, 64));
        return success();
      }

      // If dispatch hits abstract base get_factory (returns null), force the
      // concrete default implementation to preserve lazy factory allocation.
      if (coreServicePtrForFallback != 0 && rootModule) {
        auto defaultGetFactoryFunc = rootModule.lookupSymbol<func::FuncOp>(
            "uvm_pkg::uvm_default_coreservice_t::get_factory");
        if (defaultGetFactoryFunc && !defaultGetFactoryFunc.getBody().empty()) {
          auto &state = processStates[procId];
          constexpr size_t maxCallDepth = 200;
          if (state.callDepth < maxCallDepth) {
            ++state.callDepth;
            SmallVector<InterpretedValue, 1> invokeResults;
            LogicalResult invokeStatus = interpretFuncBody(
                procId, defaultGetFactoryFunc,
                {InterpretedValue(coreServicePtrForFallback, 64)}, invokeResults,
                callIndirectOp.getOperation());
            --state.callDepth;
            if (succeeded(invokeStatus) && !invokeResults.empty() &&
                !invokeResults.front().isX() &&
                invokeResults.front().getUInt64() != 0) {
              setValue(procId, callIndirectOp.getResult(0), invokeResults.front());
              return success();
            }
          }
        }
      }
    }

    // Look up the function
    auto &state = processStates[procId];
    Operation *parent = state.processOrInitialOp;
    while (parent && !isa<ModuleOp>(parent))
      parent = parent->getParentOp();

    // Use rootModule as fallback for global constructors
    ModuleOp moduleOp = parent ? cast<ModuleOp>(parent) : rootModule;
    mlir::func::FuncOp funcOp;
    bool hadOverride = !overriddenCalleeName.empty();
    if (!hadOverride && !callIndirectDirectDispatchCacheDisabled) {
      auto dcIt = callIndirectDispatchCache.find(funcAddr);
      if (dcIt != callIndirectDispatchCache.end()) {
        funcOp = dcIt->second;
        ++ciDispatchCacheHits;
      } else {
        funcOp = moduleOp.lookupSymbol<func::FuncOp>(calleeName);
        if (funcOp) {
          callIndirectDispatchCache[funcAddr] = funcOp;
          ++ciDispatchCacheInstalls;
        }
        ++ciDispatchCacheMisses;
      }
    } else {
      funcOp = moduleOp.lookupSymbol<func::FuncOp>(calleeName);
    }
    if (!funcOp) {
      if (isCoverageRuntimeCallee(calleeName)) {
        processStates[procId].sawUnhandledCoverageRuntimeCall = true;
        return callIndirectOp.emitError()
               << "unhandled coverage runtime call in interpreter: "
               << calleeName;
      }
      if (auto wrappedCoverageCallee =
              findDirectCoverageRuntimeCalleeInSymbol(calleeName)) {
        processStates[procId].sawUnhandledCoverageRuntimeCall = true;
        return callIndirectOp.emitError()
               << "unhandled coverage runtime call in interpreter: "
               << *wrappedCoverageCallee;
      }
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: function '" << calleeName
                              << "' not found\n");
      for (Value result : callIndirectOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
      return success();
    }


    // Gather argument values (use getArgOperands to get just the arguments, not callee)
    SmallVector<InterpretedValue, 4> args;
    for (Value arg : callIndirectOp.getArgOperands()) {
      args.push_back(getValue(procId, arg));
    }

    auto writePointerToOutRef = [&](Value outRef, uint64_t ptrValue) -> bool {
      InterpretedValue refAddr = getValue(procId, outRef);
      if (refAddr.isX())
        return false;
      uint64_t addr = refAddr.getUInt64();
      if (!addr)
        return false;

      uint64_t offset = 0;
      MemoryBlock *refBlock = findMemoryBlockByAddress(addr, procId, &offset);
      if (refBlock && offset + 8 <= refBlock->size) {
        for (unsigned i = 0; i < 8; ++i)
          refBlock->bytes()[offset + i] =
              static_cast<uint8_t>((ptrValue >> (i * 8)) & 0xFF);
        refBlock->initialized = true;
        return true;
      }

      uint64_t nativeOffset = 0;
      size_t nativeSize = 0;
      if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
          nativeOffset + 8 <= nativeSize) {
        std::memcpy(reinterpret_cast<void *>(addr), &ptrValue, 8);
        return true;
      }
      return false;
    };

    auto waitOnHopperDataForCallIndirect = [&](uint64_t hopperAddr) {
      auto &pState = processStates[procId];
      pState.waiting = true;
      pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
      auto &waiters = phaseHopperWaiters[hopperAddr];
      if (std::find(waiters.begin(), waiters.end(), procId) == waiters.end())
        waiters.push_back(procId);
    };

    auto wakeHopperWaiters = [&](uint64_t hopperAddr) {
      auto waitIt = phaseHopperWaiters.find(hopperAddr);
      if (waitIt == phaseHopperWaiters.end())
        return;
      auto waiters = waitIt->second;
      phaseHopperWaiters.erase(waitIt);
      for (ProcessId waiterProc : waiters) {
        auto stateIt = processStates.find(waiterProc);
        if (stateIt == processStates.end())
          continue;
        auto &waiterState = stateIt->second;
        bool shouldWake = waiterState.waiting ||
                          waiterState.sequencerGetRetryCallOp ||
                          !waiterState.callStack.empty();
        if (!shouldWake)
          continue;
        waiterState.waiting = false;
        scheduler.scheduleProcess(waiterProc, SchedulingRegion::Active);
      }
    };
    auto dropHopperObjection = [&](uint64_t hopperAddr) {
      if (hopperAddr == 0)
        return;
      auto handleIt = phaseObjectionHandles.find(hopperAddr);
      if (handleIt == phaseObjectionHandles.end())
        return;
      dropPhaseObjection(handleIt->second, 1);
    };

    static bool disablePhaseHopperFastPath = []() {
      const char *env = std::getenv("CIRCT_SIM_DISABLE_PHASE_HOPPER_FASTPATH");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    const bool traceUvmObjection =
        std::getenv("CIRCT_SIM_TRACE_UVM_OBJECTION") != nullptr;

    // Intercept get_objection on call_indirect dispatch.
    // The func.call interceptor already returns synthetic objection objects
    // backed by native handles. Without the call_indirect equivalent, UVM
    // wait_for() can observe a non-synthetic object and return immediately.
    if ((calleeName.contains("uvm_phase::get_objection") ||
         calleeName.contains("phase_hopper::get_objection")) &&
        !calleeName.contains("get_objection_count") &&
        !calleeName.contains("get_objection_total") &&
        callIndirectOp.getNumResults() >= 1) {
      if (args.empty() || args[0].isX()) {
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(llvm::APInt(64, 0)));
        if (traceUvmObjection) {
          llvm::errs() << "[UVM-OBJ] proc=" << procId
                       << " callee=" << calleeName
                       << " phase=<x> handle=<invalid> synthetic=0x0\n";
        }
        return success();
      }

      uint64_t phaseAddr = normalizeUvmObjectKey(procId, args[0].getUInt64());
      if (phaseAddr == 0)
        phaseAddr = args[0].getUInt64();

      MooreObjectionHandle handle = MOORE_OBJECTION_INVALID_HANDLE;
      auto it = phaseObjectionHandles.find(phaseAddr);
      if (it != phaseObjectionHandles.end()) {
        handle = it->second;
      } else {
        std::string phaseName = "phase_" + std::to_string(phaseAddr);
        handle = __moore_objection_create(
            phaseName.c_str(), static_cast<int64_t>(phaseName.size()));
        phaseObjectionHandles[phaseAddr] = handle;
      }

      uint64_t syntheticAddr =
          0xE0000000ULL + static_cast<uint64_t>(handle);
      setValue(procId, callIndirectOp.getResult(0),
               InterpretedValue(llvm::APInt(64, syntheticAddr)));
      if (traceUvmObjection) {
        llvm::errs() << "[UVM-OBJ] proc=" << procId
                     << " callee=" << calleeName << " phase=0x"
                     << llvm::format_hex(phaseAddr, 16)
                     << " handle=" << handle << " synthetic=0x"
                     << llvm::format_hex(syntheticAddr, 16) << "\n";
      }
      return success();
    }

    // Native queue fast path for phase hopper calls dispatched via vtable.
    if (!disablePhaseHopperFastPath &&
        calleeName.ends_with("uvm_phase_hopper::try_put") &&
        args.size() >= 2 && callIndirectOp.getNumResults() >= 1) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t phaseAddr = args[1].isX() ? 0 : args[1].getUInt64();
      phaseHopperQueue[hopperAddr].push_back(phaseAddr);
      wakeHopperWaiters(hopperAddr);

      if (hopperAddr != 0) {
        auto it = phaseObjectionHandles.find(hopperAddr);
        int64_t handle = 0;
        if (it != phaseObjectionHandles.end()) {
          handle = it->second;
        } else {
          std::string hopperName = "phase_hopper_" + std::to_string(hopperAddr);
          handle = __moore_objection_create(
              hopperName.c_str(), static_cast<int64_t>(hopperName.size()));
          phaseObjectionHandles[hopperAddr] = handle;
        }
        int64_t beforeCount = __moore_objection_get_count(handle);
        raisePhaseObjection(handle, 1);
        if (traceUvmObjection) {
          int64_t afterCount = __moore_objection_get_count(handle);
          llvm::errs() << "[UVM-OBJ] proc=" << procId
                       << " callee=" << calleeName << " hopper=0x"
                       << llvm::format_hex(hopperAddr, 16) << " phase=0x"
                       << llvm::format_hex(phaseAddr, 16) << " handle="
                       << handle << " before=" << beforeCount
                       << " after=" << afterCount << "\n";
        }
      }

      unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
      setValue(procId, callIndirectOp.getResult(0),
               InterpretedValue(llvm::APInt(width, 1)));
      return success();
    }

    if (!disablePhaseHopperFastPath &&
        calleeName.ends_with("uvm_phase_hopper::try_get") &&
        args.size() >= 2 && callIndirectOp.getNumResults() >= 1) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t phaseAddr = 0;
      bool hasPhase = false;
      auto it = phaseHopperQueue.find(hopperAddr);
      if (it != phaseHopperQueue.end() && !it->second.empty()) {
        phaseAddr = it->second.front();
        hasPhase = true;
      }
      if (writePointerToOutRef(callIndirectOp.getArgOperands()[1], phaseAddr)) {
        if (hasPhase) {
          it->second.pop_front();
          dropHopperObjection(hopperAddr);
        }
        unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(llvm::APInt(width, hasPhase ? 1 : 0)));
        return success();
      }
    }

    if (!disablePhaseHopperFastPath &&
        calleeName.ends_with("uvm_phase_hopper::try_peek") &&
        args.size() >= 2 && callIndirectOp.getNumResults() >= 1) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t phaseAddr = 0;
      bool hasPhase = false;
      auto it = phaseHopperQueue.find(hopperAddr);
      if (it != phaseHopperQueue.end() && !it->second.empty()) {
        phaseAddr = it->second.front();
        hasPhase = true;
      }
      if (writePointerToOutRef(callIndirectOp.getArgOperands()[1], phaseAddr)) {
        unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(llvm::APInt(width, hasPhase ? 1 : 0)));
        return success();
      }
    }

    if (!disablePhaseHopperFastPath &&
        calleeName.ends_with("uvm_phase_hopper::peek") && args.size() >= 2) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      auto it = phaseHopperQueue.find(hopperAddr);
      if (it != phaseHopperQueue.end() && !it->second.empty()) {
        if (writePointerToOutRef(callIndirectOp.getArgOperands()[1],
                                 it->second.front()))
          return success();
      } else {
        waitOnHopperDataForCallIndirect(hopperAddr);
        return success();
      }
    }

    if (!disablePhaseHopperFastPath &&
        calleeName.ends_with("uvm_phase_hopper::get") && args.size() >= 2) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      auto it = phaseHopperQueue.find(hopperAddr);
      if (it != phaseHopperQueue.end() && !it->second.empty()) {
        uint64_t phaseAddr = it->second.front();
        if (writePointerToOutRef(callIndirectOp.getArgOperands()[1], phaseAddr)) {
          it->second.pop_front();
          dropHopperObjection(hopperAddr);
          return success();
        }
      } else {
        waitOnHopperDataForCallIndirect(hopperAddr);
        return success();
      }
    }

    // Intercept uvm_port_base::size() only when the port is tracked in our
    // native graph. Otherwise, fall through to the UVM implementation so we do
    // not override valid m_imp_list bookkeeping for untracked ports.
    if (enableUvmAnalysisNativeInterceptors &&
        calleeName.contains("uvm_port_base") &&
        calleeName.ends_with("::size") &&
        callIndirectOp.getNumResults() >= 1) {
      uint64_t rawSelfAddr =
          (!args.empty() && !args[0].isX()) ? args[0].getUInt64() : 0;
      uint64_t selfAddr = canonicalizeUvmObjectAddress(procId, rawSelfAddr);
      if (analysisPortConnections.count(selfAddr)) {
        int32_t count = getNativeUvmPortSize(analysisPortConnections, selfAddr);
        Value result = callIndirectOp.getResult(0);
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result,
                 InterpretedValue(
                     llvm::APInt(width, static_cast<uint64_t>(count), false)));
        LLVM_DEBUG(
            llvm::dbgs() << "  call_indirect: uvm_port_base::size self=0x"
                         << llvm::format_hex(selfAddr, 16) << " -> " << count
                         << "\n");
        return success();
      }
    }

    // Intercept resource_db_implementation via call_indirect (vtable dispatch).
    // resource_db#(T) is parametric, so each specialization has its own vtable
    // and calls go through call_indirect, not func.call.
    if (calleeName.contains("resource_db") && calleeName.contains("implementation") &&
        (calleeName.contains("::set") || calleeName.contains("::read_by_name"))) {

      auto readStr = [&](unsigned argIdx) -> std::string {
        if (argIdx >= args.size())
          return "";
        return readMooreStringStruct(procId, args[argIdx]);
      };

      if (calleeName.contains("::set") && !calleeName.contains("set_default") &&
          !calleeName.contains("set_override") &&
          !calleeName.contains("set_anonymous")) {
        // set(self, scope, field_name, value, ...)
        if (args.size() >= 4) {
          std::string scope = readStr(1);
          std::string fieldName = readStr(2);
          std::string key = scope + "." + fieldName;
          InterpretedValue &valueArg = args[3];
          unsigned valueBits = valueArg.getWidth();
          bool truncatedValue = false;
          std::vector<uint8_t> valueData =
              serializeInterpretedValueBytes(valueArg, /*maxBytes=*/1ULL << 20,
                                             &truncatedValue);
          unsigned valueBytes = static_cast<unsigned>(valueData.size());
          configDbEntries[key] = std::move(valueData);
          if (traceConfigDbEnabled && truncatedValue) {
            llvm::errs() << "[RSRC-CI-SET] truncated oversized value payload key=\""
                         << key << "\" bitWidth=" << valueBits << "\n";
          }
        }
        return success();
      }

      if (calleeName.contains("::read_by_name")) {
        // read_by_name(self, scope, field_name, output_ref, ...) -> i1
        if (args.size() >= 4 && callIndirectOp.getNumResults() >= 1) {
          std::string scope = readStr(1);
          std::string fieldName = readStr(2);
          std::string key = scope + "." + fieldName;

          auto it = configDbEntries.find(key);
          if (it == configDbEntries.end()) {
            for (auto &[k, v] : configDbEntries) {
              size_t dotPos = k.rfind('.');
              if (dotPos != std::string::npos &&
                  k.substr(dotPos + 1) == fieldName) {
                it = configDbEntries.find(k);
                break;
              }
            }
          }

          if (it != configDbEntries.end()) {
            // Write value to output_ref (arg 3)
            Value outputRef = callIndirectOp.getArgOperands()[3];
            const std::vector<uint8_t> &valueData = it->second;
            Type refType = outputRef.getType();

            if (auto refT = dyn_cast<llhd::RefType>(refType)) {
              Type innerType = refT.getNestedType();
              unsigned innerBits = getTypeWidth(innerType);
              unsigned innerBytes = (innerBits + 7) / 8;
              llvm::APInt valueBits(innerBits, 0);
              for (unsigned i = 0;
                   i < std::min(innerBytes, (unsigned)valueData.size()); ++i)
                safeInsertBits(valueBits,llvm::APInt(8, valueData[i]), i * 8);
              SignalId sigId2 = resolveSignalId(outputRef);
              if (sigId2 != 0)
                pendingEpsilonDrives[sigId2] = InterpretedValue(valueBits);
              // Also write directly to memory so that llvm.load can read it.
              // The ref value holds the address of the backing memory.
              InterpretedValue refAddr = getValue(procId, outputRef);
          if (!refAddr.isX()) {
            uint64_t addr = refAddr.getUInt64();
            uint64_t off3 = 0;
            MemoryBlock *blk =
                findMemoryBlockByAddress(addr, procId, &off3);
            if (!blk)
              blk = findBlockByAddress(addr, off3);
            if (blk) {
                  writeConfigDbBytesToMemoryBlock(
                      blk, off3, valueData, innerBytes,
                      /*zeroFillMissing=*/true);
                } else {
                  uint64_t nativeOff = 0;
                  size_t nativeSize = 0;
                  if (findNativeMemoryBlockByAddress(addr, &nativeOff,
                                                     &nativeSize)) {
                    writeConfigDbBytesToNativeMemory(
                        addr, nativeOff, nativeSize, valueData, innerBytes,
                        /*zeroFillMissing=*/true);
                  }
                }
              }
            } else if (isa<LLVM::LLVMPointerType>(refType)) {
              // Pointer output: write directly to memory
              if (!args[3].isX()) {
                uint64_t outputAddr = args[3].getUInt64();
                uint64_t outOff = 0;
                MemoryBlock *outBlock =
                    findMemoryBlockByAddress(outputAddr, procId, &outOff);
                if (!outBlock)
                  outBlock = findBlockByAddress(outputAddr, outOff);
                if (outBlock) {
                  writeConfigDbBytesToMemoryBlock(
                      outBlock, outOff, valueData,
                      static_cast<unsigned>(valueData.size()),
                      /*zeroFillMissing=*/false);
                } else {
                  uint64_t nativeOff = 0;
                  size_t nativeSize = 0;
                  if (findNativeMemoryBlockByAddress(outputAddr, &nativeOff,
                                                     &nativeSize)) {
                    writeConfigDbBytesToNativeMemory(
                        outputAddr, nativeOff, nativeSize, valueData,
                        static_cast<unsigned>(valueData.size()),
                        /*zeroFillMissing=*/false);
                  }
                }
              }
            }

            setValue(procId, callIndirectOp.getResult(0),
                    InterpretedValue(llvm::APInt(1, 1)));
            return success();
          }

          setValue(procId, callIndirectOp.getResult(0),
                  InterpretedValue(llvm::APInt(1, 0)));
          return success();
        }
      }
    }

    if (tryInterceptConfigDbCallIndirect(procId, callIndirectOp, calleeName,
                                         args))
      return success();

    // Intercept UVM port connect() via call_indirect.
    // Record port→provider connections in the native map, but still execute the
    // original UVM connect() implementation so m_provided_by/m_provided_to and
    // resolved m_if pointers are populated for regular TLM port operations.
    // The native map remains useful for analysis fast paths and sequencer
    // rendezvous fallbacks.
    auto isNativeConnectCallee = [&](llvm::StringRef name) {
      if (!name.contains("::connect"))
        return false;
      return name.contains("uvm_port_base") ||
             name.contains("uvm_analysis_port") ||
             name.contains("uvm_analysis_export") ||
             name.contains("uvm_analysis_imp") ||
             name.contains("uvm_seq_item_pull_") ||
             (name.contains("uvm_tlm_") &&
              (name.contains("_port") || name.contains("_export") ||
               name.contains("_imp")));
    };
    if (isNativeConnectCallee(calleeName) &&
        !calleeName.contains("connect_phase") && args.size() >= 2) {
      uint64_t rawSelfAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t rawProviderAddr = args[1].isX() ? 0 : args[1].getUInt64();
      uint64_t selfAddr = canonicalizeUvmObjectAddress(procId, rawSelfAddr);
      uint64_t providerAddr =
          canonicalizeUvmObjectAddress(procId, rawProviderAddr);
      recordUvmPortConnection(procId, rawSelfAddr, rawProviderAddr);
      if (traceAnalysisEnabled)
        llvm::errs() << "[ANALYSIS-CONNECT] " << calleeName
                     << " self_raw=0x" << llvm::format_hex(rawSelfAddr, 0)
                     << " provider_raw=0x"
                     << llvm::format_hex(rawProviderAddr, 0)
                     << " self=0x" << llvm::format_hex(selfAddr, 0)
                     << " provider=0x" << llvm::format_hex(providerAddr, 0)
                     << "\n";
      // Fall through to UVM connect() for canonical bookkeeping.
    }

    // Intercept analysis write entrypoints to broadcast to connected ports.
    // When the native port_base connect() is rejected due to "Late Connection",
    // the UVM write() loop finds 0 subscribers. We use our native connection
    // map to resolve the correct imp write function via vtable dispatch.
    // Supports multi-hop chains: port → port/export → imp.
    if (enableUvmAnalysisNativeInterceptors &&
        isNativeAnalysisWriteCalleeCI(calleeName) && args.size() >= 2) {
      uint64_t rawPortAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t portAddr = canonicalizeUvmObjectAddress(procId, rawPortAddr);
      if (traceAnalysisEnabled)
        llvm::errs() << "[ANALYSIS-WRITE] " << calleeName
                     << " portAddr=0x" << llvm::format_hex(portAddr, 0)
                     << " inMap=" << analysisPortConnections.count(portAddr)
                     << "\n";
      // Flatten the connection chain to find all terminal imps.
      // A terminal is any address that doesn't appear as a key in our map.
      llvm::SmallVector<uint64_t, 4> terminals;
      llvm::SmallVector<uint64_t, 8> worklist;
      llvm::DenseSet<uint64_t> visited;
      seedAnalysisPortConnectionWorklist(procId, portAddr, worklist);
      while (!worklist.empty()) {
        uint64_t addr = worklist.pop_back_val();
        if (!visited.insert(addr).second)
          continue;
        auto chainIt = analysisPortConnections.find(addr);
        if (chainIt != analysisPortConnections.end() && !chainIt->second.empty()) {
          // This is an intermediate port/export — follow its connections.
          for (uint64_t next : chainIt->second)
            worklist.push_back(next);
        } else {
          // Terminal (imp or unconnected export).
          terminals.push_back(addr);
        }
      }
      if (traceAnalysisEnabled && terminals.empty())
        llvm::errs() << "[ANALYSIS-WRITE] NO terminals found for portAddr=0x"
                     << llvm::format_hex(portAddr, 0) << "\n";
      if (!terminals.empty()) {
        if (traceAnalysisEnabled)
          llvm::errs() << "[ANALYSIS-WRITE] " << terminals.size()
                       << " terminal(s) found\n";
        for (uint64_t impAddr : terminals) {
          auto writeTarget = tryResolveAnalysisWriteTarget(
              moduleOp, impAddr, "ANALYSIS-WRITE");
          if (!writeTarget)
            continue;
          dispatchAnalysisWriteTarget(writeTarget->first, writeTarget->second,
                                      "ANALYSIS-WRITE", impAddr, args[1]);
        }
        return success();
      }
      if (calleeName.contains("uvm_tlm_if_base")) {
        if (tryDispatchAnalysisWriteSelfFallback(
                moduleOp, "ANALYSIS-WRITE", calleeName, portAddr, args[1]))
          return success();
        if (traceAnalysisEnabled)
          llvm::errs() << "[ANALYSIS-WRITE] NO terminals for tlm_if_base "
                       << "self=0x" << llvm::format_hex(portAddr, 0)
                       << " (no-op)\n";
        return success();
      }
      // If no native connections, fall through to normal execution
      // (which will iterate the UVM m_imp_list -- may be empty).
    }

    // Intercept UVM sequencer interface: start_item, finish_item, and
    // seq_item_pull_port::get to implement a native rendezvous between
    // sequence producer and driver consumer. This bypasses the complex
    // UVM sequencer arbitration/FIFO machinery (m_safe_select_item,
    // wait_for_grant, send_request, wait_for_item_done) that requires
    // fully functional TLM FIFOs and process synchronization.
    if (traceSeq &&
        (calleeName.contains("::start_item") ||
         calleeName.contains("::finish_item") ||
         calleeName.contains("::wait_for_grant") ||
         calleeName.contains("::send_request") ||
         calleeName.contains("::wait_for_item_done") ||
         ((calleeName.contains("seq_item_pull_port") ||
           calleeName.contains("seq_item_pull_imp") ||
           calleeName.contains("sqr_if_base") ||
         calleeName.contains("uvm_sequencer")) &&
          (calleeName.ends_with("::get") ||
           calleeName.ends_with("::get_next_item") ||
           calleeName.ends_with("::item_done"))))) {
      uint64_t a0 = args.size() > 0 && !args[0].isX() ? args[0].getUInt64() : 0;
      uint64_t a1 = args.size() > 1 && !args[1].isX() ? args[1].getUInt64() : 0;
      llvm::errs() << "[SEQ-CI] " << calleeName << " a0=0x"
                   << llvm::format_hex(a0, 16) << " a1=0x"
                   << llvm::format_hex(a1, 16)
                   << " fifo_maps=" << sequencerItemFifo.size() << "\n";
    }

    // Intercept low-level sequencer handshake used by many AVIP/UVM benches:
    // wait_for_grant -> send_request -> wait_for_item_done.
    if ((calleeName.contains("uvm_sequencer") ||
         calleeName.contains("sqr_if_base")) &&
        (calleeName.ends_with("::wait_for_grant") ||
         calleeName.ends_with("::send_request") ||
         calleeName.ends_with("::wait_for_item_done"))) {
      if (calleeName.ends_with("::wait_for_grant")) {
        if (!args.empty() && !args[0].isX()) {
          uint64_t sqrAddr =
              normalizeUvmSequencerAddress(procId, args[0].getUInt64());
          if (sqrAddr != 0)
            itemToSequencer[sequencerProcKey(procId)] = sqrAddr;
        }
        // Native path grants immediately.
        return success();
      }

      if (calleeName.ends_with("::send_request") && args.size() >= 3) {
        uint64_t sqrAddr = args[0].isX()
                               ? 0
                               : normalizeUvmSequencerAddress(
                                     procId, args[0].getUInt64());
        uint64_t seqAddr = args[1].isX()
                               ? 0
                               : normalizeUvmObjectKey(procId,
                                                       args[1].getUInt64());
        uint64_t itemAddr = args[2].isX() ? 0 : args[2].getUInt64();
        uint64_t queueAddr = 0;
        if (itemAddr != 0) {
          if (auto ownerIt = itemToSequencer.find(itemAddr);
              ownerIt != itemToSequencer.end())
            queueAddr = ownerIt->second;
        }
        if (queueAddr == 0) {
          if (auto procIt = itemToSequencer.find(sequencerProcKey(procId));
              procIt != itemToSequencer.end())
            queueAddr = procIt->second;
        }
        if (queueAddr == 0)
          queueAddr = sqrAddr;
        queueAddr = normalizeUvmSequencerAddress(procId, queueAddr);
        if (itemAddr != 0 && seqAddr != 0)
          recordUvmSequenceItemOwner(itemAddr, seqAddr);
        if (itemAddr != 0 && queueAddr != 0) {
          sequencerItemFifo[queueAddr].push_back(itemAddr);
          recordUvmSequencerItemOwner(itemAddr, queueAddr);
          sequencePendingItemsByProc[procId].push_back(itemAddr);
          if (seqAddr != 0)
            sequencePendingItemsBySeq[seqAddr].push_back(itemAddr);
          wakeUvmSequencerGetWaiterForPush(queueAddr);
          if (traceSeq) {
            llvm::errs() << "[SEQ-CI] send_request item=0x"
                         << llvm::format_hex(itemAddr, 16) << " sqr=0x"
                         << llvm::format_hex(queueAddr, 16) << " depth="
                         << sequencerItemFifo[queueAddr].size() << "\n";
          }
        }
        return success();
      }

      if (calleeName.ends_with("::wait_for_item_done")) {
        uint64_t seqAddr = args.size() > 1 && !args[1].isX()
                               ? normalizeUvmObjectKey(procId,
                                                       args[1].getUInt64())
                               : 0;

        uint64_t itemAddr = 0;
        auto seqIt = sequencePendingItemsBySeq.end();
        if (seqAddr != 0) {
          seqIt = sequencePendingItemsBySeq.find(seqAddr);
          if (seqIt != sequencePendingItemsBySeq.end() &&
              !seqIt->second.empty())
            itemAddr = seqIt->second.front();
        }
        auto procIt = sequencePendingItemsByProc.find(procId);
        if (itemAddr == 0 && procIt != sequencePendingItemsByProc.end() &&
            !procIt->second.empty())
          itemAddr = procIt->second.front();
        if (itemAddr == 0) {
          if (traceSeq) {
            llvm::errs() << "[SEQ-CI] wait_for_item_done miss proc=" << procId
                         << " seq=0x" << llvm::format_hex(seqAddr, 16)
                         << "\n";
          }
          return success();
        }

        auto erasePendingItemByProc = [&](uint64_t item) {
          auto it = sequencePendingItemsByProc.find(procId);
          if (it == sequencePendingItemsByProc.end())
            return;
          auto &pending = it->second;
          auto match = std::find(pending.begin(), pending.end(), item);
          if (match != pending.end())
            pending.erase(match);
          if (pending.empty())
            sequencePendingItemsByProc.erase(it);
        };
        auto erasePendingItemBySeq = [&](uint64_t seqKey, uint64_t item) {
          if (seqKey == 0)
            return;
          auto it = sequencePendingItemsBySeq.find(seqKey);
          if (it == sequencePendingItemsBySeq.end())
            return;
          auto &pending = it->second;
          auto match = std::find(pending.begin(), pending.end(), item);
          if (match != pending.end())
            pending.erase(match);
          if (pending.empty())
            sequencePendingItemsBySeq.erase(it);
        };

        if (itemDoneReceived.count(itemAddr)) {
          erasePendingItemByProc(itemAddr);
          erasePendingItemBySeq(seqAddr, itemAddr);
          itemDoneReceived.erase(itemAddr);
          finishItemWaiters.erase(itemAddr);
          (void)takeUvmSequencerItemOwner(itemAddr);
          (void)takeUvmSequenceItemOwner(itemAddr);
          return success();
        }

        finishItemWaiters[itemAddr] = procId;
        auto &pState = processStates[procId];
        pState.waiting = true;
        pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
        if (traceSeq)
          llvm::errs() << "[SEQ-CI] wait_for_item_done item=0x"
                       << llvm::format_hex(itemAddr, 16) << "\n";
        return success();
      }
    }

    // start_item: Record item→sequencer mapping and return immediately
    // (grants arbitration instantly). Args: (self, item, priority, sequencer).
    if (calleeName.contains("::start_item") && args.size() >= 4) {
      uint64_t seqAddr = args[0].isX()
                             ? 0
                             : normalizeUvmObjectKey(procId,
                                                     args[0].getUInt64());
      uint64_t itemAddr = args[1].isX() ? 0 : args[1].getUInt64();
      uint64_t sqrAddr = args[3].isX()
                             ? 0
                             : normalizeUvmSequencerAddress(procId,
                                                            args[3].getUInt64());
      InterpretedValue sqrArg = args[3];
      // If sequencer arg is null, get it from the sequence object.
      // The sequence's m_sequencer is set by seq.start(sqr).
      if (seqAddr != 0) {
        auto getSequencerFunc = moduleOp.lookupSymbol<func::FuncOp>(
            "uvm_pkg::uvm_sequence_item::get_sequencer");
        if (getSequencerFunc) {
          SmallVector<InterpretedValue, 1> getSeqArgs;
          getSeqArgs.push_back(args[0]); // self sequence
          SmallVector<InterpretedValue, 1> getSeqResults;
          auto &cState = processStates[procId];
          ++cState.callDepth;
          auto getSeqResult = interpretFuncBody(procId, getSequencerFunc,
                                                getSeqArgs, getSeqResults,
                                                callIndirectOp);
          --cState.callDepth;
          if (succeeded(getSeqResult) && !getSeqResults.empty() &&
              !getSeqResults[0].isX()) {
            sqrAddr = normalizeUvmSequencerAddress(procId,
                                                   getSeqResults[0].getUInt64());
            sqrArg = InterpretedValue(llvm::APInt(64, sqrAddr));
          }
        }
      }
      if (itemAddr != 0 && sqrAddr != 0) {
        uint64_t queueAddr = normalizeUvmSequencerAddress(procId, sqrAddr);
        itemToSequencer[sequencerProcKey(procId)] = queueAddr;
        recordUvmSequencerItemOwner(itemAddr, queueAddr);
        recordUvmSequenceItemOwner(itemAddr, seqAddr);
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: start_item intercepted: item 0x"
                   << llvm::format_hex(itemAddr, 16) << " → sequencer 0x"
                   << llvm::format_hex(queueAddr, 16)
                   << " (raw=0x" << llvm::format_hex(sqrAddr, 16) << ")\n");
      }
      // Call set_item_context to set up the item's parent sequence and
      // sequencer references, then return (skip wait_for_grant).
      auto setContextFunc =
          moduleOp.lookupSymbol<func::FuncOp>(
              "uvm_pkg::uvm_sequence_item::set_item_context");
      if (setContextFunc && itemAddr != 0 && seqAddr != 0) {
        SmallVector<InterpretedValue, 3> ctxArgs;
        ctxArgs.push_back(args[1]); // item
        ctxArgs.push_back(args[0]); // parent sequence
        // For the sequencer arg, use sqrAddr if available, else null.
        ctxArgs.push_back(sqrArg);
        SmallVector<InterpretedValue, 1> ctxResults;
        auto &cState = processStates[procId];
        ++cState.callDepth;
        (void)interpretFuncBody(procId, setContextFunc, ctxArgs, ctxResults,
                               callIndirectOp);
        --cState.callDepth;
      }
      return success();
    }

    // finish_item: Push item to sequencer FIFO and block until the driver
    // calls item_done for this item. This implements the standard UVM
    // handshake where finish_item blocks until the driver completes.
    // Args: (self, item, priority).
    if (calleeName.contains("::finish_item") && args.size() >= 2) {
      // [SEQ-DIRECT] finish_item diagnostic removed
      uint64_t seqAddr = args[0].isX()
                             ? 0
                             : normalizeUvmObjectKey(procId,
                                                     args[0].getUInt64());
      uint64_t itemAddr = args[1].isX() ? 0 : args[1].getUInt64();
      if (itemAddr != 0) {
        recordUvmSequenceItemOwner(itemAddr, seqAddr);
        // Check if item_done was already received (re-poll after wake)
        if (itemDoneReceived.count(itemAddr)) {
          (void)takeUvmSequencerItemOwner(itemAddr);
          (void)takeUvmSequenceItemOwner(itemAddr);
          itemDoneReceived.erase(itemAddr);
          finishItemWaiters.erase(itemAddr);
          LLVM_DEBUG(llvm::dbgs()
                     << "  call_indirect: finish_item completed: item 0x"
                     << llvm::format_hex(itemAddr, 16) << " got item_done\n");
          return success();
        }

        // First call (not a re-poll): push item to FIFO
        if (!finishItemWaiters.count(itemAddr)) {
          uint64_t sqrAddr = takeUvmSequencerItemOwner(itemAddr);
          if (sqrAddr == 0) {
            if (auto procIt = itemToSequencer.find(sequencerProcKey(procId));
                procIt != itemToSequencer.end())
              sqrAddr = procIt->second;
          }
          uint64_t queueAddr = normalizeUvmSequencerAddress(procId, sqrAddr);
          if (queueAddr == 0)
            return success();
          sequencerItemFifo[queueAddr].push_back(itemAddr);
          LLVM_DEBUG(llvm::dbgs()
                     << "  call_indirect: finish_item intercepted: item 0x"
                     << llvm::format_hex(itemAddr, 16) << " pushed to "
                     << "sequencer FIFO 0x"
                     << llvm::format_hex(queueAddr, 16) << " (raw=0x"
                     << llvm::format_hex(sqrAddr, 16) << ", depth "
                     << sequencerItemFifo[queueAddr].size() << ")\n");
          if (traceSeq) {
            llvm::errs() << "[SEQ-CI] push item=0x"
                         << llvm::format_hex(itemAddr, 16) << " sqr=0x"
                         << llvm::format_hex(queueAddr, 16) << " depth="
                         << sequencerItemFifo[queueAddr].size() << "\n";
          }
          // Wake any process blocked in get/get_next_item on this queue.
          wakeUvmSequencerGetWaiterForPush(queueAddr);
        }

        // Record waiter and suspend. The item_done interceptor will
        // directly resume this process when the driver completes.
        finishItemWaiters[itemAddr] = procId;
        auto &pState = processStates[procId];
        pState.waiting = true;
        pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: finish_item blocking proc=" << procId
                   << " on item 0x" << llvm::format_hex(itemAddr, 16)
                   << "\n");
        return success();
      }
      return success();
    }

    // seq_item_pull_port/imp::get/get_next_item: Pull item from sequencer FIFO.
    // The port is connected to sequencer's seq_item_export via our
    // analysisPortConnections map. We follow the connection chain to
    // find the sequencer, then pull from its FIFO.
    // get_next_item is functionally identical to get for pull ports.
    // Args: (self_port, output_ref).
    if ((calleeName.contains("seq_item_pull_port") ||
         calleeName.contains("seq_item_pull_imp") ||
         calleeName.contains("sqr_if_base") ||
         calleeName.contains("uvm_sequencer")) &&
        (calleeName.ends_with("::get") ||
         calleeName.ends_with("::get_next_item") ||
         calleeName.ends_with("::try_next_item")) &&
        args.size() >= 2) {
      uint64_t portAddr = args[0].isX() ? 0 : args[0].getUInt64();
      bool isTryNextItem = calleeName.ends_with("::try_next_item");
      uint64_t seqrQueueAddr = 0;
      bool resolvedSeqrQueueHint = resolveUvmSequencerQueueAddress(
          procId, portAddr, callIndirectOp.getOperation(), seqrQueueAddr);

      // Cache only explicit routing hints (cache hit or resolved connection
      // chain). Do not cache opportunistic fallback choices.
      if (resolvedSeqrQueueHint && seqrQueueAddr != 0)
        cacheUvmSequencerQueueAddress(portAddr, seqrQueueAddr);

      // Try to find an item for the resolved queue first.
      uint64_t itemAddr = 0;
      bool found = false;
      bool fromFallbackSearch = false;

      if (seqrQueueAddr != 0) {
        auto fifoIt = sequencerItemFifo.find(seqrQueueAddr);
        if (fifoIt != sequencerItemFifo.end() && !fifoIt->second.empty()) {
          itemAddr = fifoIt->second.front();
          fifoIt->second.pop_front();
          found = true;
        }
      }
      bool allowGlobalFallbackSearch = (seqrQueueAddr == 0);
      bool allowSingleQueueFallback =
          allowGlobalFallbackSearch && !isTryNextItem &&
          sequencerItemFifo.size() == 1;
      if (!found && allowSingleQueueFallback) {
        auto it = sequencerItemFifo.begin();
        if (it != sequencerItemFifo.end() && !it->second.empty()) {
          seqrQueueAddr = it->first;
          itemAddr = it->second.front();
          it->second.pop_front();
          found = true;
          fromFallbackSearch = true;
        }
      }
      if (found && itemAddr != 0) {
        // Track the dequeued item by both pull-port and resolved queue alias.
        recordUvmDequeuedItem(procId, portAddr, seqrQueueAddr, itemAddr);
        // Write item address to output ref (args[1]).
        // The output ref is an llhd.ref or alloca-backed ptr.
        uint64_t refAddr = args[1].isX() ? 0 : args[1].getUInt64();
        LLVM_DEBUG(llvm::dbgs()
                   << "  seq_item_pull_port::get: item found 0x"
                   << llvm::format_hex(itemAddr, 16) << " → ref 0x"
                   << llvm::format_hex(refAddr, 16) << "\n");
        if (traceSeq) {
          llvm::errs() << "[SEQ-CI] pop item=0x"
                       << llvm::format_hex(itemAddr, 16) << " port=0x"
                       << llvm::format_hex(portAddr, 16) << " seqr_hint=0x"
                       << llvm::format_hex(seqrQueueAddr, 16)
                       << " seqr_q=0x" << llvm::format_hex(seqrQueueAddr, 16)
                       << " fallback=" << (fromFallbackSearch ? 1 : 0)
                       << "\n";
        }
        if (refAddr != 0) {
          uint64_t offset = 0;
          MemoryBlock *refBlock =
              findMemoryBlockByAddress(refAddr, procId, &offset);
          if (refBlock &&
              offset + 8 <= refBlock->size) {
            for (unsigned i = 0; i < 8; ++i)
              refBlock->bytes()[offset + i] =
                  static_cast<uint8_t>((itemAddr >> (i * 8)) & 0xFF);
            refBlock->initialized = true;
          } else {
            uint64_t nativeOffset = 0;
            size_t nativeSize = 0;
            if (findNativeMemoryBlockByAddress(refAddr, &nativeOffset,
                                               &nativeSize) &&
                nativeOffset + 8 <= nativeSize) {
              std::memcpy(reinterpret_cast<void *>(refAddr), &itemAddr, 8);
            }
          }
        }
        return success();
      }
      if (isTryNextItem) {
        uint64_t refAddr = args[1].isX() ? 0 : args[1].getUInt64();
        if (traceSeq)
          llvm::errs() << "[SEQ-CI] try_next_item miss port=0x"
                       << llvm::format_hex(portAddr, 16) << " ref=0x"
                       << llvm::format_hex(refAddr, 16) << "\n";
        if (refAddr != 0) {
          uint64_t offset = 0;
          MemoryBlock *refBlock =
              findMemoryBlockByAddress(refAddr, procId, &offset);
          if (refBlock &&
              offset + 8 <= refBlock->size) {
            for (unsigned i = 0; i < 8; ++i)
              refBlock->bytes()[offset + i] = 0;
            refBlock->initialized = true;
          } else {
            uint64_t nativeOffset = 0;
            size_t nativeSize = 0;
            if (findNativeMemoryBlockByAddress(refAddr, &nativeOffset,
                                               &nativeSize) &&
                nativeOffset + 8 <= nativeSize) {
              uint64_t nullItem = 0;
              std::memcpy(reinterpret_cast<void *>(refAddr), &nullItem, 8);
            }
          }
        }
        return success();
      }
      // If no item available, register as a queue waiter so that
      // finish_item's push will wake this process directly (event-based
      // instead of delta-cycle polling). Return success() so the halt
      // check in interpretFuncBody saves call stack frames.
      LLVM_DEBUG(llvm::dbgs() << "  seq_item_pull_port::get: FIFO empty, "
                              << "registering as queue waiter\n");
      {
        // Only emit the trace on first registration (avoid duplicate
        // messages from re-wakeup via unrelated signal sensitivity).
        bool alreadyWaiting =
            sequencerGetWaitQueueByProc.count(procId) != 0;
        if (traceSeq && !alreadyWaiting) {
          llvm::errs() << "[SEQ-CI] wait port=0x"
                       << llvm::format_hex(portAddr, 16) << " seqr=0x"
                       << llvm::format_hex(seqrQueueAddr, 16) << "\n";
        }
      }
      auto &pState = processStates[procId];
      pState.waiting = true;
      pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
      uint64_t waitQueueAddr = allowGlobalFallbackSearch ? 0 : seqrQueueAddr;
      enqueueUvmSequencerGetWaiter(waitQueueAddr, procId,
                                   callIndirectOp.getOperation());
      return success();
    }

    // seq_item_pull_port/imp::item_done: Signal completion of the current
    // transaction item. This unblocks the sequence's finish_item wait.
    if ((calleeName.contains("seq_item_pull_port") ||
         calleeName.contains("seq_item_pull_imp") ||
         calleeName.contains("sqr_if_base") ||
         calleeName.contains("uvm_sequencer")) &&
        calleeName.ends_with("::item_done")) {
      uint64_t doneAddr =
          args.size() > 0 && !args[0].isX() ? args[0].getUInt64() : 0;
      uint64_t itemAddr = takeUvmDequeuedItemForDone(
          procId, doneAddr, callIndirectOp.getOperation());
      if (itemAddr != 0) {
        itemDoneReceived.insert(itemAddr);
        InterpretedValue responseVal =
            args.size() > 1 ? args[1] : InterpretedValue::makeX(64);
        deliverUvmSequenceResponse(procId, itemAddr, responseVal,
                                   callIndirectOp.getOperation());
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: item_done: item 0x"
                   << llvm::format_hex(itemAddr, 16)
                   << " marked done (caller 0x"
                   << llvm::format_hex(doneAddr, 16) << ")\n");

        // Directly resume the process blocked in finish_item for this item.
        auto waiterIt = finishItemWaiters.find(itemAddr);
        if (waiterIt != finishItemWaiters.end()) {
          ProcessId waiterProcId = waiterIt->second;
          finishItemWaiters.erase(waiterIt);
          auto waiterStateIt = processStates.find(waiterProcId);
          if (waiterStateIt != processStates.end()) {
            auto &waiterState = waiterStateIt->second;
            bool shouldWake = waiterState.waiting ||
                              waiterState.sequencerGetRetryCallOp ||
                              !waiterState.callStack.empty();
            if (shouldWake) {
              waiterState.waiting = false;
              scheduler.scheduleProcess(waiterProcId,
                                        SchedulingRegion::Active);
              LLVM_DEBUG(llvm::dbgs()
                         << "  item_done: resuming finish_item waiter proc="
                         << waiterProcId << "\n");
            }
          }
        }
      } else if (traceSeq) {
        llvm::errs() << "[SEQ-CI] item_done miss caller=0x"
                     << llvm::format_hex(doneAddr, 16) << "\n";
      }
      return success();
    }

    auto &callState = processStates[procId];

    // Recursive DFS depth detection (same as func.call handler)
    Operation *indFuncKey = funcOp.getOperation();
    uint64_t indArg0Val = 0;
    bool indHasArg0 = !args.empty() && !args[0].isX();
    if (indHasArg0)
      indArg0Val = args[0].getUInt64();
    constexpr unsigned maxRecursionDepth = 20;
    auto &indDepthMap = callState.recursionVisited[indFuncKey];
    if (indHasArg0 && callState.callDepth > 0) {
      unsigned &depth = indDepthMap[indArg0Val];
      if (depth >= maxRecursionDepth) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: recursion depth " << depth
                   << " exceeded for '" << calleeName << "' with arg0=0x"
                   << llvm::format_hex(indArg0Val, 16) << "\n");
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
        return success();
      }
    }
    bool indAddedToVisited = indHasArg0;
    if (indHasArg0)
      ++indDepthMap[indArg0Val];

    // Intercept UVM phase/objection methods called via vtable dispatch.
    // These bypass interpretFuncCall() so we handle them here directly.
    if (calleeName.contains("uvm_phase::raise_objection") ||
        calleeName.contains("uvm_phase::drop_objection")) {
      const bool traceUvmObjection =
          std::getenv("CIRCT_SIM_TRACE_UVM_OBJECTION") != nullptr;
      if (!calleeName.contains("phase_hopper") && !args.empty() && !args[0].isX()) {
        uint64_t phaseAddr = normalizeUvmObjectKey(procId, args[0].getUInt64());
        if (phaseAddr == 0)
          phaseAddr = args[0].getUInt64();
        InterpretedValue countVal =
            args.size() > 3 ? args[3] : InterpretedValue(llvm::APInt(32, 1));
        int64_t count = countVal.isX() ? 1 : static_cast<int64_t>(countVal.getUInt64());
        auto it2 = phaseObjectionHandles.find(phaseAddr);
        MooreObjectionHandle handle;
        if (it2 != phaseObjectionHandles.end()) {
          handle = it2->second;
        } else {
          std::string phaseName = "phase_" + std::to_string(phaseAddr);
          handle = __moore_objection_create(
              phaseName.c_str(), static_cast<int64_t>(phaseName.size()));
          phaseObjectionHandles[phaseAddr] = handle;
        }
        int64_t beforeCount = __moore_objection_get_count(handle);
        if (calleeName.contains("raise_objection")) {
          raisePhaseObjection(handle, count);
        } else {
          dropPhaseObjection(handle, count);
        }
        if (args.size() > 1 && !args[1].isX()) {
          InterpretedValue descVal =
              args.size() > 2 ? args[2] : InterpretedValue(llvm::APInt(128, 0));
          maybeDispatchUvmComponentObjectionCallback(
              procId, args[1].getUInt64(), handle,
              /*isRaise=*/calleeName.contains("raise_objection"), descVal,
              countVal, callIndirectOp.getOperation());
        }
        int64_t afterCount = __moore_objection_get_count(handle);
        if (beforeCount > 0 || afterCount > 0)
          executePhasePhaseSawPositiveObjection[phaseAddr] = true;
        if (traceUvmObjection) {
          llvm::errs() << "[UVM-OBJ] proc=" << procId
                       << " callee=" << calleeName << " phase=0x"
                       << llvm::format_hex(phaseAddr, 16)
                       << " handle=" << handle << " delta=" << count
                       << " before=" << beforeCount
                       << " after=" << afterCount << "\n";
        }
      }
      if (indAddedToVisited)
        decrementRecursionDepthEntry(callState, indFuncKey, indArg0Val);
      return success();
    }

    // E5: Populate per-call-site cache for non-intercepted calls.
    if (!callIndirectDirectDispatchCacheDisabled && funcOp) {
      auto &se = callIndirectSiteCache[callIndirectOp.getOperation()];
      bool hadOverride = !overriddenCalleeName.empty();
      if (!hadOverride) {
        se.funcAddr = funcAddr;
        se.funcOp = funcOp;
        se.valid = true;
        se.isIntercepted = false;
        se.hadVtableOverride = false;
        // Populate entry-table pointer for site cache dispatch.
        if (compiledFuncEntries && funcAddr >= 0xF0000000ULL &&
            funcAddr < 0x100000000ULL) {
          uint32_t fid = static_cast<uint32_t>(funcAddr - 0xF0000000ULL);
          if (fid < numCompiledAllFuncs && compiledFuncEntries[fid]) {
            bool isNativeEntry =
                (fid < compiledFuncIsNative.size() && compiledFuncIsNative[fid]);
            bool hasTrampolineEntry =
                (fid < compiledFuncHasTrampoline.size() &&
                 compiledFuncHasTrampoline[fid]);
            if (isNativeEntry || hasTrampolineEntry) {
              se.cachedFid = fid;
              se.cachedEntryPtr = const_cast<void *>(compiledFuncEntries[fid]);
            }
          }
        }
      } else {
        se.valid = false;
        se.hadVtableOverride = true;
      }
    }

    // Entry-table dispatch: try compiled dispatch (native + trampoline) before
    // interpretFuncBody.
    if (compiledFuncEntries && funcAddr >= 0xF0000000ULL &&
        funcAddr < 0x100000000ULL && callState.callDepth < 2000) {
      uint32_t fid = static_cast<uint32_t>(funcAddr - 0xF0000000ULL);
      noteAotFuncIdCall(fid);
      if (aotDepth != 0) {
        ++entryTableSkippedDepthCount;
      } else if (fid < numCompiledAllFuncs && compiledFuncEntries[fid]) {
        bool isNativeEntry =
            (fid < compiledFuncIsNative.size() && compiledFuncIsNative[fid]);
        bool hasTrampolineEntry =
            (fid < compiledFuncHasTrampoline.size() &&
             compiledFuncHasTrampoline[fid]);
        // Deny/trap checks for call_indirect main dispatch path.
        if (isNativeEntry && aotDenyFids.count(fid))
          goto ci_main_interpreted;
        if (isNativeEntry && static_cast<int32_t>(fid) == aotTrapFid) {
          llvm::errs() << "[AOT TRAP] ci-main fid=" << fid;
          if (fid < aotFuncEntryNamesById.size())
            llvm::errs() << " name=" << aotFuncEntryNamesById[fid];
          llvm::errs() << "\n";
          __builtin_trap();
        }
        // Runtime interception policy may mark a FuncId as non-native even
        // when the compiled module still has a direct entry pointer. Only
        // call non-native entries through generated trampolines.
        if (!isNativeEntry && !hasTrampolineEntry)
          goto ci_main_interpreted;
        // Skip native dispatch for yield-capable functions outside process
        // context.
        if (shouldSkipMayYieldEntry(fid, isNativeEntry)) {
          ++entryTableSkippedYieldCount;
          noteAotEntryYieldSkip(fid);
          goto ci_main_interpreted;
        }
        void *entryPtr = const_cast<void *>(compiledFuncEntries[fid]);
        if (entryPtr) {
          unsigned numArgs = funcOp.getNumArguments();
          unsigned numResults = funcOp.getNumResults();
          bool eligible = (numArgs <= 8 && numResults <= 1);
          if (eligible) {
            for (unsigned i = 0; i < numArgs && eligible; ++i) {
              auto ty = funcOp.getArgumentTypes()[i];
              if (auto intTy = dyn_cast<mlir::IntegerType>(ty)) {
                if (intTy.getWidth() > 64) eligible = false;
              } else if (isa<mlir::IndexType>(ty) ||
                         isa<mlir::LLVM::LLVMPointerType>(ty)) {
                // OK
              } else {
                eligible = false;
              }
            }
            if (numResults == 1) {
              auto resTy = funcOp.getResultTypes()[0];
              if (auto intTy = dyn_cast<mlir::IntegerType>(resTy)) {
                if (intTy.getWidth() > 64) eligible = false;
              } else if (!isa<mlir::IndexType>(resTy) &&
                         !isa<mlir::LLVM::LLVMPointerType>(resTy)) {
                eligible = false;
              }
            }
          }
          if (eligible) {
            uint64_t a[8] = {};
            bool normalizePointerArgs = isNativeEntry;
            if (shouldForceInterpretedFragileUvmCallee(calleeName))
              goto ci_main_interpreted;
            bool forcePredicateFalse = false;
            eligible = fillNativeCallArgs(args, funcOp.getArgumentTypes(),
                                          calleeName, numArgs, a,
                                          normalizePointerArgs,
                                          forcePredicateFalse);
            if (forcePredicateFalse) {
              setCallIndirectResults({});
              if (indAddedToVisited)
                decrementRecursionDepthEntry(callState, indFuncKey, indArg0Val);
              return success();
            }
            if (!eligible)
              goto ci_main_interpreted;
            maybeTraceIndirectNative(fid, calleeName, isNativeEntry, numArgs,
                                     numResults, a);

            if (eligible) {

            // Set TLS context so Moore runtime helpers can normalize ptrs.
            void *prevTls = __circt_sim_get_tls_ctx();
            __circt_sim_set_tls_ctx(static_cast<void *>(this));
            __circt_sim_set_tls_normalize(LLHDProcessInterpreter::normalizeVirtualPtr);
            uint64_t result = 0;
            using F0 = uint64_t (*)();
            using F1 = uint64_t (*)(uint64_t);
            using F2 = uint64_t (*)(uint64_t, uint64_t);
            using F3 = uint64_t (*)(uint64_t, uint64_t, uint64_t);
            using F4 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);
            using F5 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t);
            using F6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t);
            using F7 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t);
            using F8 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                    uint64_t, uint64_t, uint64_t, uint64_t);
            ++callState.callDepth;
            switch (numArgs) {
            case 0: result = reinterpret_cast<F0>(entryPtr)(); break;
            case 1: result = reinterpret_cast<F1>(entryPtr)(a[0]); break;
            case 2: result = reinterpret_cast<F2>(entryPtr)(a[0], a[1]); break;
            case 3: result = reinterpret_cast<F3>(entryPtr)(a[0], a[1], a[2]); break;
            case 4: result = reinterpret_cast<F4>(entryPtr)(a[0], a[1], a[2], a[3]); break;
            case 5: result = reinterpret_cast<F5>(entryPtr)(a[0], a[1], a[2], a[3], a[4]); break;
            case 6: result = reinterpret_cast<F6>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5]); break;
            case 7: result = reinterpret_cast<F7>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6]); break;
            case 8: result = reinterpret_cast<F8>(entryPtr)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]); break;
            }
            __circt_sim_set_tls_ctx(prevTls);
            --callState.callDepth;
            SmallVector<InterpretedValue, 2> results;
            if (numResults == 1) {
              auto resTy = funcOp.getResultTypes()[0];
              unsigned bits = 64;
              if (auto intTy = dyn_cast<mlir::IntegerType>(resTy))
                bits = intTy.getWidth();
              results.push_back(InterpretedValue(
                  llvm::APInt(64, result).zextOrTrunc(bits)));
            }
            setCallIndirectResults(results);
            // Decrement depth counter after returning.
            if (indAddedToVisited)
              decrementRecursionDepthEntry(callState, indFuncKey, indArg0Val);
            if (isNativeEntry)
              ++nativeEntryCallCount;
            else
              ++trampolineEntryCallCount;
            return success();
            } // eligible (no fake addr)
          }
        }
      }
    }

  ci_main_interpreted:
    // Call the function with depth tracking
    ++interpretedCallCounts[funcOp.getOperation()];
    ++callState.callDepth;
    SmallVector<InterpretedValue, 2> results;
    // Pass the call operation so it can be saved in call stack frames
    LogicalResult funcResult =
        interpretFuncBody(procId, funcOp, args, results, callIndirectOp);
    --callState.callDepth;

    // Decrement depth counter after returning
    if (indAddedToVisited)
      decrementRecursionDepthEntry(callState, indFuncKey, indArg0Val);

    if (failed(funcResult)) {
      static bool traceCallFailures = []() {
        const char *env = std::getenv("CIRCT_SIM_TRACE_CALL_FAILURES");
        return env && env[0] != '\0' && env[0] != '0';
      }();
      // Check if the failure was actually a suspension that we should propagate
      // rather than swallow. When a virtual method call causes the process to
      // suspend (e.g., wait_for_state inside sync_phase called from
      // process_phase), interpretFuncBody returns success() with
      // state.waiting=true. But if the function body reached a point that
      // returns failure() (e.g., the while-loop fallthrough at the end of
      // interpretFuncBody), AND the process is actually in waiting state,
      // it means the suspension was set but the function body loop exited
      // without returning success(). In this case, the suspension is valid
      // and we should propagate it, not treat it as an error.
      auto &suspState = processStates[procId];
      if (suspState.waiting) {
        // The function suspended -- this is not an error. Propagate the
        // suspension so the caller can save a call stack frame.
        // callDepth was already decremented above (line after interpretFuncBody).
        LLVM_DEBUG(llvm::dbgs() << "  call_indirect: '" << calleeName
                                << "' returned failure but process is waiting"
                                << " -- treating as suspension\n");
        return success();
      }
      if (shouldPropagateCoverageRuntimeFailure(procId))
        return failure();
      if (traceCallFailures) {
        auto &failState = processStates[procId];
        llvm::errs() << "[CALLFAIL] proc=" << procId
                     << " callee=" << calleeName
                     << " waiting=" << failState.waiting
                     << " halted=" << failState.halted
                     << " callStack=" << failState.callStack.size()
                     << " lastOp="
                     << (failState.lastOp
                             ? failState.lastOp->getName().getStringRef()
                             : llvm::StringRef("<none>"))
                     << "\n";
      }
      // uvm_root::die can intentionally unwind through termination/fatal
      // paths. Treat this as an absorbed terminal call, not an internal
      // virtual-dispatch failure warning.
      if (calleeName == "uvm_pkg::uvm_root::die" ||
          calleeName.ends_with("::die")) {
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
        return success();
      }
      // Don't propagate internal failures from virtual method calls.
      // During UVM phase traversal, individual component methods may fail
      // (e.g., unimplemented sequencer interfaces, missing config_db entries).
      // Propagating the failure would cascade through the recursive
      // traverse_on -> traverse -> traverse_on chain, halting all phase
      // processing.  Instead, absorb the failure, log a warning, and
      // return zero results so the traversal can continue to the next
      // component.
      // Suppress the warning when the failure is from abort/timeout —
      // the abort itself is the expected message.
      if (!isAbortRequested())
        emitVtableWarning("dispatched function '" + calleeName.str() +
                          "' failed internally");
      for (Value result : callIndirectOp.getResults()) {
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
      }
      return success();
    }

    // Check if process suspended during function execution (e.g., due to wait)
    // If so, return early without setting results - the function didn't complete
    auto &postCallState = processStates[procId];
    if (postCallState.waiting) {
      LLVM_DEBUG(llvm::dbgs() << "  call_indirect: process suspended during call to '"
                              << calleeName << "'\n");
      return success();
    }

    // Set results.
    setCallIndirectResults(results);

    if (!disableTypeNameCache && calleeName.contains("get_type_name") &&
        !calleeName.contains("get_type_name_enabled") &&
        !results.empty()) {
      cachedTypeNameByCallee[calleeName] = results.front();
    }

    return success();
}
