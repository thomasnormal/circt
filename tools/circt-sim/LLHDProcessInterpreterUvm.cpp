//===- LLHDProcessInterpreterUvm.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains LLHDProcessInterpreter UVM adapter/cache helpers
// extracted from LLHDProcessInterpreter.cpp.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "circt/Runtime/MooreRuntime.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace mlir;
using namespace circt;
using namespace circt::sim;

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

bool LLHDProcessInterpreter::lookupUvmSequencerQueueCache(
    uint64_t portAddr, uint64_t &queueAddr) {
  if (portAddr == 0)
    return false;
  auto it = portToSequencerQueue.find(portAddr);
  if (it == portToSequencerQueue.end()) {
    ++uvmSeqQueueCacheMisses;
    return false;
  }
  ++uvmSeqQueueCacheHits;
  queueAddr = it->second;
  return true;
}

void LLHDProcessInterpreter::cacheUvmSequencerQueueAddress(uint64_t portAddr,
                                                           uint64_t queueAddr) {
  if (portAddr == 0 || queueAddr == 0)
    return;

  auto it = portToSequencerQueue.find(portAddr);
  if (it != portToSequencerQueue.end()) {
    it->second = queueAddr;
    return;
  }

  if (uvmSeqQueueCacheMaxEntries &&
      portToSequencerQueue.size() >= uvmSeqQueueCacheMaxEntries) {
    if (!uvmSeqQueueCacheEvictOnCap) {
      ++uvmSeqQueueCacheCapacitySkips;
      return;
    }
    if (!portToSequencerQueue.empty()) {
      portToSequencerQueue.erase(portToSequencerQueue.begin());
      ++uvmSeqQueueCacheEvictions;
    }
  }

  portToSequencerQueue.try_emplace(portAddr, queueAddr);
  ++uvmSeqQueueCacheInstalls;
}

void LLHDProcessInterpreter::invalidateUvmSequencerQueueCache(
    uint64_t portAddr) {
  if (portAddr == 0)
    return;
  portToSequencerQueue.erase(portAddr);
}

uint64_t LLHDProcessInterpreter::canonicalizeUvmObjectAddress(ProcessId procId,
                                                              uint64_t addr) {
  if (addr == 0)
    return 0;
  uint64_t off = 0;
  MemoryBlock *objBlock = findMemoryBlockByAddress(addr, procId, &off);
  if (!objBlock)
    objBlock = findBlockByAddress(addr, off);
  if (objBlock && off != 0 && off < objBlock->size)
    return addr - off;
  uint64_t nativeOff = 0;
  size_t nativeSize = 0;
  if (findNativeMemoryBlockByAddress(addr, &nativeOff, &nativeSize) &&
      nativeOff != 0 && nativeOff < nativeSize)
    return addr - nativeOff;
  return addr;
}

uint64_t LLHDProcessInterpreter::resolveQueueStructAddress(ProcessId procId,
                                                           uint64_t rawAddr) {
  if (rawAddr == 0)
    return 0;

  auto isValidQueueStructAddress = [&](uint64_t addr) -> bool {
    if (addr == 0)
      return false;

    uint64_t offset = 0;
    if (auto *block = findMemoryBlockByAddress(addr, procId, &offset))
      return offset + 16 <= block->size;
    if (auto *block = findBlockByAddress(addr, offset))
      return offset + 16 <= block->size;

    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    return findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize) &&
           nativeOffset + 16 <= nativeSize;
  };

  auto tryResolve = [&](uint64_t addr) -> uint64_t {
    if (isValidQueueStructAddress(addr))
      return addr;

    const uint64_t maskedCandidates[] = {
        addr & ~uint64_t(1), addr & ~uint64_t(3), addr & ~uint64_t(7)};
    for (uint64_t masked : maskedCandidates) {
      if (isValidQueueStructAddress(masked))
        return masked;
    }
    return 0;
  };

  uint64_t resolved = tryResolve(rawAddr);
  if (resolved != 0)
    return resolved;

  uint64_t canonical = canonicalizeUvmObjectAddress(procId, rawAddr);
  if (uint64_t canonicalResolved = tryResolve(canonical))
    return canonicalResolved;

  return rawAddr;
}

uint64_t LLHDProcessInterpreter::readUvmPhaseImpAddress(ProcessId procId,
                                                        uint64_t phaseAddr) {
  if (phaseAddr == 0)
    return 0;

  static std::optional<unsigned> cachedImpOffset;
  unsigned impOffset = 44;
  if (!cachedImpOffset) {
    if (rootModule) {
      if (auto getImpFunc = rootModule.lookupSymbol<func::FuncOp>(
              "uvm_pkg::uvm_phase::get_imp")) {
        for (Block &block : getImpFunc.getBody()) {
          for (Operation &op : block) {
            auto gepOp = dyn_cast<LLVM::GEPOp>(&op);
            if (!gepOp)
              continue;
            auto structTy = dyn_cast<LLVM::LLVMStructType>(gepOp.getElemType());
            if (!structTy || !structTy.isIdentified() ||
                !structTy.getName().contains("uvm_phase"))
              continue;
            impOffset = getLLVMStructFieldOffset(structTy, 3);
            cachedImpOffset = impOffset;
            break;
          }
          if (cachedImpOffset)
            break;
        }
      }
    }
    if (!cachedImpOffset)
      cachedImpOffset = impOffset;
  } else {
    impOffset = *cachedImpOffset;
  }

  uint64_t impAddrField = phaseAddr + impOffset;
  uint64_t off = 0;
  if (MemoryBlock *blk = findMemoryBlockByAddress(impAddrField, procId, &off)) {
    if (blk->initialized && off + 8 <= blk->size) {
      uint64_t impAddr = 0;
      for (unsigned i = 0; i < 8; ++i)
        impAddr |= static_cast<uint64_t>(blk->bytes()[off + i]) << (i * 8);
      return impAddr;
    }
  }

  if (MemoryBlock *blk = findBlockByAddress(impAddrField, off)) {
    if (blk->initialized && off + 8 <= blk->size) {
      uint64_t impAddr = 0;
      for (unsigned i = 0; i < 8; ++i)
        impAddr |= static_cast<uint64_t>(blk->bytes()[off + i]) << (i * 8);
      return impAddr;
    }
  }

  uint64_t nativeOff = 0;
  size_t nativeSize = 0;
  if (findNativeMemoryBlockByAddress(impAddrField, &nativeOff, &nativeSize) &&
      nativeOff + 8 <= nativeSize) {
    uint64_t impAddr = 0;
    std::memcpy(&impAddr, reinterpret_cast<void *>(impAddrField),
                sizeof(impAddr));
    return impAddr;
  }

  return 0;
}

void LLHDProcessInterpreter::recordUvmPhaseAddSequence(
    ProcessId procId, llvm::ArrayRef<InterpretedValue> args) {
  if (args.size() < 2 || args[0].isX() || args[1].isX())
    return;

  uint64_t rootAddr = normalizeUvmObjectKey(procId, args[0].getUInt64());
  if (rootAddr == 0)
    rootAddr = args[0].getUInt64();
  uint64_t phaseImpAddr = args[1].getUInt64();
  if (rootAddr == 0 || phaseImpAddr == 0)
    return;

  auto &sequence = phaseRootImpSequence[rootAddr];
  if (std::find(sequence.begin(), sequence.end(), phaseImpAddr) ==
      sequence.end())
    sequence.push_back(phaseImpAddr);
}

uint64_t LLHDProcessInterpreter::mapUvmPhaseAddressToActiveGraph(
    ProcessId procId, uint64_t phaseAddr) {
  static bool tracePhaseRemap = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_PHASE_REMAP");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto traceReturn = [&](llvm::StringRef reason, uint64_t ret,
                         uint64_t normalized = 0, uint64_t imp = 0,
                         uint64_t mappedImp = 0) -> uint64_t {
    if (tracePhaseRemap) {
      auto readPhaseName = [&](uint64_t addr) -> std::string {
        if (addr == 0)
          return {};
        uint64_t nameOff = 0;
        MemoryBlock *nameBlk = findBlockByAddress(addr + 12, nameOff);
        if (!nameBlk || nameOff + 16 > nameBlk->size)
          return {};
        uint64_t namePtr = 0;
        uint64_t nameLen = 0;
        for (int i = 0; i < 8; ++i) {
          namePtr |= static_cast<uint64_t>(nameBlk->bytes()[nameOff + i])
                     << (i * 8);
          nameLen |= static_cast<uint64_t>(nameBlk->bytes()[nameOff + 8 + i])
                     << (i * 8);
        }
        if (namePtr == 0 || nameLen == 0 || nameLen > 128)
          return {};
        uint64_t strOff = 0;
        MemoryBlock *strBlk = findBlockByAddress(namePtr, strOff);
        if (!strBlk || strOff + nameLen > strBlk->size)
          return {};
        return std::string(
            reinterpret_cast<const char *>(strBlk->bytes() + strOff),
            static_cast<size_t>(nameLen));
      };
      std::string normalizedName = readPhaseName(normalized);
      std::string retName = readPhaseName(ret);
      llvm::errs() << "[PHASE-REMAP-DETAIL] proc=" << procId
                   << " phase=" << llvm::format_hex(phaseAddr, 16)
                   << " normalized=" << llvm::format_hex(normalized, 16)
                   << " active_root="
                   << llvm::format_hex(activePhaseRootAddr, 16)
                   << " imp=" << llvm::format_hex(imp, 16)
                   << " mapped_imp=" << llvm::format_hex(mappedImp, 16)
                   << " ret=" << llvm::format_hex(ret, 16)
                   << " normalized_name=\""
                   << (normalizedName.empty() ? "<unknown>" : normalizedName)
                   << "\""
                   << " ret_name=\"" << (retName.empty() ? "<unknown>" : retName)
                   << "\""
                   << " reason=" << reason << "\n";
    }
    return ret;
  };
  if (phaseAddr == 0 || activePhaseRootAddr == 0)
    return traceReturn("phase-or-active-root-zero", phaseAddr);

  uint64_t normalized = normalizeUvmObjectKey(procId, phaseAddr);
  if (normalized == 0)
    normalized = phaseAddr;

  uint64_t phaseImpAddr = readUvmPhaseImpAddress(procId, normalized);
  if (phaseImpAddr == 0) {
    // Root/common-domain wrappers can have m_imp==0. In stale-domain waits we
    // still need to remap those wrappers onto the active common domain.
    auto activeSeqIt = phaseRootImpSequence.find(activePhaseRootAddr);
    if (activeSeqIt != phaseRootImpSequence.end()) {
      if (phaseRootImpSequence.find(normalized) != phaseRootImpSequence.end() &&
          normalized != activePhaseRootAddr) {
        return traceReturn("phase-imp-zero-root-key", activePhaseRootAddr,
                           normalized, phaseImpAddr);
      }
      constexpr uint64_t kCommonWrapperDelta = 0xD0;
      if (normalized > kCommonWrapperDelta) {
        uint64_t maybeRootKey = normalized - kCommonWrapperDelta;
        if (phaseRootImpSequence.find(maybeRootKey) !=
                phaseRootImpSequence.end() &&
            maybeRootKey != activePhaseRootAddr) {
          return traceReturn("phase-imp-zero-wrapper-root-key",
                             activePhaseRootAddr, normalized, phaseImpAddr);
        }
      }
    }
    return traceReturn("phase-imp-zero", normalized, normalized, phaseImpAddr);
  }

  auto activeSeqIt = phaseRootImpSequence.find(activePhaseRootAddr);
  if (activeSeqIt == phaseRootImpSequence.end() || activeSeqIt->second.empty())
    return traceReturn("active-seq-missing", normalized, normalized,
                       phaseImpAddr);
  const auto &activeSeq = activeSeqIt->second;

  auto readPhaseName = [&](uint64_t addr) -> std::string {
    if (addr == 0)
      return {};
    uint64_t nameOff = 0;
    MemoryBlock *nameBlk = findBlockByAddress(addr + 12, nameOff);
    if (!nameBlk || nameOff + 16 > nameBlk->size)
      return {};
    uint64_t namePtr = 0;
    uint64_t nameLen = 0;
    for (int i = 0; i < 8; ++i) {
      namePtr |= static_cast<uint64_t>(nameBlk->bytes()[nameOff + i])
                 << (i * 8);
      nameLen |= static_cast<uint64_t>(nameBlk->bytes()[nameOff + 8 + i])
                 << (i * 8);
    }
    if (namePtr == 0 || nameLen == 0 || nameLen > 128)
      return {};
    uint64_t strOff = 0;
    MemoryBlock *strBlk = findBlockByAddress(namePtr, strOff);
    if (!strBlk || strOff + nameLen > strBlk->size)
      return {};
    return std::string(reinterpret_cast<const char *>(strBlk->bytes() + strOff),
                       static_cast<size_t>(nameLen));
  };

  auto activePos =
      std::find(activeSeq.begin(), activeSeq.end(), phaseImpAddr);
  std::string phaseName;
  if (activePos == activeSeq.end()) {
    phaseName = readPhaseName(normalized);
    if (phaseName.empty())
      phaseName = readPhaseName(phaseImpAddr);
    auto isLegacyRuntimePhaseName = [&](llvm::StringRef name) {
      return name == "pre_reset" || name == "reset" || name == "post_reset" ||
             name == "pre_configure" || name == "configure" ||
             name == "post_configure" || name == "pre_main" ||
             name == "main" || name == "post_main" ||
             name == "pre_shutdown" || name == "shutdown" ||
             name == "post_shutdown";
    };
    if (!phaseName.empty() && isLegacyRuntimePhaseName(phaseName)) {
      return traceReturn("legacy-runtime-phase-to-active-root",
                         activePhaseRootAddr, normalized, phaseImpAddr);
    }
    // If we cannot identify the stale phase reliably, prefer converging onto
    // the active root over index-based remaps, which can deadlock waits by
    // targeting unrelated active phases.
    if (phaseName.empty()) {
      return traceReturn("unknown-stale-phase-to-active-root",
                         activePhaseRootAddr, normalized, phaseImpAddr);
    }
  }
  if (activePos != activeSeq.end())
    return traceReturn("already-in-active-seq", normalized, normalized,
                       phaseImpAddr, phaseImpAddr);

  uint64_t mappedImpAddr = 0;
  for (const auto &entry : phaseRootImpSequence) {
    if (entry.first == activePhaseRootAddr)
      continue;
    const auto &seq = entry.second;
    auto pos = std::find(seq.begin(), seq.end(), phaseImpAddr);
    if (pos == seq.end())
      continue;
    size_t index = static_cast<size_t>(pos - seq.begin());
    // Some legacy/stale phase graphs include extra leading runtime phases
    // (e.g. pre_reset/reset) that are absent from the active common-domain
    // graph. Align by dropping those leading entries and map dropped ones to
    // the active root.
    if (seq.size() > activeSeq.size()) {
      size_t leadingExtra = seq.size() - activeSeq.size();
      if (index < leadingExtra)
        return traceReturn("index-in-legacy-leading-extra",
                           activePhaseRootAddr, normalized, phaseImpAddr);
      index -= leadingExtra;
    }
    if (index < activeSeq.size())
      mappedImpAddr = activeSeq[index];
    break;
  }
  if (mappedImpAddr == 0)
    return traceReturn("mapped-imp-missing", normalized, normalized,
                       phaseImpAddr, mappedImpAddr);

  auto tryCandidate = [&](uint64_t delta) -> uint64_t {
    if (delta > std::numeric_limits<uint64_t>::max() - mappedImpAddr)
      return 0;
    uint64_t candidate = mappedImpAddr + delta;
    return readUvmPhaseImpAddress(procId, candidate) == mappedImpAddr
               ? candidate
               : 0;
  };

  uint64_t delta = normalized > phaseImpAddr ? (normalized - phaseImpAddr) : 0;
  if (delta <= 0x1000) {
    if (uint64_t candidate = tryCandidate(delta))
      return traceReturn("candidate-from-delta", candidate, normalized,
                         phaseImpAddr, mappedImpAddr);
  }
  if (uint64_t candidate = tryCandidate(/*common wrapper delta=*/0xD0))
    return traceReturn("candidate-from-wrapper-delta", candidate, normalized,
                       phaseImpAddr, mappedImpAddr);

  for (const auto &[phaseKey, _] : phaseObjectionHandles) {
    if (readUvmPhaseImpAddress(procId, phaseKey) == mappedImpAddr)
      return traceReturn("candidate-from-objection-map", phaseKey, normalized,
                         phaseImpAddr, mappedImpAddr);
  }

  return traceReturn("fallback-normalized", normalized, normalized,
                     phaseImpAddr, mappedImpAddr);
}

void LLHDProcessInterpreter::maybeRemapUvmPhaseArgsToActiveGraph(
    ProcessId procId, llvm::StringRef calleeName,
    llvm::SmallVectorImpl<InterpretedValue> &args) {
  bool remapArg0ToActiveGraph =
      !args.empty() && !args[0].isX() &&
      (calleeName == "uvm_pkg::uvm_phase::wait_for_state" ||
       calleeName == "uvm_pkg::uvm_phase::get_predecessors" ||
       calleeName == "uvm_pkg::uvm_phase::get_sync_relationships");
  bool remapJumpTargetToActiveGraph =
      args.size() >= 2 && !args[1].isX() &&
      calleeName == "uvm_pkg::uvm_phase::set_jump_phase";
  if (!remapArg0ToActiveGraph && !remapJumpTargetToActiveGraph)
    return;

  static bool tracePhaseRemap = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_PHASE_REMAP");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto rewriteArg = [&](size_t argIndex, uint64_t newPhase) {
    if (argIndex >= args.size())
      return;
    unsigned width = args[argIndex].getWidth();
    if (width == 0)
      width = 64;
    args[argIndex] = InterpretedValue(llvm::APInt(width, newPhase));
  };

  if (remapArg0ToActiveGraph) {
    uint64_t oldPhase = args[0].getUInt64();
    uint64_t newPhase = mapUvmPhaseAddressToActiveGraph(procId, oldPhase);
    if (calleeName == "uvm_pkg::uvm_phase::wait_for_state" && newPhase != 0 &&
        args.size() >= 3 && !args[1].isX() && !args[2].isX()) {
      auto currentPhaseIt = currentExecutingPhaseAddr.find(procId);
      uint64_t currentPhase =
          currentPhaseIt != currentExecutingPhaseAddr.end()
              ? currentPhaseIt->second
              : 0;
      constexpr uint64_t kUvmPhaseReadyToEnd = 32;
      constexpr uint64_t kUvmWaitGte = 5;
      uint64_t waitState = args[1].getUInt64();
      uint64_t waitCmp = args[2].getUInt64();
      if (currentPhase != 0 && newPhase == currentPhase &&
          waitCmp == kUvmWaitGte && waitState >= kUvmPhaseReadyToEnd &&
          activePhaseRootAddr != 0 && activePhaseRootAddr != newPhase) {
        if (tracePhaseRemap) {
          llvm::errs() << "[PHASE-REMAP] proc=" << procId
                       << " fn=" << calleeName
                       << " self-phase-deadlock-avoid old="
                       << llvm::format_hex(newPhase, 16) << " new="
                       << llvm::format_hex(activePhaseRootAddr, 16)
                       << " waitMask=0x"
                       << llvm::format_hex_no_prefix(waitState, 0)
                       << " waitCmp=" << waitCmp << "\n";
        }
        newPhase = activePhaseRootAddr;
      }
    }
    if (tracePhaseRemap) {
      llvm::errs() << "[PHASE-REMAP] proc=" << procId << " fn=" << calleeName
                   << " old=" << llvm::format_hex(oldPhase, 16)
                   << " new=" << llvm::format_hex(newPhase, 16) << "\n";
    }
    if (newPhase != 0 && newPhase != oldPhase)
      rewriteArg(0, newPhase);
  }

  if (remapJumpTargetToActiveGraph) {
    uint64_t oldPhase = args[1].getUInt64();
    uint64_t newPhase = mapUvmPhaseAddressToActiveGraph(procId, oldPhase);
    if (tracePhaseRemap) {
      llvm::errs() << "[PHASE-REMAP] proc=" << procId << " fn=" << calleeName
                   << " arg1-old=" << llvm::format_hex(oldPhase, 16)
                   << " arg1-new=" << llvm::format_hex(newPhase, 16) << "\n";
    }
    if (newPhase != 0 && newPhase != oldPhase)
      rewriteArg(1, newPhase);
  }
}

void LLHDProcessInterpreter::maybeCanonicalizeUvmPhasePredecessorSet(
    ProcessId procId, llvm::StringRef calleeName,
    llvm::ArrayRef<InterpretedValue> args) {
  if (args.size() < 2)
    return;
  bool isPhaseSetGetter =
      calleeName == "uvm_pkg::uvm_phase::get_predecessors" ||
      calleeName == "uvm_pkg::uvm_phase::get_sync_relationships";
  if (!isPhaseSetGetter)
    return;
  if (args[1].isX())
    return;

  uint64_t assocAddr = args[1].getUInt64();
  if (assocAddr == 0)
    return;

  std::vector<std::pair<uint64_t, uint64_t>> rewrites;
  uint64_t key = 0;
  void *assocPtr = reinterpret_cast<void *>(assocAddr);
  if (!__moore_assoc_first(assocPtr, &key))
    return;
  do {
    uint64_t mappedKey = mapUvmPhaseAddressToActiveGraph(procId, key);
    if (mappedKey != 0 && mappedKey != key)
      rewrites.emplace_back(key, mappedKey);
  } while (__moore_assoc_next(assocPtr, &key));

  if (rewrites.empty())
    return;

  static bool tracePhasePredCanonicalization = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_PHASE_PRED_CANON");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  for (const auto &[oldKey, newKey] : rewrites) {
    uint64_t keyToInsert = newKey;
    if (void *ref = __moore_assoc_get_ref(assocPtr, &keyToInsert,
                                          /*value_size=*/1))
      *reinterpret_cast<uint8_t *>(ref) = 1;
    uint64_t keyToDelete = oldKey;
    __moore_assoc_delete_key(assocPtr, &keyToDelete);

    if (tracePhasePredCanonicalization) {
      llvm::errs() << "[PHASE-PRED-CANON] proc=" << procId
                   << " fn=" << calleeName
                   << " old=0x" << llvm::format_hex(oldKey, 16)
                   << " new=0x" << llvm::format_hex(newKey, 16) << "\n";
    }
  }
}

uint64_t LLHDProcessInterpreter::normalizeNativeCallPointerArg(
    ProcessId procId, llvm::StringRef calleeName, uint64_t rawAddr) {
  if (rawAddr == 0)
    return 0;

  bool tracePtr = std::getenv("CIRCT_AOT_TRACE_PTR") != nullptr;
  auto traceRewrite = [&](llvm::StringRef kind, uint64_t from, uint64_t to) {
    if (!tracePtr || from == to)
      return;
    llvm::errs() << "[AOT PTR] native-arg-normalize kind=" << kind
                 << " callee=" << calleeName << " from="
                 << llvm::format_hex(from, 16) << " to="
                 << llvm::format_hex(to, 16) << "\n";
  };

  if (tracePtr && calleeName.contains("uvm_callback_iter::first")) {
    uint64_t blockOff = 0;
    MemoryBlock *blk = findMemoryBlockByAddress(rawAddr, procId, &blockOff);
    llvm::errs() << "[AOT PTR] callback_iter::first raw="
                 << llvm::format_hex(rawAddr, 16)
                 << " mapped_block=" << (blk ? 1 : 0);
    if (blk)
      llvm::errs() << " off=" << blockOff << " size=" << blk->size
                   << " host="
                   << llvm::format_hex(
                          reinterpret_cast<uint64_t>(blk->bytes()), 16);
    llvm::errs() << "\n";
  }

  auto tryVirtualToHost = [&](uint64_t candidate,
                              llvm::StringRef kind) -> uint64_t {
    if (candidate == 0)
      return 0;
    if (void *mapped = normalizeVirtualPtr(static_cast<void *>(this), candidate)) {
      uint64_t mappedAddr = reinterpret_cast<uint64_t>(mapped);
      if (mappedAddr != 0 && mappedAddr != candidate) {
        traceRewrite(kind, rawAddr, mappedAddr);
        return mappedAddr;
      }
    }
    void *normalized =
        normalizeAssocRuntimePointer(reinterpret_cast<void *>(candidate));
    uint64_t normalizedAddr = reinterpret_cast<uint64_t>(normalized);
    if (normalizedAddr != candidate) {
      traceRewrite(kind, rawAddr, normalizedAddr);
      return normalizedAddr;
    }
    return 0;
  };

  // Low-bit tagging is used on UVM object handles in several paths
  // (especially around phase graph/state operations). Native code expects a
  // canonical object pointer, not tagged variants.
  bool isLikelyObjectMethod =
      (calleeName.contains("uvm_") || calleeName.contains("::")) &&
      !calleeName.starts_with("__");
  if (isLikelyObjectMethod && (rawAddr & uint64_t(7)) != 0) {
    llvm::SmallVector<uint64_t, 8> untagCandidates;
    auto addUntagCandidate = [&](uint64_t candidate) {
      if (candidate == 0 || candidate == rawAddr)
        return;
      if (std::find(untagCandidates.begin(), untagCandidates.end(), candidate) !=
          untagCandidates.end())
        return;
      untagCandidates.push_back(candidate);
    };

    const uint64_t maskedCandidates[] = {
        rawAddr & ~uint64_t(1), rawAddr & ~uint64_t(3), rawAddr & ~uint64_t(7)};
    for (uint64_t masked : maskedCandidates) {
      addUntagCandidate(masked);
      addUntagCandidate(canonicalizeUvmObjectAddress(procId, masked));
    }
    addUntagCandidate(canonicalizeUvmObjectAddress(procId, rawAddr));

    for (uint64_t candidate : untagCandidates) {
      if (uint64_t hostAddr = tryVirtualToHost(candidate, "virtual-untagged"))
        return hostAddr;
    }
    if (!untagCandidates.empty()) {
      traceRewrite("untag-fallback", rawAddr, untagCandidates.front());
      return untagCandidates.front();
    }
  }

  if (uint64_t hostAddr = tryVirtualToHost(rawAddr, "virtual-direct"))
    return hostAddr;

  if (!isLikelyObjectMethod)
    return rawAddr;

  llvm::SmallVector<uint64_t, 8> candidates;
  auto addCandidate = [&](uint64_t candidate) {
    if (candidate == 0)
      return;
    if (std::find(candidates.begin(), candidates.end(), candidate) !=
        candidates.end())
      return;
    candidates.push_back(candidate);
  };

  addCandidate(rawAddr);
  addCandidate(canonicalizeUvmObjectAddress(procId, rawAddr));

  for (uint64_t candidate : candidates) {
    if (uint64_t hostAddr = tryVirtualToHost(candidate, "virtual-tagged"))
      return hostAddr;
  }

  for (uint64_t candidate : candidates) {
    if (candidate == rawAddr)
      continue;
    traceRewrite("untag-fallback", rawAddr, candidate);
    return candidate;
  }

  return rawAddr;
}

void LLHDProcessInterpreter::cacheSequenceRuntimeVtableForObject(
    ProcessId procId, uint64_t objectAddr, uint64_t vtableAddr) {
  if (objectAddr == 0 || vtableAddr == 0)
    return;

  auto classifySequenceBodyVtable = [&](uint64_t candidateVtable) -> int {
    // -1: unknown, 0: non-base body target, 1: base body target.
    auto globalIt = addressToGlobal.find(candidateVtable);
    if (globalIt == addressToGlobal.end())
      return -1;
    auto blockIt = globalMemoryBlocks.find(globalIt->second);
    if (blockIt == globalMemoryBlocks.end())
      return -1;
    auto &vtableBlock = blockIt->second;
    constexpr uint64_t kBodySlot = 43;
    unsigned slotOffset = static_cast<unsigned>(kBodySlot * 8);
    if (slotOffset + 8 > vtableBlock.size)
      return -1;

    uint64_t slotFuncAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      slotFuncAddr |= static_cast<uint64_t>(vtableBlock[slotOffset + i])
                      << (i * 8);
    if (slotFuncAddr == 0)
      return -1;

    auto funcIt = addressToFunction.find(slotFuncAddr);
    if (funcIt == addressToFunction.end())
      return -1;
    if (funcIt->second.find("::body") == std::string::npos)
      return -1;
    if (funcIt->second == "uvm_pkg::uvm_sequence_base::body")
      return 1;
    return 0;
  };
  auto isBaseSequenceVtable = [&](uint64_t candidateVtable) -> bool {
    auto globalIt = addressToGlobal.find(candidateVtable);
    if (globalIt == addressToGlobal.end())
      return false;
    return globalIt->second == "uvm_pkg::uvm_sequence_base::__vtable__";
  };

  llvm::SmallVector<uint64_t, 8> candidates;
  auto addCandidate = [&](uint64_t addr) {
    if (addr == 0)
      return;
    if (std::find(candidates.begin(), candidates.end(), addr) !=
        candidates.end())
      return;
    candidates.push_back(addr);
  };

  addCandidate(objectAddr);
  addCandidate(canonicalizeUvmObjectAddress(procId, objectAddr));
  const uint64_t maskedCandidates[] = {
      objectAddr & ~uint64_t(1), objectAddr & ~uint64_t(3),
      objectAddr & ~uint64_t(7)};
  for (uint64_t masked : maskedCandidates) {
    addCandidate(masked);
    addCandidate(canonicalizeUvmObjectAddress(procId, masked));
  }

  int newClass = classifySequenceBodyVtable(vtableAddr);
  if (newClass == -1)
    return;
  for (uint64_t candidate : candidates) {
    uint64_t &slot = sequenceRuntimeVtableByObjectAddr[candidate];
    if (slot == 0 || slot == vtableAddr) {
      slot = vtableAddr;
      continue;
    }
    bool oldIsBase = isBaseSequenceVtable(slot);
    bool newIsBase = isBaseSequenceVtable(vtableAddr);
    // Never downgrade a discovered non-base sequence vtable to the generic
    // uvm_sequence_base fallback.
    if (!oldIsBase && newIsBase)
      continue;
    // Prefer a discovered non-base vtable over a previously cached base one.
    if (oldIsBase && !newIsBase) {
      slot = vtableAddr;
      continue;
    }
    int oldClass = classifySequenceBodyVtable(slot);
    // Keep derived/non-base mapping when a later fallback only infers base body.
    if (oldClass == 0 && newClass == 1)
      continue;
    // Upgrade cached base-body mapping when we later discover a derived body.
    if (oldClass == 1 && newClass == 0) {
      slot = vtableAddr;
      continue;
    }
    // Unknown/equal-class cases: prefer the latest observation.
    slot = vtableAddr;
  }

  if (traceSeqEnabled) {
    llvm::errs() << "[SEQ-VTBL-CACHE] store proc=" << procId << " obj=0x"
                 << llvm::format_hex(objectAddr, 16) << " vtbl=0x"
                 << llvm::format_hex(vtableAddr, 16) << " cands=";
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (i)
        llvm::errs() << ",";
      llvm::errs() << "0x" << llvm::format_hex(candidates[i], 16);
    }
    llvm::errs() << "\n";
  }
}

bool LLHDProcessInterpreter::lookupCachedSequenceRuntimeVtable(
    ProcessId procId, uint64_t objectAddr, uint64_t &vtableAddr) {
  vtableAddr = 0;
  if (objectAddr == 0)
    return false;

  llvm::SmallVector<uint64_t, 8> candidates;
  auto addCandidate = [&](uint64_t addr) {
    if (addr == 0)
      return;
    if (std::find(candidates.begin(), candidates.end(), addr) !=
        candidates.end())
      return;
    candidates.push_back(addr);
  };

  addCandidate(objectAddr);
  addCandidate(canonicalizeUvmObjectAddress(procId, objectAddr));
  const uint64_t maskedCandidates[] = {
      objectAddr & ~uint64_t(1), objectAddr & ~uint64_t(3),
      objectAddr & ~uint64_t(7)};
  for (uint64_t masked : maskedCandidates) {
    addCandidate(masked);
    addCandidate(canonicalizeUvmObjectAddress(procId, masked));
  }

  for (uint64_t candidate : candidates) {
    auto it = sequenceRuntimeVtableByObjectAddr.find(candidate);
    if (it == sequenceRuntimeVtableByObjectAddr.end() || it->second == 0)
      continue;
    vtableAddr = it->second;
    if (traceSeqEnabled) {
      llvm::errs() << "[SEQ-VTBL-CACHE] hit proc=" << procId << " obj=0x"
                   << llvm::format_hex(objectAddr, 16) << " cand=0x"
                   << llvm::format_hex(candidate, 16) << " vtbl=0x"
                   << llvm::format_hex(vtableAddr, 16) << "\n";
    }
    return true;
  }

  if (traceSeqEnabled) {
    llvm::errs() << "[SEQ-VTBL-CACHE] miss proc=" << procId << " obj=0x"
                 << llvm::format_hex(objectAddr, 16) << " cands=";
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (i)
        llvm::errs() << ",";
      llvm::errs() << "0x" << llvm::format_hex(candidates[i], 16);
    }
    llvm::errs() << "\n";
  }

  return false;
}

void LLHDProcessInterpreter::maybeSeedSequenceRuntimeVtableFromFunction(
    ProcessId procId, llvm::StringRef funcName,
    llvm::ArrayRef<InterpretedValue> args) {
  if (args.empty() || args[0].isX())
    return;
  uint64_t objectAddr = args[0].getUInt64();
  if (objectAddr == 0)
    return;

  auto isSequenceBodyVtable = [&](uint64_t candidateVtable) -> bool {
    auto globalIt = addressToGlobal.find(candidateVtable);
    if (globalIt == addressToGlobal.end())
      return false;
    auto blockIt = globalMemoryBlocks.find(globalIt->second);
    if (blockIt == globalMemoryBlocks.end())
      return false;
    auto &vtableBlock = blockIt->second;
    constexpr uint64_t kBodySlot = 43;
    unsigned slotOffset = static_cast<unsigned>(kBodySlot * 8);
    if (slotOffset + 8 > vtableBlock.size)
      return false;
    uint64_t slotFuncAddr = 0;
    for (unsigned i = 0; i < 8; ++i)
      slotFuncAddr |= static_cast<uint64_t>(vtableBlock[slotOffset + i]) << (i * 8);
    if (slotFuncAddr == 0)
      return false;
    auto funcIt = addressToFunction.find(slotFuncAddr);
    if (funcIt == addressToFunction.end())
      return false;
    return funcIt->second.find("::body") != std::string::npos;
  };

  // First prefer a real runtime-header read when available.
  uint64_t runtimeVtableAddr = 0;
  if (readObjectVTableAddress(objectAddr, runtimeVtableAddr, procId) &&
      isSequenceBodyVtable(runtimeVtableAddr)) {
    cacheSequenceRuntimeVtableForObject(procId, objectAddr, runtimeVtableAddr);
    return;
  }

  // Otherwise infer from the statically-known class vtable symbol of the
  // current method (Class::method -> Class::__vtable__). Restrict this to
  // vtables whose slot 43 maps to a ::body method (sequence classes).
  size_t scopeSep = funcName.rfind("::");
  if (scopeSep == llvm::StringRef::npos)
    return;
  std::string className = funcName.substr(0, scopeSep).str();
  std::string vtableName = className + "::__vtable__";

  auto vtableBlockIt = globalMemoryBlocks.find(vtableName);
  if (vtableBlockIt == globalMemoryBlocks.end())
    return;

  uint64_t inferredVtableAddr = 0;
  for (const auto &entry : addressToGlobal) {
    if (entry.second == vtableName) {
      inferredVtableAddr = entry.first;
      break;
    }
  }
  if (inferredVtableAddr == 0)
    return;

  constexpr uint64_t kBodySlot = 43;
  auto &vtableBlock = vtableBlockIt->second;
  unsigned slotOffset = static_cast<unsigned>(kBodySlot * 8);
  if (slotOffset + 8 > vtableBlock.size)
    return;

  uint64_t slotFuncAddr = 0;
  for (unsigned i = 0; i < 8; ++i)
    slotFuncAddr |= static_cast<uint64_t>(vtableBlock[slotOffset + i]) << (i * 8);
  if (slotFuncAddr == 0)
    return;

  auto funcIt = addressToFunction.find(slotFuncAddr);
  if (funcIt == addressToFunction.end() ||
      funcIt->second.find("::body") == std::string::npos)
    return;

  cacheSequenceRuntimeVtableForObject(procId, objectAddr, inferredVtableAddr);
}

bool LLHDProcessInterpreter::canonicalizeUvmSequencerQueueAddress(
    ProcessId procId, uint64_t &queueAddr, Operation *callSite) {
  if (queueAddr == 0)
    return false;

  bool strongHint = false;
  auto promoteToSequencerQueue =
      [&](uint64_t candidateAddr) -> std::pair<uint64_t, bool> {
    if (candidateAddr == 0)
      return {0, false};
    if (sequencerItemFifo.contains(candidateAddr))
      return {candidateAddr, true};

    uint64_t ownerOff = 0;
    MemoryBlock *ownerBlock =
        findMemoryBlockByAddress(candidateAddr, procId, &ownerOff);
    if (!ownerBlock)
      ownerBlock = findBlockByAddress(candidateAddr, ownerOff);
    if (ownerBlock && ownerOff != 0 && ownerOff < ownerBlock->size) {
      uint64_t ownerAddr = candidateAddr - ownerOff;
      if (sequencerItemFifo.contains(ownerAddr))
        return {ownerAddr, true};
      return {candidateAddr, false};
    }

    uint64_t nativeOff = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(candidateAddr, &nativeOff, &nativeSize) &&
        nativeOff != 0 && nativeOff < nativeSize) {
      uint64_t ownerAddr = candidateAddr - nativeOff;
      if (sequencerItemFifo.contains(ownerAddr))
        return {ownerAddr, true};
      return {candidateAddr, false};
    }

    // Reject unresolved/non-backed candidates to avoid binding waiters to
    // malformed packed values (for example ptr|metadata composites).
    // Callers can still fall back to unresolved queue behavior (queue=0),
    // which is later woken by producer pushes.
    return {0, false};
  };

  uint64_t rawQueueAddr = queueAddr;
  auto [promotedAddr, promotedStrongHint] = promoteToSequencerQueue(queueAddr);
  // Preserve unresolved raw pull-port candidates long enough to run
  // get_parent/get_comp owner resolution. Dropping to zero here prevents
  // valid terminals from ever reaching their owning sequencer queue.
  queueAddr = (promotedAddr != 0) ? promotedAddr : rawQueueAddr;
  strongHint = strongHint || promotedStrongHint;

  auto resolvePortOwner =
      [&](func::FuncOp ownerFunc, uint64_t ownerThisAddr) -> bool {
    SmallVector<InterpretedValue, 1> ownerArgs;
    ownerArgs.push_back(InterpretedValue(llvm::APInt(64, ownerThisAddr)));
    SmallVector<InterpretedValue, 1> ownerResults;
    auto &cState = processStates[procId];
    ++cState.callDepth;
    auto ownerRes =
        interpretFuncBody(procId, ownerFunc, ownerArgs, ownerResults, callSite);
    --cState.callDepth;
    if (failed(ownerRes) || ownerResults.empty() || ownerResults[0].isX() ||
        ownerResults[0].getWidth() != 64)
      return false;
    uint64_t resolvedOwnerAddr = ownerResults[0].getUInt64();
    if (resolvedOwnerAddr == 0)
      return false;
    auto [resolvedQueueAddr, resolvedStrongHint] =
        promoteToSequencerQueue(resolvedOwnerAddr);
    // During initial get_next_item waits, the producer may not have pushed yet,
    // so the sequencer queue won't exist in `sequencerItemFifo` yet. Still
    // accept the owner-resolved queue address so waiters bind to the right
    // queue instead of falling back to global wakeups.
    if (resolvedQueueAddr == 0)
      return false;
    if (resolvedQueueAddr != queueAddr)
      strongHint = true;
    strongHint = strongHint || resolvedStrongHint;
    queueAddr = resolvedQueueAddr;
    return true;
  };

  auto resolvePortOwnerByName = [&](llvm::StringRef symbolName,
                                    uint64_t ownerThisAddr) -> bool {
    if (!rootModule)
      return false;
    auto ownerFunc = rootModule.lookupSymbol<func::FuncOp>(symbolName);
    if (!ownerFunc)
      return false;
    return resolvePortOwner(ownerFunc, ownerThisAddr);
  };

  if (queueAddr != 0 && !sequencerItemFifo.contains(queueAddr)) {
    // For pull-imps, get_parent() resolves to the owning sequencer component.
    // get_comp() resolves to the proxy component and is only a fallback.
    (void)resolvePortOwnerByName("uvm_pkg::uvm_port_base::get_parent",
                                 queueAddr);
    if (!sequencerItemFifo.contains(queueAddr))
      (void)resolvePortOwnerByName("uvm_pkg::uvm_port_base::get_comp",
                                   queueAddr);
  }

  if (queueAddr != 0 && !sequencerItemFifo.contains(queueAddr)) {
    uint64_t vtableAddr = 0;
    bool hasVtableAddr = false;
    uint64_t portOff = 0;
    MemoryBlock *portBlock =
        findMemoryBlockByAddress(queueAddr, procId, &portOff);
    if (!portBlock)
      portBlock = findBlockByAddress(queueAddr, portOff);
    if (portBlock && portBlock->initialized &&
        portOff + 12 <= portBlock->size) {
      for (unsigned i = 0; i < 8; ++i)
        vtableAddr |=
            static_cast<uint64_t>(portBlock->bytes()[portOff + 4 + i]) << (i * 8);
      hasVtableAddr = true;
    }
    if (!hasVtableAddr) {
      uint64_t nativeOff = 0;
      size_t nativeSize = 0;
      if (findNativeMemoryBlockByAddress(queueAddr, &nativeOff, &nativeSize) &&
          nativeOff + 12 <= nativeSize) {
        std::memcpy(&vtableAddr, reinterpret_cast<void *>(queueAddr + 4), 8);
        hasVtableAddr = true;
      }
    }
    if (hasVtableAddr) {
      auto globalIt = addressToGlobal.find(vtableAddr);
      if (globalIt != addressToGlobal.end()) {
        auto vtableBlockIt = globalMemoryBlocks.find(globalIt->second);
        if (vtableBlockIt != globalMemoryBlocks.end()) {
          auto &vtableBlock = vtableBlockIt->second;
          auto resolveVtableSlot = [&](unsigned slot) -> bool {
            unsigned slotOffset = slot * 8;
            if (slotOffset + 8 > vtableBlock.size)
              return false;
            uint64_t methodAddr = 0;
            for (unsigned i = 0; i < 8; ++i)
              methodAddr |=
                  static_cast<uint64_t>(vtableBlock[slotOffset + i])
                  << (i * 8);
            auto fnIt = addressToFunction.find(methodAddr);
            if (fnIt == addressToFunction.end() || !rootModule)
              return false;
            auto ownerFunc = rootModule.lookupSymbol<func::FuncOp>(fnIt->second);
            if (!ownerFunc)
              return false;
            return resolvePortOwner(ownerFunc, queueAddr);
          };

          (void)resolveVtableSlot(/*get_parent slot=*/12);
          if (!sequencerItemFifo.contains(queueAddr))
            (void)resolveVtableSlot(/*get_comp slot=*/13);
        }
      }
    }
  }

  auto [finalAddr, finalStrongHint] = promoteToSequencerQueue(queueAddr);
  if (finalAddr != 0 && finalAddr != queueAddr)
    strongHint = true;
  queueAddr = finalAddr;
  strongHint = strongHint || finalStrongHint;
  if (queueAddr == 0)
    return false;
  if (queueAddr != 0 && sequencerItemFifo.contains(queueAddr))
    strongHint = true;
  return strongHint;
}

bool LLHDProcessInterpreter::resolveUvmSequencerQueueAddress(
    ProcessId procId, uint64_t portAddr, Operation *callSite,
    uint64_t &queueAddr) {
  queueAddr = 0;
  if (portAddr == 0)
    return false;

  bool traceResolve = false;
  if (traceSeqEnabled && traceSeqResolveLimit > 0) {
    uint32_t &printed = traceSeqResolvePrints[portAddr];
    if (printed < traceSeqResolveLimit) {
      ++printed;
      traceResolve = true;
      llvm::errs() << "[SEQ-RESOLVE] port=0x"
                   << llvm::format_hex(portAddr, 16)
                   << " fifo_maps=" << sequencerItemFifo.size() << "\n";
    }
  }

  if (lookupUvmSequencerQueueCache(portAddr, queueAddr)) {
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] cache-hit queue=0x"
                   << llvm::format_hex(queueAddr, 16) << "\n";
    (void)canonicalizeUvmSequencerQueueAddress(procId, queueAddr, callSite);
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] cache-hit canonical queue=0x"
                   << llvm::format_hex(queueAddr, 16)
                   << " in_fifo=" << (sequencerItemFifo.contains(queueAddr) ? 1 : 0)
                   << "\n";
    return queueAddr != 0;
  }

  llvm::SmallVector<uint64_t, 8> portLookupKeys;
  llvm::DenseSet<uint64_t> seenPortLookupKeys;
  auto addPortLookupKey = [&](uint64_t key) {
    if (key != 0 && seenPortLookupKeys.insert(key).second)
      portLookupKeys.push_back(key);
  };
  addPortLookupKey(portAddr);

  uint64_t portOwnerAddr = canonicalizeUvmObjectAddress(procId, portAddr);
  addPortLookupKey(portOwnerAddr);
  if (traceResolve)
    llvm::errs() << "[SEQ-RESOLVE] owner=0x"
                 << llvm::format_hex(portOwnerAddr, 16) << "\n";

  llvm::SmallVector<std::pair<uint64_t, uint64_t>, 8> ownerMatchedKeys;
  if (portOwnerAddr != 0) {
    for (const auto &entry : analysisPortConnections) {
      uint64_t key = entry.first;
      if (key == 0 || key == portAddr || key == portOwnerAddr)
        continue;
      if (canonicalizeUvmObjectAddress(procId, key) != portOwnerAddr)
        continue;
      uint64_t distance = key > portAddr ? key - portAddr : portAddr - key;
      ownerMatchedKeys.push_back({distance, key});
    }
    llvm::sort(ownerMatchedKeys, [](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    for (const auto &entry : ownerMatchedKeys)
      addPortLookupKey(entry.second);
  }

  if (traceResolve) {
    llvm::errs() << "[SEQ-RESOLVE] lookup_keys=";
    for (uint64_t key : portLookupKeys)
      llvm::errs() << " 0x" << llvm::format_hex(key, 16);
    llvm::errs() << "\n";
  }

  llvm::SmallVector<uint64_t, 4> terminals;

  bool hasStrongHint = false;
  for (uint64_t portLookupKey : portLookupKeys) {
    uint64_t candidate = portLookupKey;
    bool candidateStrong =
        canonicalizeUvmSequencerQueueAddress(procId, candidate, callSite);
    bool inFifo = candidate != 0 && sequencerItemFifo.contains(candidate);
    bool changed = candidate != 0 && candidate != portLookupKey;
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] direct key=0x"
                   << llvm::format_hex(portLookupKey, 16)
                   << " -> candidate=0x" << llvm::format_hex(candidate, 16)
                   << " strong=" << (candidateStrong ? 1 : 0)
                   << " changed=" << (changed ? 1 : 0)
                   << " in_fifo=" << (inFifo ? 1 : 0) << "\n";
    // Ignore no-op self mappings when they are not backed by active FIFO
    // state; these are commonly unresolved raw port object pointers.
    if (candidate == 0 || (!candidateStrong && !inFifo && !changed))
      continue;
    if (queueAddr == 0) {
      queueAddr = candidate;
      hasStrongHint = candidateStrong;
    } else if (!hasStrongHint && candidateStrong) {
      queueAddr = candidate;
      hasStrongHint = true;
    }
    if (inFifo) {
      queueAddr = candidate;
      if (traceResolve)
        llvm::errs() << "[SEQ-RESOLVE] select direct queue=0x"
                     << llvm::format_hex(queueAddr, 16) << "\n";
      return true;
    }
  }

  for (uint64_t portLookupKey : portLookupKeys) {
    collectNativeUvmPortTerminals(analysisPortConnections, portLookupKey, terminals);
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] key=0x"
                   << llvm::format_hex(portLookupKey, 16)
                   << " terminals=" << terminals.size() << "\n";
    for (uint64_t terminalAddr : terminals) {
      uint64_t candidate = terminalAddr;
      bool candidateStrong =
          canonicalizeUvmSequencerQueueAddress(procId, candidate, callSite);
      if (traceResolve)
        llvm::errs() << "[SEQ-RESOLVE]   terminal=0x"
                     << llvm::format_hex(terminalAddr, 16) << " -> candidate=0x"
                     << llvm::format_hex(candidate, 16)
                     << " strong=" << (candidateStrong ? 1 : 0)
                     << " in_fifo="
                     << (sequencerItemFifo.contains(candidate) ? 1 : 0) << "\n";
      if (candidate == 0)
        continue;
      if (queueAddr == 0) {
        queueAddr = candidate;
        hasStrongHint = candidateStrong;
      } else if (!hasStrongHint && candidateStrong) {
        queueAddr = candidate;
        hasStrongHint = true;
      }
      if (candidateStrong && sequencerItemFifo.contains(candidate)) {
        queueAddr = candidate;
        if (traceResolve)
          llvm::errs() << "[SEQ-RESOLVE] select terminal queue=0x"
                       << llvm::format_hex(queueAddr, 16) << "\n";
        return true;
      }
    }
  }
  if (queueAddr != 0) {
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] select weak queue=0x"
                   << llvm::format_hex(queueAddr, 16)
                   << " strong=" << (hasStrongHint ? 1 : 0)
                   << " in_fifo=" << (sequencerItemFifo.contains(queueAddr) ? 1 : 0)
                   << "\n";
    return hasStrongHint;
  }

  for (uint64_t portLookupKey : portLookupKeys) {
    auto connIt = analysisPortConnections.find(portLookupKey);
    if (connIt == analysisPortConnections.end() || connIt->second.empty())
      continue;
    queueAddr = connIt->second.back();
    bool strong =
        canonicalizeUvmSequencerQueueAddress(procId, queueAddr, callSite);
    if (traceResolve)
      llvm::errs() << "[SEQ-RESOLVE] fallback key=0x"
                   << llvm::format_hex(portLookupKey, 16) << " queue=0x"
                   << llvm::format_hex(queueAddr, 16)
                   << " strong=" << (strong ? 1 : 0)
                   << " in_fifo=" << (sequencerItemFifo.contains(queueAddr) ? 1 : 0)
                   << "\n";
    if (strong)
      return true;
  }
  if (traceResolve)
    llvm::errs() << "[SEQ-RESOLVE] miss\n";
  return false;
}

void LLHDProcessInterpreter::seedAnalysisPortConnectionWorklist(
    ProcessId procId, uint64_t portAddr,
    llvm::SmallVectorImpl<uint64_t> &worklist) {
  if (portAddr == 0)
    return;

  llvm::DenseSet<uint64_t> visitedProvider;
  auto appendProviders = [&](uint64_t keyAddr) {
    auto it = analysisPortConnections.find(keyAddr);
    if (it == analysisPortConnections.end())
      return;
    for (uint64_t providerAddr : it->second)
      if (providerAddr != 0 && visitedProvider.insert(providerAddr).second)
        worklist.push_back(providerAddr);
  };

  appendProviders(portAddr);

  uint64_t canonicalPort = canonicalizeUvmObjectAddress(procId, portAddr);
  if (canonicalPort != portAddr)
    appendProviders(canonicalPort);

  for (const auto &entry : analysisPortConnections) {
    uint64_t keyAddr = entry.first;
    if (keyAddr == portAddr || keyAddr == canonicalPort)
      continue;
    uint64_t canonicalKey = canonicalizeUvmObjectAddress(procId, keyAddr);
    if (canonicalKey == canonicalPort)
      appendProviders(keyAddr);
  }
}

//===----------------------------------------------------------------------===//
// Stub implementations for functions declared but not yet defined
//===----------------------------------------------------------------------===//

static unsigned writeConfigDbBytesUvm(MemoryBlock *blk, uint64_t off,
                                      const std::vector<uint8_t> &data,
                                      unsigned innerBytes, bool zeroFill) {
  unsigned n = std::min(innerBytes, static_cast<unsigned>(data.size()));
  if (off + n > blk->size)
    return 0;
  for (unsigned i = 0; i < n; ++i)
    blk->bytes()[off + i] = data[i];
  if (zeroFill) {
    for (unsigned i = n; i < innerBytes && off + i < blk->size; ++i)
      blk->bytes()[off + i] = 0;
  }
  blk->initialized = true;
  return n;
}

static bool globMatchConfigDbPattern(llvm::StringRef pattern,
                                     llvm::StringRef text) {
  // Preserve UVM legacy behavior where empty pattern matches any string.
  if (pattern.empty())
    return true;

  // Reuse Moore runtime regex/glob matching used by uvm_is_match.
  MooreString expr{const_cast<char *>(pattern.data()),
                   static_cast<int64_t>(pattern.size())};
  MooreString candidate{const_cast<char *>(text.data()),
                        static_cast<int64_t>(text.size())};
  int32_t execRet = 0;
  int32_t ok =
      uvm_re_compexecfree(&expr, &candidate, /*deglob=*/1, &execRet);
  return ok != 0 && execRet >= 0;
}

static bool splitConfigDbKey(llvm::StringRef key, llvm::StringRef &instPattern,
                             llvm::StringRef &fieldName) {
  size_t dotPos = key.rfind('.');
  if (dotPos == llvm::StringRef::npos)
    return false;
  instPattern = key.take_front(dotPos);
  fieldName = key.drop_front(dotPos + 1);
  return true;
}

void LLHDProcessInterpreter::storeConfigDbEntry(llvm::StringRef instName,
                                                llvm::StringRef fieldName,
                                                std::vector<uint8_t> valueData) {
  std::string key = (instName + "." + fieldName).str();
  configDbEntries[key] = std::move(valueData);
  configDbEntryOrder[key] = nextConfigDbEntryOrder++;
}

bool LLHDProcessInterpreter::lookupConfigDbEntry(
    llvm::StringRef instName, llvm::StringRef fieldName,
    const std::vector<uint8_t> *&valueData, std::string *matchedKey) const {
  valueData = nullptr;
  uint64_t bestOrder = 0;
  std::string bestKey;

  auto getOrder = [&](const std::string &key) -> uint64_t {
    auto orderIt = configDbEntryOrder.find(key);
    return orderIt == configDbEntryOrder.end() ? 0 : orderIt->second;
  };

  auto maybeSelect = [&](const std::string &candidateKey,
                         const std::vector<uint8_t> &candidateValue) {
    llvm::StringRef pattern;
    llvm::StringRef storedField;
    if (!splitConfigDbKey(candidateKey, pattern, storedField))
      return;
    if (storedField != fieldName)
      return;
    if (!globMatchConfigDbPattern(pattern, instName))
      return;

    uint64_t order = getOrder(candidateKey);
    if (!valueData || order >= bestOrder) {
      valueData = &candidateValue;
      bestOrder = order;
      bestKey = candidateKey;
    }
  };

  for (const auto &entry : configDbEntries)
    maybeSelect(entry.first, entry.second);

  // Legacy fallback: for names ending in "_x", allow matching "_0", "_1", ...
  if (!valueData && fieldName.size() > 2 && fieldName.back() == 'x' &&
      fieldName[fieldName.size() - 2] == '_') {
    llvm::StringRef baseName = fieldName.drop_back();
    for (const auto &entry : configDbEntries) {
      llvm::StringRef pattern;
      llvm::StringRef storedField;
      if (!splitConfigDbKey(entry.first, pattern, storedField))
        continue;
      if (!globMatchConfigDbPattern(pattern, instName))
        continue;
      if (storedField.size() <= baseName.size())
        continue;
      if (storedField.take_front(baseName.size()) != baseName)
        continue;
      if (!std::isdigit(
              static_cast<unsigned char>(storedField[baseName.size()])))
        continue;
      maybeSelect(entry.first, entry.second);
    }
  }

  if (valueData && matchedKey)
    *matchedKey = bestKey;
  return valueData != nullptr;
}

bool LLHDProcessInterpreter::tryInterceptConfigDbCallIndirect(
    ProcessId procId, mlir::func::CallIndirectOp callIndirectOp,
    llvm::StringRef calleeName,
    llvm::ArrayRef<InterpretedValue> args) {
  // Match config_db_implementation or config_db_default_implementation
  if (!calleeName.contains("config_db") ||
      !calleeName.contains("implementation"))
    return false;
  if (!calleeName.contains("::set") && !calleeName.contains("::get"))
    return false;

  auto readStr = [&](unsigned argIdx) -> std::string {
    if (argIdx >= args.size())
      return "";
    return readMooreStringStruct(procId, args[argIdx]);
  };

  // --- SET ---
  if (calleeName.contains("::set") && !calleeName.contains("set_default") &&
      !calleeName.contains("set_override") &&
      !calleeName.contains("set_anonymous")) {
    if (args.size() >= 5) {
      std::string str1 = readStr(1);
      std::string str2 = readStr(2);
      std::string str3 = readStr(3);

      std::string instName = str2;
      std::string fieldName = str3;
      if (fieldName.empty()) {
        instName = str1;
        fieldName = str2;
      }
      std::string key = instName + "." + fieldName;

      InterpretedValue valueArg = args[4];
      unsigned valueBits = valueArg.getWidth();
      bool truncatedValue = false;
      std::vector<uint8_t> valueData =
          serializeInterpretedValueBytes(valueArg, /*maxBytes=*/1ULL << 20,
                                         &truncatedValue);
      if (traceConfigDbEnabled) {
        llvm::errs() << "[CFG-CI-XFALLBACK-SET] callee=" << calleeName
                     << " key=\"" << key << "\" s1=\"" << str1
                     << "\" s2=\"" << str2 << "\" s3=\"" << str3
                     << "\" entries_before=" << configDbEntries.size() << "\n";
      }
      storeConfigDbEntry(instName, fieldName, std::move(valueData));
      if (traceConfigDbEnabled) {
        llvm::errs() << "[CFG-CI-XFALLBACK-SET] stored key=\"" << key
                     << "\" entries_after=" << configDbEntries.size() << "\n";
        if (truncatedValue) {
          llvm::errs() << "[CFG-CI-XFALLBACK-SET] truncated oversized value payload"
                       << " key=\"" << key << "\" bitWidth=" << valueBits
                       << "\n";
        }
      }
    }
    return true;
  }

  // --- GET ---
  if (calleeName.contains("::get") && !calleeName.contains("get_default")) {
    if (args.size() >= 5 && callIndirectOp.getNumResults() >= 1) {
      std::string str1 = readStr(1);
      std::string str2 = readStr(2);
      std::string str3 = readStr(3);

      std::string instName = str2;
      std::string fieldName = str3;
      if (fieldName.empty()) {
        instName = str1;
        fieldName = str2;
      }
      std::string key = instName + "." + fieldName;
      if (traceConfigDbEnabled) {
        llvm::errs() << "[CFG-CI-XFALLBACK-GET] callee=" << calleeName
                     << " key=\"" << key << "\" s1=\"" << str1
                     << "\" s2=\"" << str2 << "\" s3=\"" << str3
                     << "\" entries=" << configDbEntries.size() << "\n";
      }

      const std::vector<uint8_t> *matchedValue = nullptr;
      std::string matchedKey;
      if (lookupConfigDbEntry(instName, fieldName, matchedValue, &matchedKey)) {
        if (traceConfigDbEnabled) {
          llvm::errs() << "[CFG-CI-XFALLBACK-GET] hit key=\"" << matchedKey
                       << "\" bytes=" << matchedValue->size() << "\n";
        }
        Value outputRef = callIndirectOp.getArgOperands()[4];
        const std::vector<uint8_t> &valueData = *matchedValue;
        Type refType = outputRef.getType();

        if (auto refT = dyn_cast<llhd::RefType>(refType)) {
          Type innerType = refT.getNestedType();
          unsigned innerBits = getTypeWidth(innerType);
          unsigned innerBytes = (innerBits + 7) / 8;
          llvm::APInt valueBits2(innerBits, 0);
          for (unsigned i = 0;
               i < std::min(innerBytes, (unsigned)valueData.size()); ++i)
            valueBits2.insertBits(llvm::APInt(8, valueData[i]), i * 8);
          SignalId sigId2 = resolveSignalId(outputRef);
          if (sigId2 != 0)
            pendingEpsilonDrives[sigId2] = InterpretedValue(valueBits2);
          InterpretedValue refAddr = getValue(procId, outputRef);
          if (!refAddr.isX()) {
            uint64_t addr = refAddr.getUInt64();
            uint64_t off3 = 0;
            MemoryBlock *blk = findMemoryBlockByAddress(addr, procId, &off3);
            if (blk) {
              writeConfigDbBytesUvm(blk, off3, valueData, innerBytes, true);
            }
          }
        } else if (isa<LLVM::LLVMPointerType>(refType)) {
          if (!args[4].isX()) {
            uint64_t outputAddr = args[4].getUInt64();
            uint64_t outOff = 0;
            MemoryBlock *outBlock =
                findMemoryBlockByAddress(outputAddr, procId, &outOff);
            if (outBlock) {
              writeConfigDbBytesUvm(
                  outBlock, outOff, valueData,
                  static_cast<unsigned>(valueData.size()), false);
            }
          }
        }

        setValue(procId, callIndirectOp.getResult(0),
                InterpretedValue(llvm::APInt(1, 1)));
        return true;
      }

      // Miss — return 0 (not found)
      setValue(procId, callIndirectOp.getResult(0),
              InterpretedValue(llvm::APInt(1, 0)));
      if (traceConfigDbEnabled)
        llvm::errs() << "[CFG-CI-XFALLBACK-GET] miss key=\"" << key << "\"\n";
      return true;
    }
  }

  return false;
}

bool LLHDProcessInterpreter::tryInterceptUvmPortCall(
    ProcessId /*procId*/, llvm::StringRef /*calleeName*/,
    mlir::LLVM::CallOp /*callOp*/) {
  return false;
}

bool LLHDProcessInterpreter::readObjectVTableAddress(uint64_t objectAddr,
                                                      uint64_t &vtableAddr,
                                                      ProcessId procId) {
  vtableAddr = 0;
  if (objectAddr == 0)
    return false;

  auto decodeVtableAt = [&](const uint8_t *bytes, size_t size,
                            uint64_t baseOffset) -> bool {
    // Runtime object header layout: [i32 class_id][ptr vtable_ptr]...
    constexpr uint64_t kHeaderSize = 12; // 4-byte class id + 8-byte ptr
    if (baseOffset + kHeaderSize > size)
      return false;
    uint64_t decoded = 0;
    for (unsigned i = 0; i < 8; ++i)
      decoded |= static_cast<uint64_t>(bytes[baseOffset + 4 + i]) << (i * 8);
    if (decoded == 0)
      return false;
    vtableAddr = decoded;
    return true;
  };

  auto tryDecodeAtAddress = [&](uint64_t addr) -> bool {
    // First try process-visible memory (stack allocas, parent frames, module
    // allocas) when a process context is available.
    if (procId != InvalidProcessId) {
      uint64_t processOffset = 0;
      if (MemoryBlock *block =
              findMemoryBlockByAddress(addr, procId, &processOffset)) {
        if (block->initialized &&
            decodeVtableAt(block->bytes(), block->size, processOffset))
          return true;
      }
    }

    // Then try interpreter-managed global/malloc blocks.
    uint64_t blockOffset = 0;
    if (MemoryBlock *block = findBlockByAddress(addr, blockOffset)) {
      if (block->initialized &&
          decodeVtableAt(block->bytes(), block->size, blockOffset))
        return true;
    }

    // Then try native memory-backed allocations tracked by the interpreter.
    uint64_t nativeOffset = 0;
    size_t nativeSize = 0;
    if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize)) {
      auto *nativeBytes = reinterpret_cast<const uint8_t *>(addr - nativeOffset);
      if (decodeVtableAt(nativeBytes, nativeSize, nativeOffset))
        return true;
    }
    return false;
  };

  auto tryDecodeObject = [&](uint64_t addr) -> bool {
    // UVM methods are often called with base-subobject pointers; canonicalize
    // to the allocation base before decoding the standard object header.
    uint64_t canonicalAddr = addr;
    if (procId != InvalidProcessId)
      canonicalAddr = canonicalizeUvmObjectAddress(procId, addr);
    if (tryDecodeAtAddress(canonicalAddr))
      return true;
    if (canonicalAddr != addr && tryDecodeAtAddress(addr))
      return true;
    return false;
  };

  // Some UVM callsites pass tagged object pointers (low bits set). Probe a
  // small set of low-bit-cleared candidates before giving up.
  const uint64_t candidates[] = {
      objectAddr, objectAddr & ~uint64_t(1), objectAddr & ~uint64_t(3),
      objectAddr & ~uint64_t(7)};
  for (unsigned i = 0; i < 4; ++i) {
    uint64_t candidate = candidates[i];
    if (candidate == 0)
      continue;
    bool duplicate = false;
    for (unsigned j = 0; j < i; ++j) {
      if (candidates[j] == candidate) {
        duplicate = true;
        break;
      }
    }
    if (duplicate)
      continue;
    if (tryDecodeObject(candidate))
      return true;
  }

  return false;
}

void LLHDProcessInterpreter::traceAhbTxnPayload(
    ProcessId /*procId*/, llvm::StringRef /*stage*/,
    llvm::StringRef /*calleeName*/, llvm::StringRef /*impName*/,
    uint64_t /*portAddr*/, uint64_t /*itemAddr*/) {}
