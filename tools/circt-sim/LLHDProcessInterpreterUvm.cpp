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
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

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

    return {candidateAddr, false};
  };

  auto [promotedAddr, promotedStrongHint] = promoteToSequencerQueue(queueAddr);
  queueAddr = promotedAddr;
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
    if (resolvedQueueAddr == 0 || !sequencerItemFifo.contains(resolvedQueueAddr))
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
        portOff + 12 <= portBlock->data.size()) {
      for (unsigned i = 0; i < 8; ++i)
        vtableAddr |=
            static_cast<uint64_t>(portBlock->data[portOff + 4 + i]) << (i * 8);
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
                  static_cast<uint64_t>(vtableBlock.data[slotOffset + i])
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
  if (finalAddr != queueAddr)
    strongHint = true;
  queueAddr = finalAddr;
  strongHint = strongHint || finalStrongHint;
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
  if (off + n > blk->data.size())
    return 0;
  for (unsigned i = 0; i < n; ++i)
    blk->data[off + i] = data[i];
  if (zeroFill) {
    for (unsigned i = n; i < innerBytes && off + i < blk->data.size(); ++i)
      blk->data[off + i] = 0;
  }
  blk->initialized = true;
  return n;
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
      unsigned valueBytes = (valueBits + 7) / 8;
      std::vector<uint8_t> valueData(valueBytes, 0);
      if (!valueArg.isX()) {
        llvm::APInt valBits = valueArg.getAPInt();
        for (unsigned i = 0; i < valueBytes; ++i)
          valueData[i] = static_cast<uint8_t>(
              valBits.extractBits(8, i * 8).getZExtValue());
      }
      if (traceConfigDbEnabled) {
        llvm::errs() << "[CFG-CI-XFALLBACK-SET] callee=" << calleeName
                     << " key=\"" << key << "\" s1=\"" << str1
                     << "\" s2=\"" << str2 << "\" s3=\"" << str3
                     << "\" entries_before=" << configDbEntries.size() << "\n";
      }
      configDbEntries[key] = std::move(valueData);
      if (traceConfigDbEnabled) {
        llvm::errs() << "[CFG-CI-XFALLBACK-SET] stored key=\"" << key
                     << "\" entries_after=" << configDbEntries.size() << "\n";
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

      auto it = configDbEntries.find(key);
      // Wildcard match: look for entries where field name matches
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
      // Fuzzy match: "bfm_x" matches "bfm_0"
      if (it == configDbEntries.end() && fieldName.size() > 2 &&
          fieldName.back() == 'x' &&
          fieldName[fieldName.size() - 2] == '_') {
        std::string baseName = fieldName.substr(0, fieldName.size() - 1);
        for (auto &[k, v] : configDbEntries) {
          size_t dotPos = k.rfind('.');
          if (dotPos != std::string::npos) {
            std::string storedField = k.substr(dotPos + 1);
            if (storedField.size() > baseName.size() &&
                storedField.substr(0, baseName.size()) == baseName &&
                std::isdigit(storedField[baseName.size()])) {
              it = configDbEntries.find(k);
              break;
            }
          }
        }
      }

      if (it != configDbEntries.end()) {
        if (traceConfigDbEnabled) {
          llvm::errs() << "[CFG-CI-XFALLBACK-GET] hit key=\"" << it->first
                       << "\" bytes=" << it->second.size() << "\n";
        }
        Value outputRef = callIndirectOp.getArgOperands()[4];
        const std::vector<uint8_t> &valueData = it->second;
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

bool LLHDProcessInterpreter::readObjectVTableAddress(uint64_t /*objectAddr*/,
                                                      uint64_t &/*vtableAddr*/) {
  return false;
}

void LLHDProcessInterpreter::traceAhbTxnPayload(
    ProcessId /*procId*/, llvm::StringRef /*stage*/,
    llvm::StringRef /*calleeName*/, llvm::StringRef /*impName*/,
    uint64_t /*portAddr*/, uint64_t /*itemAddr*/) {}
