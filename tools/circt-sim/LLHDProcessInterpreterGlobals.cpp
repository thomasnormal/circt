//===- LLHDProcessInterpreterGlobals.cpp - Global lifecycle support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <cstring>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Public Finalization
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::finalizeInit() {
  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: finalizeInit - executing global "
                "constructors after all modules initialized\n");

  // Execute LLVM global constructors (e.g., __moore_global_init_uvm_pkg::uvm_top)
  // This triggers UVM run_test() → build_phase → config_db::get(), so it MUST
  // run AFTER all modules' executeModuleLevelLLVMOps() have completed (which
  // includes hdl_top's initial blocks that call config_db::set()).
  if (failed(executeGlobalConstructors(cachedGlobalOps)))
    return failure();

  inGlobalInit = false;

  // Reset terminationRequested after global init. During UVM initialization,
  // m_uvm_get_root() triggers uvm_fatal → die() → sim.terminate, which sets
  // terminationRequested = true. Without reset, all processes get killed.
  if (terminationRequested) {
    LLVM_DEBUG(llvm::dbgs()
               << "LLHDProcessInterpreter: clearing terminationRequested "
               << "set during global init\n");
    terminationRequested = false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Signal table (signalIdToName=" << signalIdToName.size()
                 << ", valueToSignal=" << valueToSignal.size() << "):\n";
    for (const auto &kv : signalIdToName)
      llvm::dbgs() << "  signal " << kv.first << " = " << kv.second << "\n";
    for (const auto &instKv : instanceValueToSignal) {
      for (const auto &valKv : instKv.second) {
        llvm::dbgs() << "  inst[" << instKv.first << "] signal " << valKv.second;
        if (signalIdToName.count(valKv.second))
          llvm::dbgs() << " = " << signalIdToName[valKv.second];
        llvm::dbgs() << "\n";
      }
    }
  });

  // Set up signal change callback to re-evaluate module drives that depend
  // on signals (via llhd.prb in the combinational chain).
  {
    scheduler.setSignalChangeCallback(
        [this](SignalId sigId, const SignalValue &newVal) {
          // Interface field updates often arrive through reactive signal
          // propagation (scheduler.updateSignal) rather than direct
          // llvm.store operations. Keep interface copy-links and synthetic
          // tri-state rules reactive on every relevant signal change.
          forwardPropagateOnSignalChange(sigId, newVal);
          applyInterfaceTriStateRules(sigId);
          executeModuleDrivesForSignal(sigId);
        });
    LLVM_DEBUG(llvm::dbgs() << "Registered signal change callback for "
                            << signalDependentModuleDrives.size()
                            << " signal-dependent module drive mappings\n");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Iterative Global Discovery
//===----------------------------------------------------------------------===//

void LLHDProcessInterpreter::discoverGlobalOpsIteratively(
    DiscoveredGlobalOps &ops) {
  if (!rootModule)
    return;

  // Use an explicit worklist to traverse operations iteratively
  llvm::SmallVector<Region *, 64> regionWorklist;
  regionWorklist.push_back(&rootModule.getBodyRegion());

  while (!regionWorklist.empty()) {
    Region *region = regionWorklist.pop_back_val();

    for (Block &block : *region) {
      for (Operation &op : block) {
        // Classify global operations
        if (auto globalOp = dyn_cast<LLVM::GlobalOp>(&op)) {
          ops.globals.push_back(globalOp);
        } else if (auto ctorsOp = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ops.ctors.push_back(ctorsOp);
        }

        // Add nested regions (but skip hw.module bodies - we process those separately)
        if (!isa<hw::HWModuleOp>(&op)) {
          for (Region &nestedRegion : op.getRegions()) {
            regionWorklist.push_back(&nestedRegion);
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Global discovery found: "
                          << ops.globals.size() << " globals, "
                          << ops.ctors.size() << " global ctors\n");
}

//===----------------------------------------------------------------------===//
// RTTI Parent Table (for $cast hierarchy checking)
//===----------------------------------------------------------------------===//

void LLHDProcessInterpreter::loadRTTIParentTable() {
  if (rttiTableLoaded)
    return;
  rttiTableLoaded = true;

  if (!rootModule)
    return;

  // Look for the circt.rtti_parent_table module attribute emitted by
  // MooreToCore. It maps typeId -> parentTypeId (0 = root).
  // The attribute is on the builtin.module (rootModule itself), not its parent.
  auto tableAttr =
      rootModule->getAttrOfType<DenseIntElementsAttr>("circt.rtti_parent_table");
  if (!tableAttr)
    return;

  rttiParentTable.clear();
  for (auto val : tableAttr.getValues<int32_t>())
    rttiParentTable.push_back(val);

  LLVM_DEBUG(llvm::dbgs() << "Loaded RTTI parent table with "
                          << rttiParentTable.size() << " entries\n");
}

bool LLHDProcessInterpreter::checkRTTICast(int32_t srcTypeId,
                                            int32_t targetTypeId) {
  if (srcTypeId == 0 || targetTypeId == 0)
    return false;
  if (srcTypeId == targetTypeId)
    return true;

  // Load the RTTI table on first use
  loadRTTIParentTable();

  // If we have a hierarchy table, walk the parent chain
  if (!rttiParentTable.empty()) {
    int32_t current = srcTypeId;
    // Guard against infinite loops with a max depth
    for (int i = 0; i < 1000 && current != 0; ++i) {
      if (current < 0 || current >= static_cast<int32_t>(rttiParentTable.size()))
        break;
      current = rttiParentTable[current];
      if (current == targetTypeId)
        return true;
    }
    return false;
  }

  // Fallback: use the simple >= heuristic (backward compat for old MLIR files)
  return srcTypeId >= targetTypeId;
}

//===----------------------------------------------------------------------===//
// Global Variable and VTable Support
//===----------------------------------------------------------------------===//

LogicalResult
LLHDProcessInterpreter::initializeGlobals(const DiscoveredGlobalOps &globalOps) {
  if (!rootModule)
    return success();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initializing globals\n");

  // Process all pre-discovered LLVM global operations (no walk() needed)
  for (LLVM::GlobalOp globalOp : globalOps.globals) {
    StringRef globalName = globalOp.getSymName();

    // Skip globals already allocated (dual-top: initializeGlobals is called
    // per-module but globals are shared; re-allocating would invalidate
    // addresses captured by the first module's addressof ops).
    if (globalAddresses.count(globalName))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "  Found global: " << globalName << "\n");

    // Get the global's type to calculate size
    Type globalType = globalOp.getGlobalType();
    unsigned size = getLLVMTypeSizeForGEP(globalType);
    if (size == 0)
      size = 8; // Default minimum size

    // Allocate memory for the global
    uint64_t addr = nextGlobalAddress;
    nextGlobalAddress += ((size + 7) / 8) * 8; // Align to 8 bytes

    globalAddresses[globalName] = addr;
    addrRangeIndexDirty = true;

    // Also populate the reverse map for address-to-global lookup
    addressToGlobal[addr] = globalName.str();

    // Create memory block
    MemoryBlock block(size, 64);

    // Check the initializer attribute
    // Handle both #llvm.zero and string constant initializers
    if (auto initAttr = globalOp.getValueOrNull()) {
      block.initialized = true;

      // Check if this is a string initializer
      if (auto strAttr = dyn_cast<StringAttr>(initAttr)) {
        StringRef strContent = strAttr.getValue();
        // Copy the string content to the memory block
        size_t copyLen = std::min(strContent.size(), block.data.size());
        std::memcpy(block.data.data(), strContent.data(), copyLen);
        LLVM_DEBUG(llvm::dbgs() << "    Initialized with string: \""
                                << strContent << "\" (" << copyLen << " bytes)\n");
      } else {
        // For #llvm.zero or other initializers, data is already zeroed
        LLVM_DEBUG(llvm::dbgs() << "    Initialized to zero\n");
      }
    }

    // Check if this is a vtable (has circt.vtable_entries attribute)
    if (auto vtableEntriesAttr = globalOp->getAttr("circt.vtable_entries")) {
      LLVM_DEBUG(llvm::dbgs() << "    This is a vtable with entries\n");

      if (auto entriesArray = dyn_cast<ArrayAttr>(vtableEntriesAttr)) {
        // Each entry is [index, funcSymbol]
        for (auto entry : entriesArray) {
          if (auto entryArray = dyn_cast<ArrayAttr>(entry)) {
            if (entryArray.size() >= 2) {
              auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
              auto funcSymbol = dyn_cast<FlatSymbolRefAttr>(entryArray[1]);

              if (indexAttr && funcSymbol) {
                unsigned index = indexAttr.getInt();
                StringRef funcName = funcSymbol.getValue();

                // Create a unique "function address" for this function
                // We use a simple scheme: high bits identify it as a function ptr
                uint64_t funcAddr = 0xF0000000 + addressToFunction.size();
                addressToFunction[funcAddr] = funcName.str();

                // Store the function address in the vtable memory
                // (little-endian)
                for (unsigned i = 0; i < 8 && (index * 8 + i) < block.data.size(); ++i) {
                  block.data[index * 8 + i] = (funcAddr >> (i * 8)) & 0xFF;
                }
                block.initialized = true;

                LLVM_DEBUG(llvm::dbgs() << "      Entry " << index << ": "
                                        << funcName << " -> 0x"
                                        << llvm::format_hex(funcAddr, 16) << "\n");
              }
            }
          }
        }
      }
    }

    globalMemoryBlocks[globalName] = std::move(block);
  }

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initialized "
                          << globalMemoryBlocks.size() << " globals, "
                          << addressToFunction.size() << " vtable entries\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::executeGlobalConstructors(
    const DiscoveredGlobalOps &globalOps) {
  if (!rootModule)
    return success();

  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: Executing global constructors\n");

  // Collect all constructor entries with their priorities from pre-discovered ops
  SmallVector<std::pair<int32_t, StringRef>, 4> ctorEntries;

  for (LLVM::GlobalCtorsOp ctorsOp : globalOps.ctors) {
    ArrayAttr ctors = ctorsOp.getCtors();
    ArrayAttr priorities = ctorsOp.getPriorities();

    for (auto [ctorAttr, priorityAttr] : llvm::zip(ctors, priorities)) {
      auto ctorRef = cast<FlatSymbolRefAttr>(ctorAttr);
      auto priority = cast<IntegerAttr>(priorityAttr).getInt();
      ctorEntries.emplace_back(priority, ctorRef.getValue());
      LLVM_DEBUG(llvm::dbgs() << "  Found constructor: " << ctorRef.getValue()
                              << " (priority " << priority << ")\n");
    }
  }

  if (ctorEntries.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No global constructors found\n");
    return success();
  }

  // Sort by priority (lower priority values execute first)
  llvm::sort(ctorEntries,
             [](const auto &a, const auto &b) { return a.first < b.first; });

  // Create a temporary process state for executing constructors.
  // Must use a non-zero ID because InvalidProcessId == 0 and
  // findMemoryBlockByAddress's walk loop skips process ID 0.
  ProcessExecutionState tempState;
  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId) || tempProcId == InvalidProcessId)
    tempProcId = nextTempProcId++;
  processStates[tempProcId] = std::move(tempState);

  // Execute each constructor in priority order
  for (auto &[priority, ctorName] : ctorEntries) {
    LLVM_DEBUG(llvm::dbgs() << "  Calling constructor: " << ctorName
                            << " (priority " << priority << ")\n");

    // Reset the temporary process state between constructors.
    // Global constructors are independent; if one sets halted/waiting
    // (e.g., due to an X vtable dispatch that triggers llvm.unreachable),
    // subsequent constructors must not inherit that state.
    {
      auto &ts = processStates[tempProcId];
      ts.halted = false;
      ts.waiting = false;
    }

    // Look up the LLVM function
    auto funcOp = rootModule.lookupSymbol<LLVM::LLVMFuncOp>(ctorName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: constructor function '"
                              << ctorName << "' not found\n");
      continue;
    }

    // Call the constructor with no arguments
    SmallVector<InterpretedValue, 2> results;
    if (failed(interpretLLVMFuncBody(tempProcId, funcOp, {}, results))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Warning: failed to execute constructor '" << ctorName
                 << "'\n");
      // Continue with other constructors even if one fails
    }
  }

  // Clean up the temporary process state
  processStates.erase(tempProcId);

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executed "
                          << ctorEntries.size() << " global constructors\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMAddressOf(
    ProcessId procId, LLVM::AddressOfOp addrOfOp) {
  StringRef globalName = addrOfOp.getGlobalName();

  // Look up the global's address
  auto it = globalAddresses.find(globalName);
  if (it == globalAddresses.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.addressof: global '" << globalName
                            << "' not found, returning X\n");
    setValue(procId, addrOfOp.getResult(),
             InterpretedValue::makeX(64));
    return success();
  }

  uint64_t addr = it->second;
  setValue(procId, addrOfOp.getResult(), InterpretedValue(addr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.addressof: " << globalName << " = 0x"
                          << llvm::format_hex(addr, 16) << "\n");

  return success();
}
