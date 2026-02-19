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
#include "circt/Runtime/MooreRuntime.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

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
  if (start >= block->data.size())
    return 0;

  size_t availableBytes = block->data.size() - start;
  unsigned maxWritable =
      static_cast<unsigned>(std::min<size_t>(requestedBytes, availableBytes));
  unsigned copyBytes =
      std::min(maxWritable, static_cast<unsigned>(valueData.size()));

  if (copyBytes > 0)
    std::memcpy(block->data.data() + start, valueData.data(), copyBytes);
  if (zeroFillMissing && maxWritable > copyBytes)
    std::memset(block->data.data() + start + copyBytes, 0,
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
    bool sawResolvedTarget = false;
    std::string resolvedTargetName;
    auto noteResolvedTarget = [&](llvm::StringRef name) {
      if (name.empty())
        return;
      sawResolvedTarget = true;
      resolvedTargetName = name.str();
    };
    auto noteRuntimeIndirectProfileOnExit = llvm::make_scope_exit([&]() {
      if (!jitRuntimeIndirectProfileEnabled || !compileModeEnabled)
        return;
      if (sawResolvedTarget)
        noteJitRuntimeIndirectResolvedTarget(procId, callIndirectOp,
                                            resolvedTargetName);
      else
        noteJitRuntimeIndirectUnresolved(procId, callIndirectOp);
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

    if (funcPtrVal.isX()) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: callee is X "
                              << "(uninitialized vtable pointer)\n");

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
            uint64_t vtableOff = 0;
            MemoryBlock *objBlock = findBlockByAddress(objAddr, vtableOff);
            // Vtable ptr is at byte offset 4 (after i32 class ID at offset 0)
            if (objBlock && objBlock->initialized &&
                objBlock->data.size() >= vtableOff + 12) {
              uint64_t runtimeVtableAddr = 0;
              for (unsigned i = 0; i < 8; ++i)
                runtimeVtableAddr |= static_cast<uint64_t>(
                                         objBlock->data[vtableOff + 4 + i])
                                     << (i * 8);
              auto globalIt2 = addressToGlobal.find(runtimeVtableAddr);
              if (globalIt2 != addressToGlobal.end()) {
                std::string runtimeVtableName = globalIt2->second;
                if (runtimeVtableName != vtableGlobalName &&
                    globalMemoryBlocks.count(runtimeVtableName)) {
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
              static_cast<uint64_t>(vtableBlock.data[slotOffset + i]) << (i * 8);

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
          uint64_t phaseAddr = args[0].getUInt64();
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
          if (resolvedName.contains("raise_objection")) {
            raisePhaseObjection(handle, count);
          } else {
            dropPhaseObjection(handle, count);
          }
          resolved = true;
          break;
        }

        // Record port connect() in X-fallback path and bypass UVM body.
        // This avoids uvm_port_base phase checks rejecting valid native
        // connections as "late" while still preserving routing info.
        if (resolvedName.contains("uvm_port_base") &&
            resolvedName.contains("::connect") &&
            !resolvedName.contains("connect_phase") && args.size() >= 2) {
          uint64_t selfAddr2 = args[0].isX() ? 0 : args[0].getUInt64();
          uint64_t providerAddr2 = args[1].isX() ? 0 : args[1].getUInt64();
          if (selfAddr2 != 0 && providerAddr2 != 0) {
            auto &conns = analysisPortConnections[selfAddr2];
            if (std::find(conns.begin(), conns.end(), providerAddr2) ==
                conns.end()) {
              conns.push_back(providerAddr2);
              invalidateUvmSequencerQueueCache(selfAddr2);
            }
          }
          // [SEQ-CONN] X-fallback connect diagnostic removed
          resolved = true;
          break;
        }

        // Intercept analysis_port::write in X-fallback path.
        // The UVM write() body iterates m_imp_list via get_if(i), but
        // m_if is empty because we skip resolve_bindings. Use our native
        // analysisPortConnections map to dispatch to terminal imps.
        if (resolvedName.contains("analysis_port") &&
            resolvedName.contains("::write") &&
            !resolvedName.contains("write_m_") && args.size() >= 2) {
          uint64_t portAddr = args[0].isX() ? 0 : args[0].getUInt64();
          if (traceAnalysisEnabled)
            llvm::errs() << "[ANALYSIS-WRITE-XFALLBACK] " << resolvedName
                         << " portAddr=0x" << llvm::format_hex(portAddr, 0)
                         << " inMap=" << analysisPortConnections.count(portAddr)
                         << "\n";
          // Flatten the connection chain to find all terminal imps.
          llvm::SmallVector<uint64_t, 4> terminals;
          llvm::SmallVector<uint64_t, 8> worklist;
          llvm::DenseSet<uint64_t> visited;
          auto seedIt = analysisPortConnections.find(portAddr);
          if (seedIt != analysisPortConnections.end()) {
            for (uint64_t a : seedIt->second)
              worklist.push_back(a);
          }
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
              // Read vtable pointer at byte offset 4 from the imp object.
              uint64_t vtableOff2 = 0;
              MemoryBlock *impBlock = findBlockByAddress(impAddr, vtableOff2);
              if (!impBlock || vtableOff2 + 4 + 8 > impBlock->size)
                continue;
              uint64_t vtableAddr2 = 0;
              for (unsigned i = 0; i < 8; ++i)
                vtableAddr2 |= static_cast<uint64_t>(
                                   impBlock->data[vtableOff2 + 4 + i])
                               << (i * 8);
              auto globalIt2 = addressToGlobal.find(vtableAddr2);
              if (globalIt2 == addressToGlobal.end())
                continue;
              // Read write function pointer from vtable slot 11.
              auto vtableBlockIt = globalMemoryBlocks.find(globalIt2->second);
              if (vtableBlockIt == globalMemoryBlocks.end())
                continue;
              auto &vtableBlock2 = vtableBlockIt->second;
              unsigned writeSlot = 11;
              unsigned slotOff = writeSlot * 8;
              if (slotOff + 8 > vtableBlock2.size)
                continue;
              uint64_t writeFuncAddr = 0;
              for (unsigned i = 0; i < 8; ++i)
                writeFuncAddr |=
                    static_cast<uint64_t>(vtableBlock2.data[slotOff + i])
                    << (i * 8);
              auto funcIt2 = addressToFunction.find(writeFuncAddr);
              if (funcIt2 == addressToFunction.end())
                continue;
              auto impWriteFunc = moduleOp.lookupSymbol<func::FuncOp>(
                  funcIt2->second);
              if (!impWriteFunc)
                continue;
              if (traceAnalysisEnabled)
                llvm::errs() << "[ANALYSIS-WRITE-XFALLBACK] dispatching to "
                             << funcIt2->second << "\n";
              SmallVector<InterpretedValue, 2> impArgs;
              impArgs.push_back(InterpretedValue(llvm::APInt(64, impAddr)));
              impArgs.push_back(args[1]); // transaction object
              SmallVector<InterpretedValue, 2> impResults;
              auto &cState2 = processStates[procId];
              ++cState2.callDepth;
              (void)interpretFuncBody(procId, impWriteFunc, impArgs, impResults,
                                     callIndirectOp);
              --cState2.callDepth;
            }
            resolved = true;
            break;
          }
          // If no native connections, fall through to normal UVM body dispatch.
        }

        // Intercept resolve_bindings in X-fallback path — skip it.
        if (resolvedName.contains("uvm_port_base") &&
            resolvedName.contains("::resolve_bindings")) {
          resolved = true;
          break;
        }

        // Dispatch the call
        // [SEQ-XFALLBACK] diagnostic removed
        auto &callState = processStates[procId];
        ++callState.callDepth;
        SmallVector<InterpretedValue, 4> results;
        auto callResult = interpretFuncBody(procId, funcOp, args, results,
                                            callIndirectOp);
        --callState.callDepth;

        if (failed(callResult)) {
          break;
        }

        // Set return values
        for (auto [result, val] :
             llvm::zip(callIndirectOp.getResults(), results))
          setValue(procId, result, val);

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
            uint64_t vtableOff2 = 0;
            MemoryBlock *objBlock2 = findBlockByAddress(objAddr2, vtableOff2);
            if (objBlock2 && objBlock2->initialized &&
                objBlock2->data.size() >= vtableOff2 + 12) {
              uint64_t runtimeVtableAddr2 = 0;
              for (unsigned i = 0; i < 8; ++i)
                runtimeVtableAddr2 |=
                    static_cast<uint64_t>(
                        objBlock2->data[vtableOff2 + 4 + i])
                    << (i * 8);
              auto globalIt3 = addressToGlobal.find(runtimeVtableAddr2);
              if (globalIt3 != addressToGlobal.end()) {
                std::string runtimeVtableName2 = globalIt3->second;
                if (runtimeVtableName2 != vtableGlobalName &&
                    globalMemoryBlocks.count(runtimeVtableName2)) {
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
              static_cast<uint64_t>(vtableBlock.data[slotOffset + i])
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

        // Intercept UVM phase/objection methods in non-X static fallback.
        if ((resolvedName.contains("uvm_phase::raise_objection") ||
             resolvedName.contains("uvm_phase::drop_objection")) &&
            !resolvedName.contains("phase_hopper") &&
            !sArgs.empty() && !sArgs[0].isX()) {
          uint64_t phaseAddr = sArgs[0].getUInt64();
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
          if (resolvedName.contains("raise_objection")) {
            raisePhaseObjection(handle, cnt);
          } else {
            dropPhaseObjection(handle, cnt);
          }
          staticResolved = true;
          break;
        }

        // Record port connect() in static fallback path and bypass UVM body.
        // This mirrors the direct call_indirect connect interceptor behavior.
        if (resolvedName.contains("uvm_port_base") &&
            resolvedName.contains("::connect") &&
            !resolvedName.contains("connect_phase") && sArgs.size() >= 2) {
          uint64_t selfAddr3 = sArgs[0].isX() ? 0 : sArgs[0].getUInt64();
          uint64_t providerAddr3 = sArgs[1].isX() ? 0 : sArgs[1].getUInt64();
          if (selfAddr3 != 0 && providerAddr3 != 0) {
            auto &conns = analysisPortConnections[selfAddr3];
            if (std::find(conns.begin(), conns.end(), providerAddr3) ==
                conns.end()) {
              conns.push_back(providerAddr3);
              invalidateUvmSequencerQueueCache(selfAddr3);
            }
          }
          // [SEQ-CONN] static-fallback connect diagnostic removed
          staticResolved = true;
          break;
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
              unsigned valueBytes = (valueBits + 7) / 8;
              std::vector<uint8_t> valueData(valueBytes, 0);
              if (!valueArg.isX()) {
                llvm::APInt valBits = valueArg.getAPInt();
                for (unsigned i = 0; i < valueBytes; ++i)
                  valueData[i] = static_cast<uint8_t>(
                      valBits.extractBits(8, i * 8).getZExtValue());
              }
              configDbEntries[key] = std::move(valueData);
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

        // Intercept config_db implementation in static fallback path.
        // This covers cases where call_indirect cannot resolve a runtime
        // function pointer (e.g. null/uninitialized vtable slot), but the
        // static vtable slot still identifies config_db::set/get.
        if (resolvedName.contains("config_db") &&
            resolvedName.contains("implementation") &&
            (resolvedName.contains("::set") ||
             resolvedName.contains("::get"))) {
          auto readStr2 = [&](unsigned argIdx) -> std::string {
            if (argIdx >= sArgs.size())
              return "";
            return readMooreStringStruct(procId, sArgs[argIdx]);
          };

          if (resolvedName.contains("::set") &&
              !resolvedName.contains("set_default") &&
              !resolvedName.contains("set_override") &&
              !resolvedName.contains("set_anonymous")) {
            if (sArgs.size() >= 5) {
              std::string str1 = readStr2(1);
              std::string str2 = readStr2(2);
              std::string str3 = readStr2(3);

              std::string instName = str2;
              std::string fieldName = str3;
              if (fieldName.empty()) {
                instName = str1;
                fieldName = str2;
              }
              std::string key = instName + "." + fieldName;

              InterpretedValue &valueArg = sArgs[4];
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
                llvm::errs() << "[CFG-CI-STATIC-SET] callee=" << resolvedName
                             << " key=\"" << key << "\" s1=\"" << str1
                             << "\" s2=\"" << str2 << "\" s3=\"" << str3
                             << "\" entries_before=" << configDbEntries.size()
                             << "\n";
              }
              configDbEntries[key] = std::move(valueData);
              if (traceConfigDbEnabled) {
                llvm::errs() << "[CFG-CI-STATIC-SET] stored key=\"" << key
                             << "\" entries_after=" << configDbEntries.size()
                             << "\n";
              }
            }
            staticResolved = true;
            break;
          }

          if (resolvedName.contains("::get") &&
              !resolvedName.contains("get_default") &&
              sArgs.size() >= 5 && callIndirectOp.getNumResults() >= 1) {
            std::string str1 = readStr2(1);
            std::string str2 = readStr2(2);
            std::string str3 = readStr2(3);

            std::string instName = str2;
            std::string fieldName = str3;
            if (fieldName.empty()) {
              instName = str1;
              fieldName = str2;
            }
            std::string key = instName + "." + fieldName;
            if (traceConfigDbEnabled) {
              llvm::errs() << "[CFG-CI-STATIC-GET] callee=" << resolvedName
                           << " key=\"" << key << "\" s1=\"" << str1
                           << "\" s2=\"" << str2 << "\" s3=\"" << str3
                           << "\" entries=" << configDbEntries.size() << "\n";
            }

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
            if (it == configDbEntries.end() && fieldName.size() > 2 &&
                fieldName.back() == 'x' &&
                fieldName[fieldName.size() - 2] == '_') {
              std::string baseName = fieldName.substr(0, fieldName.size() - 1);
              for (auto &[k, v] : configDbEntries) {
                size_t dotPos = k.rfind('.');
                if (dotPos == std::string::npos)
                  continue;
                std::string storedField = k.substr(dotPos + 1);
                if (storedField.size() > baseName.size() &&
                    storedField.substr(0, baseName.size()) == baseName &&
                    std::isdigit(storedField[baseName.size()])) {
                  it = configDbEntries.find(k);
                  break;
                }
              }
            }

            if (it != configDbEntries.end()) {
              if (traceConfigDbEnabled) {
                llvm::errs() << "[CFG-CI-STATIC-GET] hit key=\"" << it->first
                             << "\" bytes=" << it->second.size() << "\n";
              }
              Value outputRef = callIndirectOp.getArgOperands()[4];
              const std::vector<uint8_t> &valueData = it->second;
              Type refType = outputRef.getType();

              if (auto refT = dyn_cast<llhd::RefType>(refType)) {
                Type innerType = refT.getNestedType();
                unsigned innerBits = getTypeWidth(innerType);
                unsigned innerBytes = (innerBits + 7) / 8;
                llvm::APInt valueBits(innerBits, 0);
                for (unsigned i = 0;
                     i < std::min(innerBytes, (unsigned)valueData.size());
                     ++i)
                  safeInsertBits(valueBits, llvm::APInt(8, valueData[i]), i * 8);
                SignalId sigId3 = resolveSignalId(outputRef);
                if (sigId3 != 0)
                  pendingEpsilonDrives[sigId3] = InterpretedValue(valueBits);
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
                if (!sArgs[4].isX()) {
                  uint64_t outputAddr = sArgs[4].getUInt64();
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

            if (traceConfigDbEnabled)
              llvm::errs() << "[CFG-CI-STATIC-GET] miss key=\"" << key
                           << "\"\n";
            setValue(procId, callIndirectOp.getResult(0),
                     InterpretedValue(llvm::APInt(1, 0)));
            staticResolved = true;
            break;
          }
        }

        // Intercept analysis_port::write in non-X static fallback path.
        if (resolvedName.contains("analysis_port") &&
            resolvedName.contains("::write") &&
            !resolvedName.contains("write_m_") && sArgs.size() >= 2) {
          uint64_t portAddr3 = sArgs[0].isX() ? 0 : sArgs[0].getUInt64();
          if (traceAnalysisEnabled)
            llvm::errs() << "[ANALYSIS-WRITE-STATIC] " << resolvedName
                         << " portAddr=0x" << llvm::format_hex(portAddr3, 0)
                         << " inMap=" << analysisPortConnections.count(portAddr3)
                         << "\n";
          llvm::SmallVector<uint64_t, 4> terminals3;
          llvm::SmallVector<uint64_t, 8> worklist3;
          llvm::DenseSet<uint64_t> visited3;
          auto seedIt3 = analysisPortConnections.find(portAddr3);
          if (seedIt3 != analysisPortConnections.end()) {
            for (uint64_t a : seedIt3->second)
              worklist3.push_back(a);
          }
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
              uint64_t vtableOff3 = 0;
              MemoryBlock *impBlock3 = findBlockByAddress(impAddr, vtableOff3);
              if (!impBlock3 || vtableOff3 + 4 + 8 > impBlock3->size)
                continue;
              uint64_t vtableAddr3 = 0;
              for (unsigned i = 0; i < 8; ++i)
                vtableAddr3 |= static_cast<uint64_t>(
                                   impBlock3->data[vtableOff3 + 4 + i])
                               << (i * 8);
              auto gIt = addressToGlobal.find(vtableAddr3);
              if (gIt == addressToGlobal.end())
                continue;
              auto vbIt = globalMemoryBlocks.find(gIt->second);
              if (vbIt == globalMemoryBlocks.end())
                continue;
              auto &vb = vbIt->second;
              unsigned writeSlot3 = 11;
              unsigned slotOff3 = writeSlot3 * 8;
              if (slotOff3 + 8 > vb.size)
                continue;
              uint64_t wfa = 0;
              for (unsigned i = 0; i < 8; ++i)
                wfa |= static_cast<uint64_t>(vb.data[slotOff3 + i]) << (i * 8);
              auto fi = addressToFunction.find(wfa);
              if (fi == addressToFunction.end())
                continue;
              auto iwf = modOp.lookupSymbol<func::FuncOp>(fi->second);
              if (!iwf)
                continue;
              if (traceAnalysisEnabled)
                llvm::errs() << "[ANALYSIS-WRITE-STATIC] dispatching to "
                             << fi->second << "\n";
              SmallVector<InterpretedValue, 2> iArgs;
              iArgs.push_back(InterpretedValue(llvm::APInt(64, impAddr)));
              iArgs.push_back(sArgs[1]);
              SmallVector<InterpretedValue, 2> iRes;
              auto &cs3 = processStates[procId];
              ++cs3.callDepth;
              (void)interpretFuncBody(procId, iwf, iArgs, iRes, callIndirectOp);
              --cs3.callDepth;
            }
            staticResolved = true;
            break;
          }
          // If no native connections, fall through to normal UVM body dispatch.
        }

        // Intercept resolve_bindings in non-X static fallback path.
        if (resolvedName.contains("uvm_port_base") &&
            resolvedName.contains("::resolve_bindings")) {
          staticResolved = true;
          break;
        }

        // [SEQ-UNMAPPED] diagnostic removed
        auto &cs2 = processStates[procId];
        ++cs2.callDepth;
        SmallVector<InterpretedValue, 4> sResults;
        auto callRes = interpretFuncBody(procId, fOp, sArgs, sResults,
                                         callIndirectOp);
        --cs2.callDepth;
        if (failed(callRes))
          break;
        for (auto [result, val] :
             llvm::zip(callIndirectOp.getResults(), sResults))
          setValue(procId, result, val);
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

        auto readUnresolvedStrArg = [&](unsigned argIdx,
                                        llvm::ArrayRef<InterpretedValue> vals)
            -> std::string {
          if (argIdx >= vals.size())
            return "";
          return readMooreStringStruct(procId, vals[argIdx]);
        };

        auto writeConfigDbGetResult = [&](Value outputRef,
                                          const std::vector<uint8_t> &valueData,
                                          InterpretedValue outputArgVal) {
          Type refType = outputRef.getType();
          if (auto refT = dyn_cast<llhd::RefType>(refType)) {
            Type innerType = refT.getNestedType();
            unsigned innerBits = getTypeWidth(innerType);
            unsigned innerBytes = (innerBits + 7) / 8;
            llvm::APInt valueBits(innerBits, 0);
            for (unsigned i = 0;
                 i < std::min(innerBytes, (unsigned)valueData.size()); ++i)
              safeInsertBits(valueBits, llvm::APInt(8, valueData[i]), i * 8);
            SignalId sigId = resolveSignalId(outputRef);
            if (sigId != 0)
              pendingEpsilonDrives[sigId] = InterpretedValue(valueBits);
            InterpretedValue refAddr = getValue(procId, outputRef);
            if (!refAddr.isX()) {
              uint64_t addr = refAddr.getUInt64();
              uint64_t off = 0;
              MemoryBlock *blk = findMemoryBlockByAddress(addr, procId, &off);
              if (!blk)
                blk = findBlockByAddress(addr, off);
              if (blk) {
                writeConfigDbBytesToMemoryBlock(
                    blk, off, valueData, innerBytes,
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
          } else if (isa<LLVM::LLVMPointerType>(refType) && !outputArgVal.isX()) {
            uint64_t outputAddr = outputArgVal.getUInt64();
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
        };

        // Final fallback: unresolved call_indirect heuristic for config_db
        // implementation signatures. This is intentionally narrow and only
        // triggers when vtable resolution failed.
        {
          SmallVector<InterpretedValue, 8> unresolvedArgs;
          unresolvedArgs.reserve(callIndirectOp.getNumOperands());
          for (Value arg : callIndirectOp.getArgOperands())
            unresolvedArgs.push_back(getValue(procId, arg));

          // Heuristic config_db::set
          if (callIndirectOp.getNumResults() == 0 &&
              callIndirectOp.getNumOperands() >= 5 &&
              isa<LLVM::LLVMPointerType>(callIndirectOp.getArgOperands()[0].getType()) &&
              isMooreStringStructType(callIndirectOp.getArgOperands()[1].getType()) &&
              isMooreStringStructType(callIndirectOp.getArgOperands()[2].getType()) &&
              isMooreStringStructType(callIndirectOp.getArgOperands()[3].getType())) {
            std::string str1 = readUnresolvedStrArg(1, unresolvedArgs);
            std::string str2 = readUnresolvedStrArg(2, unresolvedArgs);
            std::string str3 = readUnresolvedStrArg(3, unresolvedArgs);
            std::string instName = str2;
            std::string fieldName = str3;
            if (fieldName.empty()) {
              instName = str1;
              fieldName = str2;
            }
            std::string key = instName + "." + fieldName;

            InterpretedValue &valueArg = unresolvedArgs[4];
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
              llvm::errs() << "[CFG-CI-UNRES-SET] key=\"" << key
                           << "\" s1=\"" << str1 << "\" s2=\"" << str2
                           << "\" s3=\"" << str3
                           << "\" entries_before=" << configDbEntries.size()
                           << "\n";
            }
            configDbEntries[key] = std::move(valueData);
            if (traceConfigDbEnabled) {
              llvm::errs() << "[CFG-CI-UNRES-SET] stored key=\"" << key
                           << "\" entries_after=" << configDbEntries.size()
                           << "\n";
            }
            return success();
          }

          // Heuristic config_db::get
          if (callIndirectOp.getNumResults() == 1 &&
              callIndirectOp.getResult(0).getType().isSignlessInteger(1) &&
              callIndirectOp.getNumOperands() >= 5 &&
              isa<LLVM::LLVMPointerType>(callIndirectOp.getArgOperands()[0].getType()) &&
              isa<LLVM::LLVMPointerType>(callIndirectOp.getArgOperands()[1].getType()) &&
              isMooreStringStructType(callIndirectOp.getArgOperands()[2].getType()) &&
              isMooreStringStructType(callIndirectOp.getArgOperands()[3].getType())) {
            std::string str1 = readUnresolvedStrArg(1, unresolvedArgs);
            std::string str2 = readUnresolvedStrArg(2, unresolvedArgs);
            std::string str3 = readUnresolvedStrArg(3, unresolvedArgs);
            std::string instName = str2;
            std::string fieldName = str3;
            if (fieldName.empty()) {
              instName = str1;
              fieldName = str2;
            }
            std::string key = instName + "." + fieldName;
            if (traceConfigDbEnabled) {
              llvm::errs() << "[CFG-CI-UNRES-GET] key=\"" << key
                           << "\" s1=\"" << str1 << "\" s2=\"" << str2
                           << "\" s3=\"" << str3
                           << "\" entries=" << configDbEntries.size() << "\n";
            }

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
            if (it == configDbEntries.end() && fieldName.size() > 2 &&
                fieldName.back() == 'x' &&
                fieldName[fieldName.size() - 2] == '_') {
              std::string baseName = fieldName.substr(0, fieldName.size() - 1);
              for (auto &[k, v] : configDbEntries) {
                size_t dotPos = k.rfind('.');
                if (dotPos == std::string::npos)
                  continue;
                std::string storedField = k.substr(dotPos + 1);
                if (storedField.size() > baseName.size() &&
                    storedField.substr(0, baseName.size()) == baseName &&
                    std::isdigit(storedField[baseName.size()])) {
                  it = configDbEntries.find(k);
                  break;
                }
              }
            }

            if (it != configDbEntries.end()) {
              if (traceConfigDbEnabled) {
                llvm::errs() << "[CFG-CI-UNRES-GET] hit key=\"" << it->first
                             << "\" bytes=" << it->second.size() << "\n";
              }
              writeConfigDbGetResult(callIndirectOp.getArgOperands()[4],
                                     it->second, unresolvedArgs[4]);
              setValue(procId, callIndirectOp.getResult(0),
                       InterpretedValue(llvm::APInt(1, 1)));
              return success();
            }

            if (traceConfigDbEnabled)
              llvm::errs() << "[CFG-CI-UNRES-GET] miss key=\"" << key
                           << "\"\n";
            setValue(procId, callIndirectOp.getResult(0),
                     InterpretedValue(llvm::APInt(1, 0)));
            return success();
          }
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: address 0x"
                   << llvm::format_hex(funcAddr, 16)
                   << " not in vtable map\n");
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
      uint64_t vtableOff = 0;
      MemoryBlock *objBlock = findBlockByAddress(objAddr, vtableOff);
      if (!objBlock || !objBlock->initialized ||
          objBlock->data.size() < vtableOff + 12)
        break;

      // Read vtable pointer (8 bytes at offset 4, after i32 class_id)
      uint64_t runtimeVtableAddr = 0;
      for (unsigned i = 0; i < 8; ++i)
        runtimeVtableAddr |= static_cast<uint64_t>(
                                 objBlock->data[vtableOff + 4 + i])
                             << (i * 8);
      uint64_t runtimeFuncAddr = 0;
      auto cacheKey = std::make_pair(runtimeVtableAddr, methodIndex);
      auto cacheIt = callIndirectRuntimeVtableSlotCache.find(cacheKey);
      if (cacheIt != callIndirectRuntimeVtableSlotCache.end()) {
        runtimeFuncAddr = cacheIt->second;
        if (traceCallIndirectSiteCacheEnabled) {
          llvm::errs() << "[CI-SITE-CACHE] runtime-slot-hit vtable=0x"
                       << llvm::format_hex(runtimeVtableAddr, 16)
                       << " method_index=" << methodIndex << " func=0x"
                       << llvm::format_hex(runtimeFuncAddr, 16) << "\n";
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
              static_cast<uint64_t>(vtableBlock.data[slotOffset + i])
              << (i * 8);
        callIndirectRuntimeVtableSlotCache[cacheKey] = runtimeFuncAddr;
        if (traceCallIndirectSiteCacheEnabled) {
          llvm::errs() << "[CI-SITE-CACHE] runtime-slot-store vtable=0x"
                       << llvm::format_hex(runtimeVtableAddr, 16)
                       << " method_index=" << methodIndex << " func=0x"
                       << llvm::format_hex(runtimeFuncAddr, 16) << "\n";
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
    if (calleeName == "uvm_pkg::uvm_default_factory::register" ||
        calleeName == "uvm_pkg::uvm_factory::register") {
      bool registered = false;
      do {
        if (callIndirectOp.getArgOperands().size() < 2)
          break;
        InterpretedValue wrapperVal =
            getValue(procId, callIndirectOp.getArgOperands()[1]);
        if (wrapperVal.isX() || wrapperVal.getUInt64() == 0)
          break;
        uint64_t wrapperAddr = wrapperVal.getUInt64();

        // Read wrapper's vtable pointer: struct uvm_void { i32, ptr }
        // vtable ptr is at offset 4 (after i32 __class_handle)
        uint64_t off = 0;
        MemoryBlock *blk = findBlockByAddress(wrapperAddr + 4, off);
        if (!blk || !blk->initialized || off + 8 > blk->data.size())
          break;
        uint64_t vtableAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          vtableAddr |= static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
        if (vtableAddr == 0)
          break;

        // Read vtable entry [2] = get_type_name
        uint64_t off2 = 0;
        MemoryBlock *vtableBlk =
            findBlockByAddress(vtableAddr + 2 * 8, off2);
        if (!vtableBlk || !vtableBlk->initialized ||
            off2 + 8 > vtableBlk->data.size())
          break;
        uint64_t funcAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          funcAddr |=
              static_cast<uint64_t>(vtableBlk->data[off2 + i]) << (i * 8);
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
        if (nameStruct.isX() || nameStruct.getWidth() < 128)
          break;
        APInt nameAPInt = nameStruct.getAPInt();
        uint64_t strAddr = nameAPInt.extractBits(64, 0).getZExtValue();
        uint64_t strLen = nameAPInt.extractBits(64, 64).getZExtValue();
        if (strLen == 0 || strLen > 1024 || strAddr == 0)
          break;

        // Read the string content from the packed string global
        uint64_t strOff = 0;
        MemoryBlock *strBlk = findBlockByAddress(strAddr, strOff);
        if (!strBlk || !strBlk->initialized ||
            strOff + strLen > strBlk->data.size())
          break;
        std::string typeName(
            reinterpret_cast<const char *>(strBlk->data.data() + strOff),
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

    // Intercept create_component_by_name — since factory.register was
    // fast-pathed (skipping MLIR-side data population), the MLIR-side
    // create_component_by_name won't find registered types. This
    // intercept looks up the wrapper from the C++ map and calls
    // create_component via the wrapper's vtable slot 1.
    // Signature: (this, requested_type_name: struct<(ptr,i64)>,
    //             parent_inst_path: struct<(ptr,i64)>,
    //             name: struct<(ptr,i64)>, parent: ptr) -> ptr
    if ((calleeName ==
             "uvm_pkg::uvm_default_factory::create_component_by_name" ||
         calleeName == "uvm_pkg::uvm_factory::create_component_by_name") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 5) {
      // Extract the requested type name string (arg1).
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      std::string requestedName;
      bool nameExtracted = false;
      if (!nameVal.isX() && nameVal.getWidth() >= 128) {
        APInt nameAPInt = nameVal.getAPInt();
        uint64_t strAddr = nameAPInt.extractBits(64, 0).getZExtValue();
        int64_t strLen =
            static_cast<int64_t>(nameAPInt.extractBits(64, 64).getZExtValue());
        if (strLen > 0 && strLen <= 1024 && strAddr != 0) {
          nameExtracted =
              tryReadStringKey(procId, strAddr, strLen, requestedName);
        }
      }
      if (nameExtracted && !requestedName.empty()) {
        auto it = nativeFactoryTypeNames.find(requestedName);
        if (it != nativeFactoryTypeNames.end()) {
          uint64_t wrapperAddr = it->second;
          // Read wrapper's vtable pointer (at offset 4 after i32).
          uint64_t off = 0;
          MemoryBlock *blk = findBlockByAddress(wrapperAddr + 4, off);
          if (blk && blk->initialized && off + 8 <= blk->data.size()) {
            uint64_t vtableAddr = 0;
            for (unsigned i = 0; i < 8; ++i)
              vtableAddr |=
                  static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
            // Read vtable slot 1 = create_component.
            uint64_t off2 = 0;
            MemoryBlock *vtBlk =
                findBlockByAddress(vtableAddr + 1 * 8, off2);
            if (vtBlk && vtBlk->initialized &&
                off2 + 8 <= vtBlk->data.size()) {
              uint64_t funcAddr = 0;
              for (unsigned i = 0; i < 8; ++i)
                funcAddr |= static_cast<uint64_t>(vtBlk->data[off2 + i])
                            << (i * 8);
              auto funcIt = addressToFunction.find(funcAddr);
              if (funcIt != addressToFunction.end()) {
                auto funcOp = rootModule.lookupSymbol<mlir::func::FuncOp>(
                    funcIt->second);
                if (funcOp) {
                  // create_component(wrapper, name, parent) -> ptr
                  InterpretedValue wrapperVal(wrapperAddr, 64);
                  InterpretedValue nameArg =
                      getValue(procId, callIndirectOp.getArgOperands()[3]);
                  InterpretedValue parentArg =
                      getValue(procId, callIndirectOp.getArgOperands()[4]);
                  SmallVector<InterpretedValue, 1> results;
                  if (succeeded(interpretFuncBody(
                          procId, funcOp,
                          {wrapperVal, nameArg, parentArg}, results)) &&
                      !results.empty()) {
                    setValue(procId, callIndirectOp.getResults()[0],
                             results[0]);
                    return success();
                  }
                }
              }
            }
          }
        }
      }
      // Fall through to MLIR interpretation if fast-path fails.
    }

    // Intercept find_wrapper_by_name — uses nativeFactoryTypeNames to look
    // up a type name registered by the fast-path factory.register above.
    // This is needed for +UVM_TESTNAME to find the test class wrapper.
    if ((calleeName == "uvm_pkg::uvm_default_factory::find_wrapper_by_name" ||
         calleeName == "uvm_pkg::uvm_factory::find_wrapper_by_name") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 2) {
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      uint64_t wrapperResult = 0;
      if (!nameVal.isX() && nameVal.getWidth() >= 128 &&
          !nativeFactoryTypeNames.empty()) {
        APInt nameAPInt = nameVal.getAPInt();
        uint64_t strAddr = nameAPInt.extractBits(64, 0).getZExtValue();
        int64_t strLen = static_cast<int64_t>(
            nameAPInt.extractBits(64, 64).getZExtValue());
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
    if ((calleeName == "uvm_pkg::uvm_default_factory::is_type_name_registered" ||
         calleeName == "uvm_pkg::uvm_factory::is_type_name_registered") &&
        callIndirectOp.getNumResults() >= 1 &&
        callIndirectOp.getArgOperands().size() >= 2) {
      InterpretedValue nameVal =
          getValue(procId, callIndirectOp.getArgOperands()[1]);
      bool found = false;
      if (!nameVal.isX() && nameVal.getWidth() >= 128 &&
          !nativeFactoryTypeNames.empty()) {
        APInt nameAPInt = nameVal.getAPInt();
        uint64_t strAddr = nameAPInt.extractBits(64, 0).getZExtValue();
        int64_t strLen = static_cast<int64_t>(
            nameAPInt.extractBits(64, 64).getZExtValue());
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
    if ((calleeName == "uvm_pkg::uvm_default_factory::is_type_registered" ||
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

    // Intercept m_set_full_name to avoid O(N^2) recursive children traversal.
    // During UVM build_phase, each component's constructor calls
    // m_set_full_name on itself. The function computes
    //   self.full_name = parent.get_full_name() + "." + self.get_name()
    // then recursively calls m_set_full_name on every child. The recursive
    // update is redundant because each child calls m_set_full_name on itself
    // during its own construction. We compute the name natively and skip the
    // children loop, turning O(N^2) into O(N).
    if (calleeName.contains("m_set_full_name")) {
      if (!callIndirectOp.getArgOperands().empty()) {
        InterpretedValue selfVal =
            getValue(procId, callIndirectOp.getArgOperands()[0]);
        if (!selfVal.isX() && selfVal.getUInt64() >= 0x1000) {
          uint64_t selfAddr = selfVal.getUInt64();

          // Byte offsets in the uvm_component struct, computed from the
          // interpreter's packed layout (getLLVMTypeSize sums getTypeWidth
          // bits per field, then rounds to bytes).
          //   field[0] uvm_report_object: 41 bytes
          //     field[0][0] uvm_object: 32 bytes
          //       field[0][0][0] uvm_void(i32,ptr): 12 bytes
          //       field[0][0][1] struct<(ptr,i64)> m_inst_name: 16 bytes
          //       field[0][0][2] i32: 4 bytes
          //     field[0][1] ptr: 8 bytes
          //     field[0][2] i1: 1 byte (ceil to 1 byte)
          //   field[1] i1: 1 byte
          //   fields[2-6] ptr x5: 40 bytes
          //   field[7] i1: 1 byte
          //   field[8] i32: 4 bytes
          //   field[9] ptr m_parent: offset 87
          //   field[10] ptr m_children: offset 95
          //   fields[11-13] ptr x3: 24 bytes
          //   field[14] struct<(ptr,i64)> full_name: offset 127
          constexpr uint64_t kParentOff = 87;
          constexpr uint64_t kInstNameOff = 12;
          constexpr uint64_t kFullNameOff = 127;

          // Read 8 little-endian bytes from a memory address.
          auto readU64 = [&](uint64_t addr) -> uint64_t {
            uint64_t off = 0;
            MemoryBlock *blk = findBlockByAddress(addr, off);
            if (!blk)
              blk = findMemoryBlockByAddress(addr, procId, &off);
            if (!blk || !blk->initialized || off + 8 > blk->data.size())
              return 0;
            uint64_t val = 0;
            for (unsigned i = 0; i < 8; ++i)
              val |= static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
            return val;
          };

          // Read a string from a struct<(ptr, i64)> stored at addr.
          auto readStr = [&](uint64_t addr) -> std::string {
            uint64_t strPtr = readU64(addr);
            int64_t strLen = static_cast<int64_t>(readU64(addr + 8));
            if (strPtr == 0 || strLen <= 0)
              return "";
            auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtr));
            if (dynIt != dynamicStrings.end() && dynIt->second.first &&
                dynIt->second.second > 0)
              return std::string(
                  dynIt->second.first,
                  std::min(static_cast<size_t>(strLen),
                           static_cast<size_t>(dynIt->second.second)));
            uint64_t off = 0;
            MemoryBlock *gBlock = findBlockByAddress(strPtr, off);
            if (gBlock && gBlock->initialized) {
              size_t avail =
                  std::min(static_cast<size_t>(strLen),
                           gBlock->data.size() - static_cast<size_t>(off));
              if (avail > 0)
                return std::string(
                    reinterpret_cast<const char *>(gBlock->data.data() + off),
                    avail);
            }
            return "";
          };

          // Write a struct<(ptr, i64)> string to addr.
          auto writeStr = [&](uint64_t addr, uint64_t strPtr, int64_t strLen) {
            uint64_t off = 0;
            MemoryBlock *blk = findBlockByAddress(addr, off);
            if (!blk)
              blk = findMemoryBlockByAddress(addr, procId, &off);
            if (!blk || off + 16 > blk->data.size())
              return;
            for (unsigned i = 0; i < 8; ++i) {
              blk->data[off + i] =
                  static_cast<uint8_t>((strPtr >> (i * 8)) & 0xFF);
              blk->data[off + 8 + i] = static_cast<uint8_t>(
                  (static_cast<uint64_t>(strLen) >> (i * 8)) & 0xFF);
            }
            blk->initialized = true;
          };

          uint64_t parentAddr = readU64(selfAddr + kParentOff);
          std::string instName = readStr(selfAddr + kInstNameOff);

          std::string fullName;
          if (parentAddr == 0) {
            fullName = instName;
          } else {
            // Check if parent IS-A uvm_root (class_id target = 93).
            uint64_t cidOff = 0;
            MemoryBlock *pBlk = findBlockByAddress(parentAddr, cidOff);
            if (!pBlk)
              pBlk = findMemoryBlockByAddress(parentAddr, procId, &cidOff);
            bool isRoot = false;
            if (pBlk && pBlk->initialized &&
                cidOff + 4 <= pBlk->data.size()) {
              int32_t cid = 0;
              for (unsigned i = 0; i < 4; ++i)
                cid |= static_cast<int32_t>(pBlk->data[cidOff + i])
                       << (i * 8);
              isRoot = checkRTTICast(cid, 93);
            }
            if (isRoot) {
              fullName = instName;
            } else {
              std::string parentFull = readStr(parentAddr + kFullNameOff);
              fullName = parentFull + "." + instName;
            }
          }

          // Persist the string and register in dynamicStrings.
          interpreterStrings.push_back(std::move(fullName));
          const std::string &stored = interpreterStrings.back();
          int64_t pv = reinterpret_cast<int64_t>(stored.data());
          int64_t lv = static_cast<int64_t>(stored.size());
          dynamicStrings[pv] = {stored.data(), lv};

          writeStr(selfAddr + kFullNameOff, static_cast<uint64_t>(pv), lv);

          LLVM_DEBUG(llvm::dbgs()
                     << "  call_indirect: m_set_full_name intercepted -> \""
                     << stored << "\"\n");
          return success();
        }
      }
    }

    // Intercept get_full_name on uvm_component (and subclasses).
    // Returns the stored full_name field at offset 127 directly,
    // avoiding vtable dispatch, string comparison, and alloca ops.
    if (calleeName.contains("get_full_name") &&
        calleeName.contains("uvm_component") &&
        callIndirectOp.getNumResults() >= 1 &&
        !callIndirectOp.getArgOperands().empty()) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (!selfVal.isX() && selfVal.getUInt64() >= 0x1000) {
        uint64_t selfAddr = selfVal.getUInt64();
        constexpr uint64_t kFullNameOff2 = 127;
        auto readU64L = [&](uint64_t addr) -> uint64_t {
          uint64_t off = 0;
          MemoryBlock *blk = findBlockByAddress(addr, off);
          if (!blk)
            blk = findMemoryBlockByAddress(addr, procId, &off);
          if (!blk || !blk->initialized || off + 8 > blk->data.size())
            return 0;
          uint64_t val = 0;
          for (unsigned i = 0; i < 8; ++i)
            val |= static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
          return val;
        };
        uint64_t strPtr = readU64L(selfAddr + kFullNameOff2);
        uint64_t strLen = readU64L(selfAddr + kFullNameOff2 + 8);
        if (strPtr != 0 && strLen > 0) {
          uint64_t words[2] = {strPtr, strLen};
          llvm::APInt resultVal(128, llvm::ArrayRef<uint64_t>(words, 2));
          setValue(procId, callIndirectOp.getResult(0),
                   InterpretedValue(resultVal));
          return success();
        }
      }
    }

    // Intercept get_name on uvm_object - returns m_inst_name at offset 12.
    if (calleeName.contains("get_name") &&
        calleeName.contains("uvm_object") &&
        !calleeName.contains("get_name_constraint") &&
        !calleeName.contains("get_name_enabled") &&
        callIndirectOp.getNumResults() >= 1 &&
        !callIndirectOp.getArgOperands().empty()) {
      InterpretedValue selfVal =
          getValue(procId, callIndirectOp.getArgOperands()[0]);
      if (!selfVal.isX() && selfVal.getUInt64() >= 0x1000) {
        uint64_t selfAddr = selfVal.getUInt64();
        constexpr uint64_t kInstNameOff2 = 12;
        auto readU64L = [&](uint64_t addr) -> uint64_t {
          uint64_t off = 0;
          MemoryBlock *blk = findBlockByAddress(addr, off);
          if (!blk)
            blk = findMemoryBlockByAddress(addr, procId, &off);
          if (!blk || !blk->initialized || off + 8 > blk->data.size())
            return 0;
          uint64_t val = 0;
          for (unsigned i = 0; i < 8; ++i)
            val |= static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
          return val;
        };
        uint64_t strPtr = readU64L(selfAddr + kInstNameOff2);
        uint64_t strLen = readU64L(selfAddr + kInstNameOff2 + 8);
        uint64_t words[2] = {strPtr, strLen};
        llvm::APInt resultVal(128, llvm::ArrayRef<uint64_t>(words, 2));
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(resultVal));
        return success();
      }
    }

    // Intercept uvm_get_report_object - trivially returns self.
    if (calleeName.contains("uvm_get_report_object") &&
        callIndirectOp.getNumResults() >= 1 &&
        !callIndirectOp.getArgOperands().empty()) {
      setValue(procId, callIndirectOp.getResult(0),
               getValue(procId, callIndirectOp.getArgOperands()[0]));
      return success();
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
        if (blk && blk->initialized && off + 8 <= blk->data.size()) {
          uint64_t parentAddr = 0;
          for (unsigned i = 0; i < 8; ++i)
            parentAddr |= static_cast<uint64_t>(blk->data[off + i]) << (i * 8);
          setValue(procId, callIndirectOp.getResult(0),
                   InterpretedValue(parentAddr, 64));
          return success();
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
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(calleeName);
    if (!funcOp) {
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
      if (refBlock && offset + 8 <= refBlock->data.size()) {
        for (unsigned i = 0; i < 8; ++i)
          refBlock->data[offset + i] =
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
        if (!stateIt->second.waiting)
          continue;
        stateIt->second.waiting = false;
        scheduler.scheduleProcess(waiterProc, SchedulingRegion::Active);
      }
    };

    // Native queue fast path for phase hopper calls dispatched via vtable.
    if (calleeName.ends_with("uvm_phase_hopper::try_put") && args.size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
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
        raisePhaseObjection(handle, 1);
      }

      unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
      setValue(procId, callIndirectOp.getResult(0),
               InterpretedValue(llvm::APInt(width, 1)));
      return success();
    }

    if (calleeName.ends_with("uvm_phase_hopper::try_get") && args.size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
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
          auto handleIt = phaseObjectionHandles.find(hopperAddr);
          if (handleIt != phaseObjectionHandles.end())
            dropPhaseObjection(handleIt->second, 1);
        }
        unsigned width = getTypeWidth(callIndirectOp.getResult(0).getType());
        setValue(procId, callIndirectOp.getResult(0),
                 InterpretedValue(llvm::APInt(width, hasPhase ? 1 : 0)));
        return success();
      }
    }

    if (calleeName.ends_with("uvm_phase_hopper::try_peek") && args.size() >= 2 &&
        callIndirectOp.getNumResults() >= 1) {
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

    if (calleeName.ends_with("uvm_phase_hopper::peek") && args.size() >= 2) {
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

    if (calleeName.ends_with("uvm_phase_hopper::get") && args.size() >= 2) {
      uint64_t hopperAddr = args[0].isX() ? 0 : args[0].getUInt64();
      auto it = phaseHopperQueue.find(hopperAddr);
      if (it != phaseHopperQueue.end() && !it->second.empty()) {
        uint64_t phaseAddr = it->second.front();
        if (writePointerToOutRef(callIndirectOp.getArgOperands()[1], phaseAddr)) {
          it->second.pop_front();
          auto handleIt = phaseObjectionHandles.find(hopperAddr);
          if (handleIt != phaseObjectionHandles.end())
            dropPhaseObjection(handleIt->second, 1);
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
    if (calleeName.contains("uvm_port_base") &&
        calleeName.ends_with("::size") &&
        callIndirectOp.getNumResults() >= 1) {
      uint64_t selfAddr =
          (!args.empty() && !args[0].isX()) ? args[0].getUInt64() : 0;
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
          unsigned valueBytes = (valueBits + 7) / 8;
          std::vector<uint8_t> valueData(valueBytes, 0);
          if (!valueArg.isX()) {
            llvm::APInt valBits = valueArg.getAPInt();
            for (unsigned i = 0; i < valueBytes; ++i)
              valueData[i] = static_cast<uint8_t>(
                  valBits.extractBits(8, i * 8).getZExtValue());
          }
          configDbEntries[key] = std::move(valueData);
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

    // Intercept config_db implementation via call_indirect (vtable dispatch).
    // config_db#(T) is parametric, so each specialization has its own vtable
    // and calls go through call_indirect, not func.call. Without this
    // interceptor, the stub functions (which just return false/void) execute
    // instead of our key-value store, causing BFM handle lookup failures.
    // Note: class name is "config_db_default_implementation_t", so we match
    // on "config_db" + "implementation" separately (not as one substring).
    if (calleeName.contains("config_db") && calleeName.contains("implementation") &&
        (calleeName.contains("::set") || calleeName.contains("::get"))) {

      auto readStr = [&](unsigned argIdx) -> std::string {
        if (argIdx >= args.size())
          return "";
        return readMooreStringStruct(procId, args[argIdx]);
      };

      if (calleeName.contains("::set") && !calleeName.contains("set_default") &&
          !calleeName.contains("set_override") &&
          !calleeName.contains("set_anonymous")) {
        // The implementation's set signature is:
        //   (self, computed_inst, sv_inst_name, sv_field_name, value, ...)
        // We want key = sv_inst_name + "." + sv_field_name
        if (args.size() >= 5) {
          // The implementation's set signature:
          //   (self, computed_inst, sv_inst_name, sv_field_name, value, ...)
          // Use sv_inst_name + "." + sv_field_name as key.
          std::string str1 = readStr(1);
          std::string str2 = readStr(2);
          std::string str3 = readStr(3);

          std::string instName = str2;
          std::string fieldName = str3;
          // Fallback: if str3 is empty, try old layout (str1=inst, str2=field)
          if (fieldName.empty()) {
            instName = str1;
            fieldName = str2;
          }
          std::string key = instName + "." + fieldName;
          if (traceConfigDbEnabled) {
            llvm::errs() << "[CFG-CI-SET] callee=" << calleeName
                         << " key=\"" << key << "\" s1=\"" << str1
                         << "\" s2=\"" << str2 << "\" s3=\"" << str3
                         << "\" entries_before=" << configDbEntries.size()
                         << "\n";
          }

          InterpretedValue &valueArg = args[4];
          unsigned valueBits = valueArg.getWidth();
          unsigned valueBytes = (valueBits + 7) / 8;
          std::vector<uint8_t> valueData(valueBytes, 0);
          if (!valueArg.isX()) {
            llvm::APInt valBits = valueArg.getAPInt();
            for (unsigned i = 0; i < valueBytes; ++i)
              valueData[i] = static_cast<uint8_t>(
                  valBits.extractBits(8, i * 8).getZExtValue());
          }
          configDbEntries[key] = std::move(valueData);
          if (traceConfigDbEnabled) {
            llvm::errs() << "[CFG-CI-SET] stored key=\"" << key
                         << "\" entries_after=" << configDbEntries.size()
                         << "\n";
          }
        }
        return success();
      }

      if (calleeName.contains("::get") && !calleeName.contains("get_default")) {
        // get(self, context_ptr, sv_inst_name, sv_field_name, value_ref) -> i1
        if (args.size() >= 5 && callIndirectOp.getNumResults() >= 1) {
          // The implementation's get signature:
          //   (self, context_ptr, sv_inst_name, sv_field_name, value_ref)
          // Use sv_inst_name + "." + sv_field_name as key.
          std::string str1 = readStr(1);
          std::string str2 = readStr(2);
          std::string str3 = readStr(3);

          std::string instName = str2;
          std::string fieldName = str3;
          // Fallback: if str3 empty, try old layout
          if (fieldName.empty()) {
            instName = str1;
            fieldName = str2;
          }
          std::string key = instName + "." + fieldName;
          if (traceConfigDbEnabled) {
            llvm::errs() << "[CFG-CI-GET] callee=" << calleeName
                         << " key=\"" << key << "\" s1=\"" << str1
                         << "\" s2=\"" << str2 << "\" s3=\"" << str3
                         << "\" entries=" << configDbEntries.size() << "\n";
          }

          auto it = configDbEntries.find(key);
          if (it == configDbEntries.end()) {
            // Wildcard match: look for entries where field name matches
            for (auto &[k, v] : configDbEntries) {
              size_t dotPos = k.rfind('.');
              if (dotPos != std::string::npos &&
                  k.substr(dotPos + 1) == fieldName) {
                it = configDbEntries.find(k);
                break;
              }
            }
          }
          // Fuzzy match: if fieldName ends with "_x" (unresolved index),
          // try matching entries where the base name matches with any
          // numeric suffix (e.g., "bfm_x" matches "bfm_0").
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
                  LLVM_DEBUG(llvm::dbgs()
                             << "  config_db fuzzy match: '" << fieldName
                             << "' -> '" << storedField << "'\n");
                  break;
                }
              }
            }
          }

          if (it != configDbEntries.end()) {
            if (traceConfigDbEnabled) {
              llvm::errs() << "[CFG-CI-GET] hit key=\"" << it->first
                           << "\" bytes=" << it->second.size() << "\n";
            }
            Value outputRef = callIndirectOp.getArgOperands()[4];
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
              // Also write directly to memory (use procId-aware lookup
              // to find alloca-backed refs)
              InterpretedValue refAddr = getValue(procId, outputRef);
              if (!refAddr.isX()) {
                uint64_t addr = refAddr.getUInt64();
                uint64_t off3 = 0;
                MemoryBlock *blk =
                    findMemoryBlockByAddress(addr, procId, &off3);
                if (blk) {
                  writeConfigDbBytesToMemoryBlock(
                      blk, off3, valueData, innerBytes,
                      /*zeroFillMissing=*/true);
                } else {
                  // Fallback: write to native memory (heap-allocated blocks)
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
              if (!args[4].isX()) {
                uint64_t outputAddr = args[4].getUInt64();
                uint64_t outOff = 0;
                MemoryBlock *outBlock =
                    findMemoryBlockByAddress(outputAddr, procId, &outOff);
                if (outBlock) {
                  writeConfigDbBytesToMemoryBlock(
                      outBlock, outOff, valueData,
                      static_cast<unsigned>(valueData.size()),
                      /*zeroFillMissing=*/false);
                } else {
                  // Fallback: write to native memory
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
            LLVM_DEBUG(llvm::dbgs()
                       << "  config_db::get(\"" << key << "\") -> found ("
                       << valueData.size() << " bytes) via call_indirect\n");
            return success();
          }

          setValue(procId, callIndirectOp.getResult(0),
                  InterpretedValue(llvm::APInt(1, 0)));
          if (traceConfigDbEnabled)
            llvm::errs() << "[CFG-CI-GET] miss key=\"" << key << "\"\n";
          LLVM_DEBUG(llvm::dbgs()
                     << "  config_db::get(\"" << key
                     << "\") -> not found via call_indirect\n");
          return success();
        }
      }
    }

    // Intercept UVM port connect() via call_indirect.
    // Stores port→provider connections natively and returns immediately,
    // bypassing the UVM "Late Connection" phase check that incorrectly
    // rejects connections when the phase hopper's state tracking marks
    // end_of_elaboration as DONE before connect_phase callbacks finish.
    // The native connection map is used by get_next_item, item_done,
    // analysis_port::write, and other TLM operations.
    auto isNativeConnectCallee = [&](llvm::StringRef name) {
      if (!name.contains("::connect"))
        return false;
      return name.contains("uvm_port_base") ||
             name.contains("uvm_analysis_port") ||
             name.contains("uvm_analysis_export") ||
             name.contains("uvm_analysis_imp");
    };
    if (isNativeConnectCallee(calleeName) &&
        !calleeName.contains("connect_phase") && args.size() >= 2) {
      uint64_t selfAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t providerAddr = args[1].isX() ? 0 : args[1].getUInt64();
      if (selfAddr != 0 && providerAddr != 0) {
        auto &conns = analysisPortConnections[selfAddr];
        if (std::find(conns.begin(), conns.end(), providerAddr) == conns.end()) {
          conns.push_back(providerAddr);
          invalidateUvmSequencerQueueCache(selfAddr);
        }
      }
      if (traceAnalysisEnabled)
        llvm::errs() << "[ANALYSIS-CONNECT] " << calleeName
                     << " self=0x" << llvm::format_hex(selfAddr, 0)
                     << " provider=0x" << llvm::format_hex(providerAddr, 0)
                     << "\n";
      // Return immediately — don't fall through to UVM code which would
      // issue "Late Connection" warning and reject the connection.
      return success();
    }

    // Intercept resolve_bindings on UVM ports — since we handle connections
    // natively (bypassing UVM's m_provided_by/m_provided_to), the
    // resolve_bindings check would fail with "connection count of 0 does not
    // meet required minimum". We skip it entirely; our native connection map
    // in analysisPortConnections handles all TLM routing.
    if (calleeName.contains("uvm_port_base") &&
        calleeName.contains("::resolve_bindings")) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  call_indirect: resolve_bindings intercepted (no-op)\n");
      return success();
    }

    // Intercept analysis_port::write to broadcast to connected ports.
    // When the native port_base connect() is rejected due to "Late Connection",
    // the UVM write() loop finds 0 subscribers. We use our native connection
    // map to resolve the correct imp write function via vtable dispatch.
    // Supports multi-hop chains: port → port/export → imp.
    if (calleeName.contains("analysis_port") && calleeName.contains("::write") &&
        !calleeName.contains("write_m_") && args.size() >= 2) {
      uint64_t portAddr = args[0].isX() ? 0 : args[0].getUInt64();
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
      auto seedIt = analysisPortConnections.find(portAddr);
      if (seedIt != analysisPortConnections.end()) {
        for (uint64_t a : seedIt->second)
          worklist.push_back(a);
      }
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
          // Resolve the imp's write function via vtable dispatch.
          // Object layout: [i32 class_id][ptr vtable_ptr][...fields...]
          // Read vtable pointer at byte offset 4 from the imp object.
          uint64_t vtableOff = 0;
          MemoryBlock *impBlock = findBlockByAddress(impAddr, vtableOff);
          if (!impBlock || vtableOff + 4 + 8 > impBlock->size) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  call_indirect: imp object at 0x"
                       << llvm::format_hex(impAddr, 16)
                       << " not found in memory\n");
            continue;
          }
          uint64_t vtableAddr = 0;
          for (unsigned i = 0; i < 8; ++i)
            vtableAddr |= static_cast<uint64_t>(
                              impBlock->data[vtableOff + 4 + i])
                          << (i * 8);
          auto globalIt = addressToGlobal.find(vtableAddr);
          if (globalIt == addressToGlobal.end()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  call_indirect: vtable at 0x"
                       << llvm::format_hex(vtableAddr, 16)
                       << " not found in addressToGlobal\n");
            continue;
          }
          // Read the write function pointer from vtable slot 11.
          auto vtableBlockIt = globalMemoryBlocks.find(globalIt->second);
          if (vtableBlockIt == globalMemoryBlocks.end())
            continue;
          auto &vtableBlock = vtableBlockIt->second;
          unsigned writeSlot = 11;
          unsigned slotOffset = writeSlot * 8;
          if (slotOffset + 8 > vtableBlock.size)
            continue;
          uint64_t writeFuncAddr = 0;
          for (unsigned i = 0; i < 8; ++i)
            writeFuncAddr |=
                static_cast<uint64_t>(vtableBlock.data[slotOffset + i])
                << (i * 8);
          auto funcIt2 = addressToFunction.find(writeFuncAddr);
          if (funcIt2 == addressToFunction.end()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  call_indirect: write func at vtable slot 11 (addr 0x"
                       << llvm::format_hex(writeFuncAddr, 16)
                       << ") not found\n");
            continue;
          }
          auto impWriteFunc = moduleOp.lookupSymbol<func::FuncOp>(funcIt2->second);
          if (!impWriteFunc) {
            if (traceAnalysisEnabled)
              llvm::errs() << "[ANALYSIS-WRITE] function '"
                           << funcIt2->second << "' not found in module\n";
            continue;
          }
          if (traceAnalysisEnabled)
            llvm::errs() << "[ANALYSIS-WRITE] dispatching to "
                         << funcIt2->second << "\n";
          SmallVector<InterpretedValue, 2> impArgs;
          impArgs.push_back(InterpretedValue(llvm::APInt(64, impAddr)));
          impArgs.push_back(args[1]); // transaction object
          SmallVector<InterpretedValue, 2> impResults;
          auto &cState = processStates[procId];
          ++cState.callDepth;
          (void)interpretFuncBody(procId, impWriteFunc, impArgs, impResults,
                                 callIndirectOp);
          --cState.callDepth;
        }
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

    // start_item: Record item→sequencer mapping and return immediately
    // (grants arbitration instantly). Args: (self, item, priority, sequencer).
    if (calleeName.contains("::start_item") && args.size() >= 4) {
      uint64_t seqAddr = args[0].isX() ? 0 : args[0].getUInt64();
      uint64_t itemAddr = args[1].isX() ? 0 : args[1].getUInt64();
      uint64_t sqrAddr = args[3].isX() ? 0 : args[3].getUInt64();
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
            sqrAddr = getSeqResults[0].getUInt64();
            sqrArg = InterpretedValue(llvm::APInt(64, sqrAddr));
          }
        }
      }
      if (itemAddr != 0 && sqrAddr != 0) {
        uint64_t canonicalSqrAddr =
            canonicalizeUvmObjectAddress(procId, sqrAddr);
        if (canonicalSqrAddr == 0)
          canonicalSqrAddr = sqrAddr;
        recordUvmSequencerItemOwner(itemAddr, canonicalSqrAddr);
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: start_item intercepted: item 0x"
                   << llvm::format_hex(itemAddr, 16) << " → sequencer 0x"
                   << llvm::format_hex(canonicalSqrAddr, 16)
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
      uint64_t itemAddr = args[1].isX() ? 0 : args[1].getUInt64();
      if (itemAddr != 0) {
        // Check if item_done was already received (re-poll after wake)
        if (itemDoneReceived.count(itemAddr)) {
          (void)takeUvmSequencerItemOwner(itemAddr);
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
          uint64_t queueAddr = canonicalizeUvmObjectAddress(procId, sqrAddr);
          if (queueAddr == 0)
            queueAddr = sqrAddr;
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
      // Fallback: when the port-to-sequencer resolution fails (no UVM
      // connection chain), scan all FIFOs for any available item. This
      // handles simple test cases with a single sequencer-driver pair.
      if (!found && seqrQueueAddr == 0) {
        for (auto &[qAddr, fifo] : sequencerItemFifo) {
          if (!fifo.empty()) {
            seqrQueueAddr = qAddr;
            itemAddr = fifo.front();
            fifo.pop_front();
            found = true;
            fromFallbackSearch = true;
            break;
          }
        }
      }
      if (found && itemAddr != 0) {
        // Track the dequeued item by both pull-port and resolved queue alias.
        recordUvmDequeuedItem(portAddr, seqrQueueAddr, itemAddr);
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
              offset + 8 <= refBlock->data.size()) {
            for (unsigned i = 0; i < 8; ++i)
              refBlock->data[offset + i] =
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
              offset + 8 <= refBlock->data.size()) {
            for (unsigned i = 0; i < 8; ++i)
              refBlock->data[offset + i] = 0;
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
      // If no item available, suspend the process and schedule a poll
      // to retry on the next delta. Return success() so the halt check
      // in interpretFuncBody saves call stack frames. Store the call op
      // in sequencerGetRetryCallOp so that on resume, the innermost
      // frame's resumeOp is overridden to re-execute this call_indirect
      // (instead of skipping past it).
      LLVM_DEBUG(llvm::dbgs() << "  seq_item_pull_port::get: FIFO empty, "
                              << "suspending for delta poll\n");
      auto &pState = processStates[procId];
      pState.waiting = true;
      pState.sequencerGetRetryCallOp = callIndirectOp.getOperation();
      SimTime currentTime = scheduler.getCurrentTime();
      constexpr uint32_t kDeltaPollSafetyMargin = 32;
      constexpr uint32_t kMaxDeltaPollBudgetCap = 256;
      constexpr int64_t kFallbackPollDelayFs = 10000000; // 10 ps
      size_t configuredMaxDeltas = scheduler.getMaxDeltaCycles();
      uint32_t deltaPollBudget = 0;
      if (configuredMaxDeltas > kDeltaPollSafetyMargin) {
        deltaPollBudget = static_cast<uint32_t>(std::min<size_t>(
            configuredMaxDeltas - kDeltaPollSafetyMargin,
            kMaxDeltaPollBudgetCap));
      }
      SimTime targetTime;
      if (currentTime.deltaStep < deltaPollBudget)
        targetTime = currentTime.nextDelta();
      else
        targetTime = currentTime.advanceTime(kFallbackPollDelayFs);
      scheduler.getEventScheduler().schedule(
          targetTime, SchedulingRegion::Active,
          Event([this, procId]() {
            auto &st = processStates[procId];
            st.waiting = false;
            scheduler.scheduleProcess(procId, SchedulingRegion::Active);
          }));
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
          if (waiterStateIt != processStates.end() &&
              waiterStateIt->second.waiting) {
            waiterStateIt->second.waiting = false;
            scheduler.scheduleProcess(waiterProcId,
                                      SchedulingRegion::Active);
            LLVM_DEBUG(llvm::dbgs()
                       << "  item_done: resuming finish_item waiter proc="
                       << waiterProcId << "\n");
          }
        }
      } else if (traceSeq) {
        llvm::errs() << "[SEQ-CI] item_done miss caller=0x"
                     << llvm::format_hex(doneAddr, 16) << "\n";
      }
      return success();
    }

    // Check call depth to prevent stack overflow from deep recursion (UVM patterns)
    auto &callState = processStates[procId];
    constexpr size_t maxCallDepth = 200;
    if (callState.callDepth >= maxCallDepth) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: max call depth ("
                              << maxCallDepth
                              << ") exceeded, returning zero\n");
      for (Value result : callIndirectOp.getResults()) {
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
      }
      return success();
    }

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
      if (!calleeName.contains("phase_hopper") && !args.empty() && !args[0].isX()) {
        uint64_t phaseAddr = args[0].getUInt64();
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
        if (calleeName.contains("raise_objection")) {
          raisePhaseObjection(handle, count);
        } else {
          dropPhaseObjection(handle, count);
        }
      }
      if (indAddedToVisited) {
        auto &depthRef = processStates[procId].recursionVisited[indFuncKey][indArg0Val];
        if (depthRef > 0)
          --depthRef;
      }
      return success();
    }

    // Call the function with depth tracking
    ++callState.callDepth;
    SmallVector<InterpretedValue, 2> results;
    // Pass the call operation so it can be saved in call stack frames
    LogicalResult funcResult =
        interpretFuncBody(procId, funcOp, args, results, callIndirectOp);
    --callState.callDepth;

    // Decrement depth counter after returning
    if (indAddedToVisited) {
      auto &depthRef = processStates[procId].recursionVisited[indFuncKey][indArg0Val];
      if (depthRef > 0)
        --depthRef;
    }

    if (failed(funcResult)) {
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
        if (indAddedToVisited) {
          auto &depthRef = processStates[procId].recursionVisited[indFuncKey][indArg0Val];
          if (depthRef > 0)
            --depthRef;
        }
        LLVM_DEBUG(llvm::dbgs() << "  call_indirect: '" << calleeName
                                << "' returned failure but process is waiting"
                                << " -- treating as suspension\n");
        return success();
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
      // Decrement depth counter since we're returning early
      if (indAddedToVisited) {
        auto &depthRef =
            processStates[procId].recursionVisited[indFuncKey][indArg0Val];
        if (depthRef > 0)
          --depthRef;
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

    // Set results
    for (auto [result, retVal] : llvm::zip(callIndirectOp.getResults(), results)) {
      setValue(procId, result, retVal);
    }

    return success();
}
