//===- LLHDProcessInterpreterWaitCondition.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains wait(condition) runtime handling extracted from
// LLHDProcessInterpreter.cpp to keep that file manageable.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Runtime/MooreRuntime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

namespace {

// Helper: given a block argument, find the corresponding operand from a
// predecessor's terminator. Returns nullptr on failure.
static Value traceBlockArgThroughPred(Block *pred, Block *block,
                                      unsigned argIdx) {
  auto *terminator = pred->getTerminator();
  if (auto brOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
    if (argIdx < brOp.getDestOperands().size())
      return brOp.getDestOperands()[argIdx];
  } else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (condBrOp.getTrueDest() == block &&
        argIdx < condBrOp.getTrueDestOperands().size())
      return condBrOp.getTrueDestOperands()[argIdx];
    if (condBrOp.getFalseDest() == block &&
        argIdx < condBrOp.getFalseDestOperands().size())
      return condBrOp.getFalseDestOperands()[argIdx];
  } else if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
    if (argIdx < waitOp.getDestOperands().size())
      return waitOp.getDestOperands()[argIdx];
  }
  return nullptr;
}

static bool hasSingleFieldIndex(ArrayRef<int64_t> pos, unsigned idx) {
  return pos.size() == 1 && pos.front() == static_cast<int64_t>(idx);
}

} // namespace

LogicalResult LLHDProcessInterpreter::interpretMooreWaitConditionCall(
    ProcessId procId, LLVM::CallOp callOp) {
  if (callOp.getNumOperands() >= 1) {
    bool traceWaitCondition =
        std::getenv("CIRCT_SIM_TRACE_WAIT_CONDITION") != nullptr;
    std::string waitProcName;
    if (traceWaitCondition) {
      if (auto *proc = scheduler.getProcess(procId))
        waitProcName = proc->getName();
    }
    InterpretedValue condArg = getValue(procId, callOp.getOperand(0));
    bool conditionTrue = !condArg.isX() && condArg.getUInt64() != 0;

    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_wait_condition("
                            << (condArg.isX() ? "X"
                                              : std::to_string(condArg.getUInt64()))
                            << ") -> condition is "
                            << (conditionTrue ? "true" : "false") << "\n");
    if (conditionTrue) {
      // Condition is already true, continue immediately.
      // Clear the restart block and delta poll counter since we're done.
      auto &state = processStates[procId];
      ++state.waitConditionPollToken;
      memoryEventWaiters.erase(procId);
      removeObjectionZeroWaiter(procId);
      if (state.waitConditionQueueAddr != 0) {
        removeQueueNotEmptyWaiter(procId);
        state.waitConditionQueueAddr = 0;
      }
      state.waitConditionRestartBlock = nullptr;
      if (traceWaitCondition) {
        llvm::errs() << "[WAITCOND] proc=" << procId;
        if (!waitProcName.empty())
          llvm::errs() << " name=" << waitProcName;
        llvm::errs() << " condition=true\n";
      }
      return success();
    }

    // Condition is false - suspend the process and set up sensitivity
    // to all probed signals so we wake up when something changes.
    auto &state = processStates[procId];
    memoryEventWaiters.erase(procId);
    removeObjectionZeroWaiter(procId);
    if (state.waitConditionQueueAddr != 0) {
      removeQueueNotEmptyWaiter(procId);
      state.waitConditionQueueAddr = 0;
    }

    state.waiting = true;

    // Find the operations that compute the condition value by walking
    // backwards from the condition argument. We need to invalidate these
    // cached values so they get recomputed when we re-check the condition.
    //
    // CRITICAL: We trace through operations that need re-execution:
    // - llvm.load: reads memory that may have changed
    // - llhd.prb: probes a signal that may have changed
    // - Arithmetic/comparison ops that depend on loads/probes
    //
    // We do NOT trace into:
    // - llvm.getelementptr: pointer arithmetic (doesn't read memory)
    // - Constant operations
    //
    // This avoids re-executing side-effecting operations like sim.fork.
    state.waitConditionValuesToInvalidate.clear();
    // Use the call operation's parent block as the restart block.
    // This is critical when wait_condition is called inside a function
    // (e.g., uvm_phase_hopper::get called via vtable dispatch).
    // state.currentBlock is the PROCESS body's block, but we need
    // the FUNCTION body's block where the condition is computed.
    Block *condBlock = callOp->getBlock();
    state.waitConditionRestartBlock = condBlock;

    // Find the load/probe/call chain that contributes to the condition.
    Value condValue = callOp.getOperand(0);
    llvm::SmallVector<Value, 16> worklist;
    llvm::SmallPtrSet<Value, 32> visited;
    llvm::SmallPtrSet<Value, 32> invalidated;
    worklist.push_back(condValue);

    Operation *restartOp = nullptr;
    uint64_t queueWaitAddr = 0;
    llvm::SmallDenseMap<uint64_t, unsigned, 4> waitConditionLoadAddrs;
    unsigned tracedBlockArgs = 0;
    unsigned tracedBlockArgEdges = 0;

    auto recordMemoryLoadAddr = [&](LLVM::LoadOp loadOp) {
      unsigned loadSizeBytes = 0;
      Type loadType = loadOp.getType();
      if (auto intType = dyn_cast<IntegerType>(loadType)) {
        unsigned width = intType.getWidth();
        if (width > 0)
          loadSizeBytes = (width + 7) / 8;
      } else if (isa<LLVM::LLVMPointerType>(loadType)) {
        loadSizeBytes = 8;
      }
      if (loadSizeBytes == 0 || loadSizeBytes > 8)
        return;

      InterpretedValue loadAddrValue = getValue(procId, loadOp.getAddr());
      if (loadAddrValue.isX())
        return;
      uint64_t loadAddr = loadAddrValue.getUInt64();
      if (loadAddr == 0)
        return;

      auto [it, inserted] =
          waitConditionLoadAddrs.try_emplace(loadAddr, loadSizeBytes);
      if (!inserted && loadSizeBytes > it->second)
        it->second = loadSizeBytes;
    };

    auto markInvalidate = [&](Value invalValue) {
      if (invalidated.insert(invalValue).second)
        state.waitConditionValuesToInvalidate.push_back(invalValue);
    };

    auto maybeChooseRestartOp = [&](Operation *candidate) {
      if (!candidate)
        return;
      if (!restartOp) {
        restartOp = candidate;
        return;
      }
      if (candidate->getBlock() == restartOp->getBlock() &&
          candidate->isBeforeInBlock(restartOp))
        restartOp = candidate;
    };

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!visited.insert(v).second)
        continue;

      if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        ++tracedBlockArgs;
        Block *argBlock = blockArg.getOwner();
        unsigned argIdx = blockArg.getArgNumber();
        for (Block *pred : argBlock->getPredecessors()) {
          Value incoming = traceBlockArgThroughPred(pred, argBlock, argIdx);
          if (!incoming || incoming == v)
            continue;
          ++tracedBlockArgEdges;
          worklist.push_back(incoming);
        }
        continue;
      }

      Operation *defOp = v.getDefiningOp();
      if (!defOp)
        continue;

      // Keep tracing local to the same function body as the wait call.
      if (defOp->getParentOp() != callOp->getParentOp())
        continue;

      // Load/probe/call operations are potential restart points.
      if (isa<LLVM::LoadOp, llhd::ProbeOp, LLVM::CallOp>(defOp)) {
        markInvalidate(v);
        maybeChooseRestartOp(defOp);
      }
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(defOp))
        recordMemoryLoadAddr(loadOp);
      if (queueWaitAddr == 0) {
        if (auto queueSizeCall = dyn_cast<LLVM::CallOp>(defOp)) {
          if (queueSizeCall.getCallee() == "__moore_queue_size" &&
              queueSizeCall.getNumOperands() >= 1) {
            InterpretedValue queueArg =
                getValue(procId, queueSizeCall.getOperand(0));
            if (!queueArg.isX())
              queueWaitAddr = queueArg.getUInt64();
          }
        }
        if (queueWaitAddr == 0) {
          if (auto extractOp = dyn_cast<LLVM::ExtractValueOp>(defOp)) {
            if (hasSingleFieldIndex(extractOp.getPosition(), 1)) {
              if (auto queueLoad =
                      extractOp.getContainer().getDefiningOp<LLVM::LoadOp>()) {
                auto queueStructType =
                    dyn_cast<LLVM::LLVMStructType>(queueLoad.getType());
                if (queueStructType && !queueStructType.isOpaque()) {
                  ArrayRef<Type> body = queueStructType.getBody();
                  if (body.size() == 2 &&
                      isa<LLVM::LLVMPointerType>(body[0]) &&
                      isa<IntegerType>(body[1])) {
                    InterpretedValue queueAddrValue =
                        getValue(procId, queueLoad.getAddr());
                    if (!queueAddrValue.isX())
                      queueWaitAddr = queueAddrValue.getUInt64();
                  }
                }
              }
            }
          }
        }
      }

      // Trace through comparison/arithmetic/extraction/value-casts and
      // through loads/probes/calls that feed those computations.
      bool shouldTrace = isa<comb::ICmpOp, LLVM::ZExtOp, LLVM::SExtOp>(defOp) ||
                         isa<LLVM::ICmpOp, LLVM::TruncOp, LLVM::BitcastOp>(defOp) ||
                         isa<comb::AddOp, comb::SubOp, comb::AndOp>(defOp) ||
                         isa<comb::OrOp, comb::XorOp>(defOp) ||
                         isa<comb::ExtractOp, LLVM::ExtractValueOp>(defOp) ||
                         isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(defOp) ||
                         isa<arith::CmpIOp, arith::AddIOp, arith::SubIOp>(defOp) ||
                         isa<LLVM::LoadOp>(defOp) ||
                         isa<llhd::ProbeOp>(defOp) ||
                         isa<LLVM::CallOp>(defOp);
      if (shouldTrace) {
        markInvalidate(v);
        for (Value operand : defOp->getOperands()) {
          if (operand == v)
            continue;
          worklist.push_back(operand);
        }
      }
    }

    // Save the restart point. If dependency tracing found a concrete
    // load/probe/call producer, restart there so CFG-carried values are
    // recomputed correctly (e.g. block args fed from predecessor blocks).
    if (restartOp) {
      state.waitConditionRestartBlock = restartOp->getBlock();
      state.waitConditionRestartOp = mlir::Block::iterator(restartOp);
    } else {
      state.waitConditionRestartBlock = condBlock;
      state.waitConditionRestartOp = mlir::Block::iterator(&*callOp);
    }

    LLVM_DEBUG(llvm::dbgs() << "    Setting restart point for wait_condition "
                            << "re-evaluation (" << state.waitConditionValuesToInvalidate.size()
                            << " values to invalidate)\n");

    // For wait(condition), use polling to re-evaluate.
    // The condition may depend on class member variables in heap memory
    // that aren't tracked via LLHD signals.
    // Use delta steps first to keep $time == 0 during UVM initialization,
    // but fall back to real time after many delta steps to avoid infinite loops.
    SimTime currentTime = scheduler.getCurrentTime();
    constexpr uint32_t kMaxDeltaPolls = 1000;
    constexpr int64_t kFallbackPollDelayFs = 10000000; // 10 ns
    constexpr int64_t kQueueFallbackPollDelayFs = 100000000; // 100 ns
    // execute_phase wait(condition) loops already register objection-zero
    // waiters. Use a sparse timed poll only as a safety fallback to avoid
    // high-frequency churn while objections remain raised.
    constexpr int64_t kExecutePhaseObjectionFallbackPollDelayFs =
        1000000000; // 1 us
    constexpr uint32_t kMemoryMaxDeltaPolls = 32;
    constexpr int64_t kMemoryFallbackPollDelayFs = 1000000000; // 1 us
    SimTime targetTime;
    uint64_t memoryWaitAddr = 0;
    int64_t objectionWaitHandle = MOORE_OBJECTION_INVALID_HANDLE;
    uint64_t waitConditionPhaseAddr = 0;
    StringRef waitConditionPhaseSource = "none";
    StringRef waitConditionParentFuncName;

    LLVM::LLVMFuncOp parentLLVMFunc =
        callOp->getParentOfType<LLVM::LLVMFuncOp>();
    func::FuncOp parentFuncFunc = callOp->getParentOfType<func::FuncOp>();
    if (parentLLVMFunc)
      waitConditionParentFuncName = parentLLVMFunc.getName();
    else if (parentFuncFunc)
      waitConditionParentFuncName = parentFuncFunc.getName();

    bool isExecutePhaseWait =
        waitConditionParentFuncName == "uvm_pkg::uvm_phase_hopper::execute_phase";
    if (isExecutePhaseWait) {
      auto phaseIt = currentExecutingPhaseAddr.find(procId);
      if (phaseIt != currentExecutingPhaseAddr.end()) {
        waitConditionPhaseAddr = phaseIt->second;
        if (waitConditionPhaseAddr != 0)
          waitConditionPhaseSource = "proc-map";
      }

      // Forked monitor branches may not have their own phase entry yet.
      // Recover from the nearest parent process that carries one.
      if (waitConditionPhaseAddr == 0) {
        ProcessId ancestor = procId;
        for (unsigned hops = 0; hops < 8; ++hops) {
          auto stIt = processStates.find(ancestor);
          if (stIt == processStates.end())
            break;
          ProcessId parentId = stIt->second.parentProcessId;
          if (parentId == InvalidProcessId)
            break;
          auto parentPhaseIt = currentExecutingPhaseAddr.find(parentId);
          if (parentPhaseIt != currentExecutingPhaseAddr.end() &&
              parentPhaseIt->second != 0) {
            waitConditionPhaseAddr = parentPhaseIt->second;
            waitConditionPhaseSource = "ancestor-map";
            break;
          }
          ancestor = parentId;
        }
      }

      // Recover phase argument from the current function frame when map
      // propagation didn't seed it.
      if (waitConditionPhaseAddr == 0) {
        if (parentLLVMFunc && parentLLVMFunc.getNumArguments() > 1) {
          InterpretedValue phaseArg =
              getValue(procId, parentLLVMFunc.getArgument(1));
          if (!phaseArg.isX()) {
            waitConditionPhaseAddr = phaseArg.getUInt64();
            if (waitConditionPhaseAddr != 0)
              waitConditionPhaseSource = "llvm-arg1";
          }
        } else if (parentFuncFunc && parentFuncFunc.getNumArguments() > 1) {
          InterpretedValue phaseArg =
              getValue(procId, parentFuncFunc.getArgument(1));
          if (!phaseArg.isX()) {
            waitConditionPhaseAddr = phaseArg.getUInt64();
            if (waitConditionPhaseAddr != 0)
              waitConditionPhaseSource = "func-arg1";
          }
        }
      }

      if (waitConditionPhaseAddr != 0) {
        currentExecutingPhaseAddr[procId] = waitConditionPhaseAddr;
        auto handleIt = phaseObjectionHandles.find(waitConditionPhaseAddr);
        if (handleIt == phaseObjectionHandles.end()) {
          std::string pn = "phase_" + std::to_string(waitConditionPhaseAddr);
          MooreObjectionHandle handle = __moore_objection_create(
              pn.c_str(), static_cast<int64_t>(pn.size()));
          phaseObjectionHandles[waitConditionPhaseAddr] = handle;
          handleIt = phaseObjectionHandles.find(waitConditionPhaseAddr);
        }
        if (handleIt != phaseObjectionHandles.end())
          objectionWaitHandle = handleIt->second;
      }
    }

    if (objectionWaitHandle != MOORE_OBJECTION_INVALID_HANDLE) {
      // execute_phase wait loops are objection-driven in UVM. Register an
      // objection-zero waiter and use a sparse timed poll as fallback.
      enqueueObjectionZeroWaiter(objectionWaitHandle, procId,
                                 callOp.getOperation());
      targetTime =
          currentTime.advanceTime(kExecutePhaseObjectionFallbackPollDelayFs);
    } else if (queueWaitAddr != 0) {
      // Queue wait conditions (`wait(__moore_queue_size(...) > 0)`) are
      // common in UVM hot loops. Register an event-style wakeup on queue
      // mutation and keep a low-frequency timed poll as a safety net.
      state.waitConditionQueueAddr = queueWaitAddr;
      enqueueQueueNotEmptyWaiter(queueWaitAddr, procId, callOp.getOperation());
      targetTime = currentTime.advanceTime(kQueueFallbackPollDelayFs);
    } else if (waitConditionLoadAddrs.size() == 1) {
      // For memory-backed waits that depend on a single load, register a
      // direct memory waiter so stores wake the process without tight polls.
      auto onlyLoad = *waitConditionLoadAddrs.begin();
      memoryWaitAddr = onlyLoad.getFirst();
      unsigned memoryWaitSize = onlyLoad.getSecond();
      MemoryEventWaiter waiter;
      waiter.address = memoryWaitAddr;
      waiter.lastValue = 0;
      waiter.valueSize = memoryWaitSize;
      waiter.waitForRisingEdge = false;

      uint64_t offset = 0;
      if (MemoryBlock *block =
              findMemoryBlockByAddress(memoryWaitAddr, procId, &offset)) {
        if (block->initialized && offset + memoryWaitSize <= block->size) {
          for (unsigned i = 0; i < memoryWaitSize; ++i)
            waiter.lastValue |=
                static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
        }
      }
      memoryEventWaiters[procId] = waiter;

      if (currentTime.deltaStep < kMemoryMaxDeltaPolls)
        targetTime = currentTime.nextDelta();
      else
        targetTime = currentTime.advanceTime(kMemoryFallbackPollDelayFs);
    } else {
      if (currentTime.deltaStep < kMaxDeltaPolls) {
        targetTime = currentTime.nextDelta();
      } else {
        targetTime = currentTime.advanceTime(kFallbackPollDelayFs);
      }
    }

    if (traceWaitCondition) {
      std::string parentFuncName;
      if (auto llvmFunc = callOp->getParentOfType<LLVM::LLVMFuncOp>())
        parentFuncName = llvmFunc.getName().str();
      else if (auto func = callOp->getParentOfType<func::FuncOp>())
        parentFuncName = func.getName().str();
      llvm::errs() << "[WAITCOND] proc=" << procId;
      if (!waitProcName.empty())
        llvm::errs() << " name=" << waitProcName;
      if (!parentFuncName.empty())
        llvm::errs() << " func=" << parentFuncName;
      if (isExecutePhaseWait)
        llvm::errs() << " phaseAddr="
                     << llvm::format_hex(waitConditionPhaseAddr, 16)
                     << " phaseSource=" << waitConditionPhaseSource;
      llvm::errs() << " condition=false queueWait="
                   << llvm::format_hex(queueWaitAddr, 16)
                   << " memoryLoads=" << waitConditionLoadAddrs.size()
                   << " memoryWait="
                   << llvm::format_hex(memoryWaitAddr, 16)
                   << " objectionWaitHandle="
                   << objectionWaitHandle
                   << " blockArgs=" << tracedBlockArgs
                   << " blockArgEdges=" << tracedBlockArgEdges
                   << " restart="
                   << (restartOp ? restartOp->getName().getStringRef()
                                 : StringRef("llvm.call"))
                   << " targetTimeFs=" << targetTime.realTime
                   << " targetDelta=" << targetTime.deltaStep << "\n";
    }

    LLVM_DEBUG(llvm::dbgs() << "    Scheduling wait_condition poll (time="
                            << targetTime.realTime << " fs, delta="
                            << targetTime.deltaStep
                            << ", queueWait=0x"
                            << llvm::format_hex(queueWaitAddr, 16)
                            << ", memoryWait=0x"
                            << llvm::format_hex(memoryWaitAddr, 16)
                            << ")\n");

    uint64_t pollToken = ++state.waitConditionPollToken;
    uint64_t scheduledQueueWaitAddr = state.waitConditionQueueAddr;
    int64_t scheduledObjectionWaitHandle = objectionWaitHandle;

    // Schedule the process to resume after the delay
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, procId, pollToken, scheduledQueueWaitAddr,
               scheduledObjectionWaitHandle]() {
          auto stIt = processStates.find(procId);
          if (stIt == processStates.end())
            return;
          auto &st = stIt->second;
          if (st.halted || !st.waiting)
            return;
          if (st.waitConditionPollToken != pollToken)
            return;
          if (scheduledQueueWaitAddr != 0 &&
              st.waitConditionQueueAddr != scheduledQueueWaitAddr)
            return;
          if (st.waitConditionQueueAddr != 0)
            removeQueueNotEmptyWaiter(procId);
          if (scheduledObjectionWaitHandle !=
              MOORE_OBJECTION_INVALID_HANDLE)
            removeObjectionZeroWaiter(procId);
          st.waiting = false;
          scheduler.scheduleProcess(procId, SchedulingRegion::Active);
        }));
  }
  return success();
}
