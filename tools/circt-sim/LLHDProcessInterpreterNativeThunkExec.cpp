//===- LLHDProcessInterpreterNativeThunkExec.cpp - Native thunk execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "JITCompileManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

static bool isPureResumableWaitPreludeOp(Operation *op) {
  if (!op || op->getNumResults() == 0)
    return false;
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<llhd::DriveOp, llhd::WaitOp, llhd::HaltOp, mlir::cf::BranchOp,
          mlir::cf::CondBranchOp, mlir::func::CallOp,
          mlir::func::CallIndirectOp, LLVM::CallOp,
          sim::PrintFormattedProcOp>(op))
    return false;
  if (isa<llhd::ProbeOp>(op))
    return true;
  return mlir::isMemoryEffectFree(op);
}

static bool isMooreWaitConditionCall(LLVM::CallOp callOp) {
  auto callee = callOp.getCallee();
  return callee && *callee == "__moore_wait_condition";
}

static bool hasMooreWaitConditionCall(Region &region) {
  bool found = false;
  region.walk([&](LLVM::CallOp callOp) {
    if (isMooreWaitConditionCall(callOp))
      found = true;
  });
  return found;
}

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

static const Region *
resolveNativeThunkProcessRegion(const ProcessExecutionState &state) {
  auto processOp = state.getProcessOp();
  if (!processOp)
    return nullptr;
  const Region *region = &processOp.getBody();
  if (state.parentProcessId != InvalidProcessId && state.currentBlock) {
    const Region *activeRegion = state.currentBlock->getParent();
    if (activeRegion && activeRegion != region)
      region = activeRegion;
  }
  return region;
}

static Region *resolveNativeThunkProcessRegion(ProcessExecutionState &state) {
  return const_cast<Region *>(
      resolveNativeThunkProcessRegion(static_cast<const ProcessExecutionState &>(
          state)));
}

bool LLHDProcessInterpreter::tryExecuteDirectProcessFastPath(
    ProcessId procId, ProcessExecutionState &state) {
  auto toFastPathMask = [](DirectProcessFastPathKind kind) -> uint8_t {
    return static_cast<uint8_t>(kind);
  };

  // Restrict this direct path to top-level llhd.process bodies. Fork children,
  // combinationals, and seq.initials keep existing dispatch.
  if (state.parentProcessId != InvalidProcessId || state.getCombinationalOp() ||
      state.getInitialOp())
    return false;

  uint8_t kindMask = 0;
  auto kindIt = directProcessFastPathKinds.find(procId);
  if (kindIt == directProcessFastPathKinds.end()) {
    PeriodicToggleClockThunkSpec periodicSpec;
    if (tryBuildPeriodicToggleClockThunkSpec(state, periodicSpec)) {
      periodicToggleClockThunkSpecs[procId] = std::move(periodicSpec);
      kindMask |= toFastPathMask(DirectProcessFastPathKind::PeriodicToggleClock);
    } else {
      periodicToggleClockThunkSpecs.erase(procId);
    }

    if (isResumableWaitSelfLoopNativeThunkCandidate(procId, state, nullptr))
      kindMask |=
          toFastPathMask(DirectProcessFastPathKind::ResumableWaitSelfLoop);

    directProcessFastPathKinds[procId] = kindMask;
  } else {
    kindMask = kindIt->second;
  }

  if (kindMask == 0)
    return false;

  auto clearKind = [&](DirectProcessFastPathKind kind) {
    kindMask &= ~toFastPathMask(kind);
    directProcessFastPathKinds[procId] = kindMask;
    if ((kind == DirectProcessFastPathKind::PeriodicToggleClock) &&
        ((kindMask &
          toFastPathMask(DirectProcessFastPathKind::PeriodicToggleClock)) == 0))
      periodicToggleClockThunkSpecs.erase(procId);
  };

  auto tryKind = [&](DirectProcessFastPathKind kind,
                     auto &&executor) -> bool {
    JITDeoptStateSnapshot deoptSnapshot;
    bool hasDeoptSnapshot = snapshotJITDeoptState(procId, deoptSnapshot);

    ProcessThunkExecutionState thunkState;
    thunkState.resumeToken = state.jitThunkResumeToken;

    if (!executor(procId, state, thunkState)) {
      clearKind(kind);
      return false;
    }

    if (!thunkState.deoptRequested) {
      state.jitThunkResumeToken = thunkState.resumeToken;
      return true;
    }

    if (hasDeoptSnapshot)
      (void)restoreJITDeoptState(procId, deoptSnapshot);
    clearKind(kind);
    return false;
  };

  if (kindMask & toFastPathMask(DirectProcessFastPathKind::PeriodicToggleClock))
    if (tryKind(DirectProcessFastPathKind::PeriodicToggleClock,
                [this](ProcessId id, ProcessExecutionState &s,
                       ProcessThunkExecutionState &thunkState) {
                  return executePeriodicToggleClockNativeThunk(id, s,
                                                               thunkState);
                }))
      return true;

  if (kindMask & toFastPathMask(DirectProcessFastPathKind::ResumableWaitSelfLoop))
    if (tryKind(DirectProcessFastPathKind::ResumableWaitSelfLoop,
                [this](ProcessId id, ProcessExecutionState &s,
                       ProcessThunkExecutionState &thunkState) {
                  return executeResumableWaitSelfLoopNativeThunk(id, s,
                                                                 thunkState);
                }))
      return true;

  return false;
}

void LLHDProcessInterpreter::executeTrivialNativeThunk(
    ProcessId procId, ProcessThunkExecutionState &thunkState) {
  if (forceJitThunkDeoptRequests) {
    thunkState.deoptRequested = true;
    return;
  }

  auto it = processStates.find(procId);
  if (it == processStates.end() || it->second.halted)
    return;

  auto guardIt = jitProcessThunkIndirectSiteGuards.find(procId);
  if (guardIt != jitProcessThunkIndirectSiteGuards.end()) {
    static bool traceThunkGuards = []() {
      const char *env = std::getenv("CIRCT_SIM_TRACE_JIT_THUNK_GUARDS");
      return env && env[0] != '\0' && env[0] != '0';
    }();
    auto requestGuardDeopt = [&](std::string detail) {
      if (traceThunkGuards) {
        llvm::errs() << "[JIT-THUNK-GUARD] proc=" << procId;
        if (auto *proc = scheduler.getProcess(procId))
          llvm::errs() << " name=" << proc->getName();
        llvm::errs() << " shape=indirect_target_set_guard reason=" << detail
                     << "\n";
      }
      thunkState.deoptRequested = true;
      thunkState.deoptDetail = std::move(detail);
      if (jitCompileManager)
        jitCompileManager->invalidateProcessThunk(procId);
      jitProcessThunkIndirectSiteGuards.erase(procId);
    };
    for (const auto &guard : guardIt->second) {
      auto callIndirect =
          dyn_cast_or_null<mlir::func::CallIndirectOp>(guard.siteOp);
      if (!callIndirect) {
        requestGuardDeopt("call_indirect_target_guard:invalid_site");
        return;
      }
      auto siteProfile = lookupJitRuntimeIndirectSiteProfile(callIndirect);
      if (!siteProfile) {
        requestGuardDeopt("call_indirect_target_guard:profile_missing");
        return;
      }
      if (siteProfile->unresolvedCalls != guard.expectedUnresolvedCalls) {
        requestGuardDeopt(
            (Twine("call_indirect_target_guard:unresolved_calls_changed:site=") +
             Twine(siteProfile->siteId) + ":expected=" +
             Twine(guard.expectedUnresolvedCalls) + ":actual=" +
             Twine(siteProfile->unresolvedCalls))
                .str());
        return;
      }
      if (siteProfile->targetSetVersion != guard.expectedTargetSetVersion ||
          siteProfile->targetSetHash != guard.expectedTargetSetHash) {
        requestGuardDeopt(
            (Twine("call_indirect_target_guard:target_set_changed:site=") +
             Twine(siteProfile->siteId) + ":expected_version=" +
             Twine(guard.expectedTargetSetVersion) + ":actual_version=" +
             Twine(siteProfile->targetSetVersion) + ":expected_hash=" +
             Twine(guard.expectedTargetSetHash) + ":actual_hash=" +
             Twine(siteProfile->targetSetHash))
                .str());
        return;
      }
    }
  }

  if (executeResumableWaitThenHaltNativeThunk(procId, it->second, thunkState))
    return;

  if (executePeriodicToggleClockNativeThunk(procId, it->second, thunkState))
    return;

  if (executeResumableWaitSelfLoopNativeThunk(procId, it->second, thunkState))
    return;

  if (executeResumableMultiblockWaitNativeThunk(procId, it->second, thunkState))
    return;

  if (executeMultiBlockTerminatingNativeThunk(procId, it->second, thunkState))
    return;

  if (executeSingleBlockTerminatingNativeThunk(procId, it->second, thunkState))
    return;

  if (executeCombinationalNativeThunk(procId, it->second, thunkState))
    return;

  if (auto processOp = it->second.getProcessOp()) {
    Block &body = processOp.getBody().front();
    auto opIt = body.begin();
    if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*opIt)) {
      (void)interpretProcPrint(procId, printOp);
      ++opIt;
    }
    if (auto haltOp = dyn_cast<llhd::HaltOp>(*opIt)) {
      (void)interpretHalt(procId, haltOp);
      auto post = processStates.find(procId);
      if (post != processStates.end()) {
        thunkState.halted = post->second.halted;
        thunkState.waiting = post->second.waiting;
      }
      return;
    }
  }

  if (auto initialOp = it->second.getInitialOp()) {
    Block *body = initialOp.getBodyBlock();
    auto yieldOp = dyn_cast<seq::YieldOp>(body->back());
    if (!yieldOp)
      return;
    if (!llvm::hasSingleElement(*body)) {
      auto printIt = std::prev(body->end());
      --printIt;
      if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*printIt))
        (void)interpretProcPrint(procId, printOp);
    }
    if (yieldOp) {
      (void)interpretSeqYield(procId, yieldOp);
      auto post = processStates.find(procId);
      if (post != processStates.end()) {
        thunkState.halted = post->second.halted;
        thunkState.waiting = post->second.waiting;
      }
      return;
    }
  }

  finalizeProcess(procId, /*killed=*/false);
  auto post = processStates.find(procId);
  if (post != processStates.end()) {
    thunkState.halted = post->second.halted;
    thunkState.waiting = post->second.waiting;
  }
}

bool LLHDProcessInterpreter::executeSingleBlockTerminatingNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isSingleBlockTerminatingNativeThunkCandidate(procId, state, nullptr))
    return false;

  Region *bodyRegion = resolveNativeThunkProcessRegion(state);
  if (!bodyRegion || !bodyRegion->hasOneBlock() || bodyRegion->front().empty()) {
    thunkState.deoptRequested = true;
    return true;
  }
  Block &body = bodyRegion->front();
  bool bodyContainsForkPrelude = false;
  for (auto it = body.begin(), e = std::prev(body.end()); it != e; ++it) {
    if (isa<sim::SimForkOp>(*it)) {
      bodyContainsForkPrelude = true;
      break;
    }
  }
  static bool traceThunkGuards = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_JIT_THUNK_GUARDS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto requestDeopt = [&](StringRef reason, bool restoreSnapshot) {
    if (traceThunkGuards) {
      llvm::errs() << "[JIT-THUNK-GUARD] proc=" << procId;
      if (auto *proc = scheduler.getProcess(procId))
        llvm::errs() << " name=" << proc->getName();
      llvm::errs() << " shape=single_block_terminating"
                   << " reason=" << reason;
      llvm::errs() << " state{halted=" << (state.halted ? 1 : 0)
                   << " waiting=" << (state.waiting ? 1 : 0)
                   << " resume_token=" << state.jitThunkResumeToken
                   << " call_stack=" << state.callStack.size()
                   << " seq_retry=" << (state.sequencerGetRetryCallOp ? 1 : 0)
                   << " resume_at_current_op="
                   << (state.resumeAtCurrentOp ? 1 : 0)
                   << " parent=" << state.parentProcessId;
      if (state.currentBlock) {
        llvm::errs()
            << " current_block_ops=" << state.currentBlock->getOperations().size();
        if (state.currentOp == state.currentBlock->end())
          llvm::errs() << " current_op=<end>";
        else
          llvm::errs() << " current_op=" << state.currentOp->getName().getStringRef();
      } else {
        llvm::errs() << " current_block=<null>";
      }
      if (state.destBlock) {
        llvm::errs() << " dest_block_ops=" << state.destBlock->getOperations().size()
                     << " dest_args=" << state.destOperands.size();
      } else {
        llvm::errs() << " dest_block=<null>";
      }
      llvm::errs() << " body_term=" << body.back().getName().getStringRef()
                   << "}\n";
      if (reason == "post_exec_not_halted_or_waiting") {
        llvm::errs() << "  [JIT-THUNK-GUARD] body:";
        for (Operation &op : body)
          llvm::errs() << " " << op.getName().getStringRef();
        llvm::errs() << "\n";
      }
    }
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = restoreSnapshot;
    thunkState.deoptDetail =
        (Twine("single_block_terminating:") + reason).str();
  };
  auto isDeferredHaltState = [&](const ProcessExecutionState &s) {
    if (!s.destBlock || s.destBlock != &body || !s.resumeAtCurrentOp)
      return false;
    if (!s.currentBlock || s.currentBlock != s.destBlock ||
        s.currentOp == s.currentBlock->end())
      return false;
    return isa<llhd::HaltOp>(*s.currentOp);
  };
  auto isResumingAfterSemaphoreGet = [&](const ProcessExecutionState &s) {
    if (!s.destBlock || s.destBlock != &body || !s.resumeAtCurrentOp)
      return false;
    if (s.pendingSemaphoreGetId == 0)
      return false;
    if (!s.currentBlock || s.currentBlock != s.destBlock ||
        s.currentOp == s.currentBlock->end())
      return false;
    return true;
  };
  auto normalizeDestResumeState = [&](ProcessExecutionState &s) {
    s.currentBlock = s.destBlock;
    for (auto [arg, val] :
         llvm::zip(s.currentBlock->getArguments(), s.destOperands)) {
      s.valueMap[arg] = val;
      if (isa<llhd::RefType>(arg.getType()))
        s.refBlockArgSources.erase(arg);
    }
    s.waiting = false;
    s.destBlock = nullptr;
    s.destOperands.clear();
    s.resumeAtCurrentOp = false;
    s.pendingSemaphoreGetId = 0;
  };
  auto isQueuedForImpPhaseOrdering = [&](ProcessId targetProcId) {
    for (const auto &entry : impWaitingProcesses) {
      for (const ImpWaiter &waiter : entry.second) {
        if (waiter.procId == targetProcId)
          return true;
      }
    }
    return false;
  };
  auto isAwaitingProcessCompletion = [&](ProcessId targetProcId) {
    for (const auto &entry : processAwaiters) {
      for (ProcessId waiterProcId : entry.second) {
        if (waiterProcId == targetProcId)
          return true;
      }
    }
    return false;
  };
  auto isWaitingOnForkJoinChildren = [&](const ProcessExecutionState &s) {
    if (!bodyContainsForkPrelude || !s.waiting || s.halted)
      return false;
    if (s.waitConditionRestartBlock || s.destBlock || s.resumeAtCurrentOp ||
        !s.callStack.empty() || s.sequencerGetRetryCallOp)
      return false;
    if (isQueuedForImpPhaseOrdering(procId) ||
        isAwaitingProcessCompletion(procId))
      return false;
    return forkJoinManager.hasActiveChildren(procId);
  };
  auto isResumingAfterForkJoinWait = [&](const ProcessExecutionState &s) {
    if (!bodyContainsForkPrelude || !s.waiting || s.halted)
      return false;
    if (s.waitConditionRestartBlock || s.destBlock || s.resumeAtCurrentOp ||
        !s.callStack.empty() || s.sequencerGetRetryCallOp)
      return false;
    if (isQueuedForImpPhaseOrdering(procId) ||
        isAwaitingProcessCompletion(procId))
      return false;
    return !forkJoinManager.hasActiveChildren(procId);
  };
  bool resumingDeferredHalt = false;

  // This thunk is non-resumable and only valid on token 0.
  if (thunkState.resumeToken != state.jitThunkResumeToken ||
      state.jitThunkResumeToken != 0) {
    requestDeopt("resume_token_mismatch", /*restoreSnapshot=*/true);
    return true;
  }
  if (state.destBlock || state.resumeAtCurrentOp) {
    if (!isDeferredHaltState(state) && !isResumingAfterSemaphoreGet(state)) {
      requestDeopt("unexpected_resume_state", /*restoreSnapshot=*/true);
      return true;
    }
    resumingDeferredHalt = isDeferredHaltState(state);
    normalizeDestResumeState(state);
  }
  if (state.waiting) {
    if (state.waitConditionRestartBlock && !state.halted) {
      // Ignore spurious wakeups while wait_condition polling remains armed.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (isDeferredHaltState(state)) {
      resumingDeferredHalt = true;
      normalizeDestResumeState(state);
    } else if (isResumingAfterSemaphoreGet(state)) {
      normalizeDestResumeState(state);
    } else if (!state.callStack.empty()) {
      // Some suspension paths can schedule the process with waiting still set.
      // Normalize this to the active execution path before call-stack resume.
      state.waiting = false;
    } else if (state.sequencerGetRetryCallOp) {
      // Process-level sequencer retries are resumed by rewinding to the
      // recorded call op on the next activation.
      state.waiting = false;
    } else if (isQueuedForImpPhaseOrdering(procId)) {
      // process_phase IMP-order gating parks processes in impWaitingProcesses.
      // Keep this suspension native until finish_phase explicitly wakes it.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isAwaitingProcessCompletion(procId)) {
      // process::await() suspension is resumed by notifyProcessAwaiters when
      // the awaited process terminates.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isWaitingOnForkJoinChildren(state)) {
      // Blocking fork joins park the process until child completion.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isResumingAfterForkJoinWait(state)) {
      // resumeProcess in ForkJoinManager clears scheduler waiting state but
      // does not clear interpreter waiting; mirror executeProcess behavior.
      state.waiting = false;
    } else {
      requestDeopt("unexpected_waiting_state", /*restoreSnapshot=*/true);
      return true;
    }
  }
  if (state.halted) {
    requestDeopt("unexpected_resume_state", /*restoreSnapshot=*/true);
    return true;
  }

  if (state.waitConditionRestartBlock) {
    if (!state.waitConditionSavedBlock) {
      if (!state.callStack.empty()) {
        auto &outermostFrame = state.callStack.back();
        if (outermostFrame.callOp) {
          state.waitConditionSavedBlock = outermostFrame.callOp->getBlock();
          state.waitConditionSavedOp =
              std::next(outermostFrame.callOp->getIterator());
        } else {
          state.waitConditionSavedBlock = state.currentBlock;
          state.waitConditionSavedOp = state.currentOp;
        }
      } else {
        state.waitConditionSavedBlock = state.currentBlock;
        state.waitConditionSavedOp = state.currentOp;
      }
    }
    state.currentBlock = state.waitConditionRestartBlock;
    state.currentOp = state.waitConditionRestartOp;
    for (Value value : state.waitConditionValuesToInvalidate)
      state.valueMap.erase(value);
    if (!state.callStack.empty()) {
      auto &innermostFrame = state.callStack.front();
      innermostFrame.resumeOp = state.waitConditionRestartOp;
      innermostFrame.resumeBlock = state.waitConditionRestartBlock;
    }
  } else if (!resumingDeferredHalt &&
             (!state.currentBlock || state.currentBlock->getParent() != bodyRegion ||
              state.currentOp == state.currentBlock->end())) {
    state.currentBlock = &body;
    state.currentOp = state.currentBlock->begin();
  }

  if (state.sequencerGetRetryCallOp && !state.callStack.empty()) {
    auto &innermostFrame = state.callStack.front();
    innermostFrame.resumeOp = state.sequencerGetRetryCallOp->getIterator();
    innermostFrame.resumeBlock = state.sequencerGetRetryCallOp->getBlock();
    state.sequencerGetRetryCallOp = nullptr;
  } else if (state.sequencerGetRetryCallOp && state.callStack.empty()) {
    state.currentOp = state.sequencerGetRetryCallOp->getIterator();
    state.currentBlock = state.sequencerGetRetryCallOp->getBlock();
    state.sequencerGetRetryCallOp = nullptr;
  }

  CallStackResumeResult callStackResume =
      resumeSavedCallStackFrames(procId, state);
  if (callStackResume == CallStackResumeResult::Failed) {
    if (state.halted) {
      thunkState.halted = true;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    requestDeopt("call_stack_resume_failed", /*restoreSnapshot=*/false);
    return true;
  }
  if (callStackResume == CallStackResumeResult::Suspended) {
    thunkState.halted = state.halted;
    thunkState.waiting = state.waiting;
    thunkState.resumeToken = state.jitThunkResumeToken;
    return true;
  }

  if (!resumingDeferredHalt && callStackResume == CallStackResumeResult::Completed &&
      (!state.currentBlock || state.currentBlock->getParent() != bodyRegion ||
       state.currentOp == state.currentBlock->end())) {
    state.currentBlock = &body;
    state.currentOp = state.currentBlock->begin();
  }

  if (!state.currentBlock || state.currentOp == state.currentBlock->end()) {
    requestDeopt("invalid_current_op_state", /*restoreSnapshot=*/true);
    return true;
  }

  ProcessId savedActiveProcessId = activeProcessId;
  ProcessExecutionState *savedActiveProcessState = activeProcessState;
  activeProcessId = procId;
  activeProcessState = &state;
  auto restoreActive = llvm::make_scope_exit([&]() {
    activeProcessId = savedActiveProcessId;
    activeProcessState = savedActiveProcessState;
  });

  // Allow enough headroom for call-heavy prelude ops (e.g. call_indirect
  // dispatch chains) while still bounding the native-thunk attempt.
  size_t maxSteps = std::max<size_t>(8192, body.getOperations().size() * 128);
  bool reachedStepLimit = true;
  for (size_t steps = 0; steps < maxSteps; ++steps) {
    if (!executeStep(procId)) {
      reachedStepLimit = false;
      break;
    }
  }
  if (reachedStepLimit) {
    requestDeopt("step_limit_reached", /*restoreSnapshot=*/false);
    return true;
  }

  auto post = processStates.find(procId);
  if (post == processStates.end()) {
    requestDeopt("process_state_missing_post_exec", /*restoreSnapshot=*/false);
    return true;
  }
  bool waitingOnWaitCondition = post->second.waiting &&
                                post->second.waitConditionRestartBlock &&
                                !post->second.halted;
  bool waitingOnDeferredHalt =
      post->second.waiting && isDeferredHaltState(post->second) &&
      !post->second.halted;
  bool waitingOnSequencerRetry =
      post->second.waiting && post->second.sequencerGetRetryCallOp &&
      !post->second.halted;
  bool waitingOnSavedCallStack =
      post->second.waiting && !post->second.callStack.empty() &&
      !post->second.halted;
  bool waitingOnImpOrderQueue =
      post->second.waiting && !post->second.halted &&
      post->second.callStack.empty() && !post->second.sequencerGetRetryCallOp &&
      !post->second.waitConditionRestartBlock && !post->second.destBlock &&
      !post->second.resumeAtCurrentOp && post->second.currentBlock &&
      post->second.currentBlock->getParent() == bodyRegion &&
      post->second.currentOp != post->second.currentBlock->end() &&
      isQueuedForImpPhaseOrdering(procId);
  bool waitingOnProcessAwaitQueue =
      post->second.waiting && !post->second.halted &&
      isAwaitingProcessCompletion(procId);
  bool waitingOnForkJoinChildren = isWaitingOnForkJoinChildren(post->second);
  bool waitingOnSemaphoreGet =
      post->second.waiting && !post->second.halted &&
      post->second.pendingSemaphoreGetId != 0 && post->second.destBlock &&
      post->second.destBlock == &body && post->second.resumeAtCurrentOp &&
      post->second.currentBlock == post->second.destBlock &&
      post->second.currentOp != post->second.currentBlock->end();
  if (!post->second.halted && !waitingOnWaitCondition &&
      !waitingOnDeferredHalt && !waitingOnSequencerRetry &&
      !waitingOnSavedCallStack && !waitingOnImpOrderQueue &&
      !waitingOnProcessAwaitQueue && !waitingOnForkJoinChildren &&
      !waitingOnSemaphoreGet) {
    requestDeopt("post_exec_not_halted_or_waiting", /*restoreSnapshot=*/false);
    return true;
  }
  if (post->second.waiting && !waitingOnWaitCondition &&
      !waitingOnDeferredHalt && !waitingOnSequencerRetry &&
      !waitingOnSavedCallStack && !waitingOnImpOrderQueue &&
      !waitingOnProcessAwaitQueue && !waitingOnForkJoinChildren &&
      !waitingOnSemaphoreGet) {
    requestDeopt("post_exec_waiting_without_wait_condition",
                 /*restoreSnapshot=*/false);
    return true;
  }

  thunkState.halted = post->second.halted;
  thunkState.waiting = post->second.waiting;
  thunkState.resumeToken = post->second.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::executeMultiBlockTerminatingNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isMultiBlockTerminatingNativeThunkCandidate(procId, state, nullptr))
    return false;

  static bool traceThunkGuards = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_JIT_THUNK_GUARDS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  Region *bodyRegion = resolveNativeThunkProcessRegion(state);
  auto requestDeopt = [&](StringRef reason, bool restoreSnapshot) {
    if (traceThunkGuards) {
      llvm::errs() << "[JIT-THUNK-GUARD] proc=" << procId;
      if (auto *proc = scheduler.getProcess(procId))
        llvm::errs() << " name=" << proc->getName();
      llvm::errs() << " shape=multi_block_terminating"
                   << " reason=" << reason;
      llvm::errs() << " state{halted=" << (state.halted ? 1 : 0)
                   << " waiting=" << (state.waiting ? 1 : 0)
                   << " resume_token=" << state.jitThunkResumeToken
                   << " call_stack=" << state.callStack.size()
                   << " seq_retry=" << (state.sequencerGetRetryCallOp ? 1 : 0)
                   << " resume_at_current_op="
                   << (state.resumeAtCurrentOp ? 1 : 0)
                   << " parent=" << state.parentProcessId;
      if (state.currentBlock) {
        llvm::errs()
            << " current_block_ops=" << state.currentBlock->getOperations().size();
        if (state.currentOp == state.currentBlock->end())
          llvm::errs() << " current_op=<end>";
        else
          llvm::errs() << " current_op="
                       << state.currentOp->getName().getStringRef();
      } else {
        llvm::errs() << " current_block=<null>";
      }
      if (state.destBlock) {
        llvm::errs() << " dest_block_ops=" << state.destBlock->getOperations().size()
                     << " dest_args=" << state.destOperands.size();
      } else {
        llvm::errs() << " dest_block=<null>";
      }
      llvm::errs() << "}\n";
      if ((reason == "post_exec_not_halted" ||
           reason == "post_exec_waiting_without_fork_wait") &&
          bodyRegion) {
        for (auto [blockIdx, block] : llvm::enumerate(*bodyRegion)) {
          llvm::errs() << "  [JIT-THUNK-GUARD] block " << blockIdx
                       << " term=" << block.back().getName().getStringRef()
                       << " ops:";
          for (Operation &op : block) {
            llvm::errs() << " " << op.getName().getStringRef();
            if (auto callOp = dyn_cast<mlir::func::CallOp>(op))
              llvm::errs() << "(" << callOp.getCallee() << ")";
            else if (auto llvmCall = dyn_cast<LLVM::CallOp>(op))
              if (auto callee = llvmCall.getCallee())
                llvm::errs() << "(" << *callee << ")";
          }
          llvm::errs() << "\n";
        }
      }
    }
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = restoreSnapshot;
    thunkState.deoptDetail = reason.str();
  };

  if (!bodyRegion || bodyRegion->empty() || bodyRegion->front().empty()) {
    requestDeopt("empty_body_region", /*restoreSnapshot=*/true);
    return true;
  }
  bool bodyContainsForkPrelude = false;
  for (Block &block : *bodyRegion) {
    if (block.empty())
      continue;
    for (auto it = block.begin(), e = std::prev(block.end()); it != e; ++it) {
      if (isa<sim::SimForkOp>(*it)) {
        bodyContainsForkPrelude = true;
        break;
      }
    }
    if (bodyContainsForkPrelude)
      break;
  }
  auto isWaitingOnForkJoinChildren = [&](const ProcessExecutionState &s) {
    if (!bodyContainsForkPrelude || !s.waiting || s.halted)
      return false;
    if (s.destBlock || s.resumeAtCurrentOp || s.waitConditionRestartBlock ||
        !s.callStack.empty() || s.sequencerGetRetryCallOp)
      return false;
    return forkJoinManager.hasActiveChildren(procId);
  };
  auto isResumingAfterForkJoinWait = [&](const ProcessExecutionState &s) {
    if (!bodyContainsForkPrelude || !s.waiting || s.halted)
      return false;
    if (s.destBlock || s.resumeAtCurrentOp || s.waitConditionRestartBlock ||
        !s.callStack.empty() || s.sequencerGetRetryCallOp)
      return false;
    return !forkJoinManager.hasActiveChildren(procId);
  };
  auto isWaitingOnObjectionWaitFor = [&](ProcessId targetProcId,
                                         const ProcessExecutionState &s) {
    if (!s.waiting || s.halted)
      return false;
    if (s.destBlock || s.resumeAtCurrentOp || s.waitConditionRestartBlock ||
        !s.callStack.empty() || s.sequencerGetRetryCallOp)
      return false;
    return objectionWaitForStateByProc.count(targetProcId) != 0;
  };
  auto isAwaitingProcessCompletion = [&](ProcessId targetProcId) {
    for (const auto &entry : processAwaiters) {
      for (ProcessId waiterProcId : entry.second) {
        if (waiterProcId == targetProcId)
          return true;
      }
    }
    return false;
  };
  auto isResumingAfterSemaphoreGet = [&](const ProcessExecutionState &s) {
    if (!s.destBlock || !s.resumeAtCurrentOp || s.pendingSemaphoreGetId == 0)
      return false;
    if (s.destBlock->getParent() != bodyRegion)
      return false;
    if (!s.currentBlock || s.currentBlock != s.destBlock ||
        s.currentOp == s.currentBlock->end())
      return false;
    return true;
  };
  auto normalizeDestResumeState = [&](ProcessExecutionState &s) {
    s.currentBlock = s.destBlock;
    for (auto [arg, val] :
         llvm::zip(s.currentBlock->getArguments(), s.destOperands)) {
      s.valueMap[arg] = val;
      if (isa<llhd::RefType>(arg.getType()))
        s.refBlockArgSources.erase(arg);
    }
    s.waiting = false;
    s.destBlock = nullptr;
    s.destOperands.clear();
    s.resumeAtCurrentOp = false;
    s.pendingSemaphoreGetId = 0;
  };

  // This thunk is non-resumable and only valid on token 0.
  if (thunkState.resumeToken != state.jitThunkResumeToken ||
      state.jitThunkResumeToken != 0) {
    requestDeopt("resume_token_mismatch", /*restoreSnapshot=*/true);
    return true;
  }
  if (state.destBlock || state.resumeAtCurrentOp) {
    if (!isResumingAfterSemaphoreGet(state)) {
      requestDeopt("unexpected_resume_state", /*restoreSnapshot=*/true);
      return true;
    }
    normalizeDestResumeState(state);
  }
  if (state.waiting) {
    if (isAwaitingProcessCompletion(procId)) {
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (isWaitingOnForkJoinChildren(state)) {
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (isWaitingOnObjectionWaitFor(procId, state)) {
      // wait_for-style objection polling is resumed by scheduled callbacks.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (isResumingAfterForkJoinWait(state)) {
      // resumeProcess in ForkJoinManager only touches scheduler state.
      state.waiting = false;
    } else {
      requestDeopt("unexpected_resume_state", /*restoreSnapshot=*/true);
      return true;
    }
  }

  if (!state.currentBlock || state.currentBlock->getParent() != bodyRegion ||
      state.currentOp == state.currentBlock->end()) {
    state.currentBlock = &bodyRegion->front();
    state.currentOp = state.currentBlock->begin();
  }

  ProcessId savedActiveProcessId = activeProcessId;
  ProcessExecutionState *savedActiveProcessState = activeProcessState;
  activeProcessId = procId;
  activeProcessState = &state;
  auto restoreActive = llvm::make_scope_exit([&]() {
    activeProcessId = savedActiveProcessId;
    activeProcessState = savedActiveProcessState;
  });

  size_t regionOps = 0;
  for (Block &block : *bodyRegion)
    regionOps += block.getOperations().size();

  // Multiblock lowering can include larger branchy preludes.
  size_t maxSteps = std::max<size_t>(16384, regionOps * 256);
  bool reachedStepLimit = true;
  for (size_t steps = 0; steps < maxSteps; ++steps) {
    if (!executeStep(procId)) {
      reachedStepLimit = false;
      break;
    }
  }
  if (reachedStepLimit) {
    requestDeopt("step_limit_reached", /*restoreSnapshot=*/false);
    return true;
  }

  auto post = processStates.find(procId);
  if (post == processStates.end()) {
    requestDeopt("process_state_missing_post_exec",
                 /*restoreSnapshot=*/false);
    return true;
  }
  bool waitingOnForkJoinChildren = isWaitingOnForkJoinChildren(post->second);
  bool waitingOnObjectionWaitFor =
      isWaitingOnObjectionWaitFor(procId, post->second);
  bool waitingOnProcessAwaitQueue =
      post->second.waiting && !post->second.halted &&
      isAwaitingProcessCompletion(procId);
  bool waitingOnSemaphoreGet =
      post->second.waiting && !post->second.halted &&
      post->second.pendingSemaphoreGetId != 0 && post->second.destBlock &&
      post->second.resumeAtCurrentOp &&
      post->second.destBlock->getParent() == bodyRegion &&
      post->second.currentBlock == post->second.destBlock &&
      post->second.currentOp != post->second.currentBlock->end();
  if (!post->second.halted && !waitingOnForkJoinChildren &&
      !waitingOnObjectionWaitFor && !waitingOnProcessAwaitQueue &&
      !waitingOnSemaphoreGet) {
    requestDeopt("post_exec_not_halted", /*restoreSnapshot=*/false);
    return true;
  }
  if (post->second.waiting && !waitingOnForkJoinChildren &&
      !waitingOnObjectionWaitFor && !waitingOnProcessAwaitQueue &&
      !waitingOnSemaphoreGet) {
    requestDeopt("post_exec_waiting_without_fork_wait",
                 /*restoreSnapshot=*/false);
    return true;
  }

  thunkState.halted = post->second.halted;
  thunkState.waiting = post->second.waiting;
  thunkState.resumeToken = post->second.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::executeResumableWaitSelfLoopNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isResumableWaitSelfLoopNativeThunkCandidate(procId, state, nullptr))
    return false;

  auto processOp = state.getProcessOp();
  if (!processOp) {
    thunkState.deoptRequested = true;
    return true;
  }

  Region &bodyRegion = processOp.getBody();
  Block *loopBlock = nullptr;
  if (bodyRegion.hasOneBlock()) {
    loopBlock = &bodyRegion.front();
  } else {
    Block &entry = bodyRegion.front();
    auto entryBranch = dyn_cast<mlir::cf::BranchOp>(entry.back());
    if (!entryBranch) {
      thunkState.deoptRequested = true;
      return true;
    }
    loopBlock = entryBranch.getDest();
  }
  if (!loopBlock || loopBlock->empty()) {
    thunkState.deoptRequested = true;
    return true;
  }
  auto waitOp = dyn_cast<llhd::WaitOp>(loopBlock->back());
  if (!waitOp || waitOp.getDest() != loopBlock) {
    thunkState.deoptRequested = true;
    return true;
  }

  // This resumable loop thunk expects a stable token.
  if (thunkState.resumeToken != state.jitThunkResumeToken ||
      state.jitThunkResumeToken != 0) {
    thunkState.deoptRequested = true;
    return true;
  }
  if (!state.callStack.empty()) {
    thunkState.deoptRequested = true;
    return true;
  }

  if (state.destBlock) {
    if (state.destBlock != loopBlock ||
        state.destOperands.size() != loopBlock->getNumArguments()) {
      thunkState.deoptRequested = true;
      return true;
    }
    state.currentBlock = state.destBlock;
    if (!state.resumeAtCurrentOp)
      state.currentOp = state.currentBlock->begin();
    for (auto [arg, val] :
         llvm::zip(state.currentBlock->getArguments(), state.destOperands)) {
      state.valueMap[arg] = val;
      if (isa<llhd::RefType>(arg.getType()))
        state.refBlockArgSources.erase(arg);
    }
    state.waiting = false;
    state.destBlock = nullptr;
    state.destOperands.clear();
    state.resumeAtCurrentOp = false;
  } else if (state.waiting) {
    // Some wait wakeups clear waiting before schedule, some do not.
    state.waiting = false;
  }

  if (!state.currentBlock || state.currentOp == state.currentBlock->end()) {
    state.currentBlock = &bodyRegion.front();
    state.currentOp = state.currentBlock->begin();
  }

  ProcessId savedActiveProcessId = activeProcessId;
  ProcessExecutionState *savedActiveProcessState = activeProcessState;
  activeProcessId = procId;
  activeProcessState = &state;
  auto restoreActive = llvm::make_scope_exit([&]() {
    activeProcessId = savedActiveProcessId;
    activeProcessState = savedActiveProcessState;
  });

  size_t maxSteps =
      std::max<size_t>(8192, countRegionOps(bodyRegion) * 256);
  bool reachedStepLimit = true;
  for (size_t steps = 0; steps < maxSteps; ++steps) {
    if (!executeStep(procId)) {
      reachedStepLimit = false;
      break;
    }
  }
  if (reachedStepLimit) {
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = false;
    return true;
  }

  auto post = processStates.find(procId);
  if (post == processStates.end()) {
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = false;
    return true;
  }
  if (!post->second.halted && !post->second.waiting) {
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = false;
    return true;
  }

  if (post->second.waiting && post->second.pendingDelayFs > 0) {
    SimTime currentTime = scheduler.getCurrentTime();
    SimTime targetTime = currentTime.advanceTime(post->second.pendingDelayFs);
    LLVM_DEBUG(llvm::dbgs()
               << "  Scheduling __moore_delay resumption (native thunk): "
               << post->second.pendingDelayFs << " fs from time "
               << currentTime.realTime << " to " << targetTime.realTime
               << "\n");
    post->second.pendingDelayFs = 0;
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, procId]() { resumeProcess(procId); }));
  }

  post->second.jitThunkResumeToken = 0;
  thunkState.halted = post->second.halted;
  thunkState.waiting = post->second.waiting;
  thunkState.resumeToken = post->second.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::executeResumableMultiblockWaitNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isResumableMultiblockWaitNativeThunkCandidate(procId, state, nullptr))
    return false;

  static bool traceThunkGuards = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_JIT_THUNK_GUARDS");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  auto requestDeopt = [&](StringRef reason, bool restoreSnapshot) {
    if (traceThunkGuards) {
      llvm::errs() << "[JIT-THUNK-GUARD] proc=" << procId;
      if (auto *proc = scheduler.getProcess(procId))
        llvm::errs() << " name=" << proc->getName();
      llvm::errs() << " shape=resumable_multiblock_wait"
                   << " reason=" << reason;
      llvm::errs() << " state{halted=" << (state.halted ? 1 : 0)
                   << " waiting=" << (state.waiting ? 1 : 0)
                   << " resume_token=" << state.jitThunkResumeToken
                   << " call_stack=" << state.callStack.size()
                   << " resume_at_current_op="
                   << (state.resumeAtCurrentOp ? 1 : 0)
                   << " parent=" << state.parentProcessId;
      if (state.currentBlock) {
        llvm::errs()
            << " current_block_ops=" << state.currentBlock->getOperations().size();
        if (state.currentOp == state.currentBlock->end())
          llvm::errs() << " current_op=<end>";
        else
          llvm::errs() << " current_op="
                       << state.currentOp->getName().getStringRef();
      } else {
        llvm::errs() << " current_block=<null>";
      }
      if (state.destBlock) {
        llvm::errs() << " dest_block_ops=" << state.destBlock->getOperations().size()
                     << " dest_args=" << state.destOperands.size();
      } else {
        llvm::errs() << " dest_block=<null>";
      }
      llvm::errs() << "}\n";
    }
    thunkState.deoptRequested = true;
    thunkState.restoreSnapshotOnDeopt = restoreSnapshot;
    thunkState.deoptDetail = reason.str();
  };

  Region *bodyRegion = state.destBlock ? state.destBlock->getParent() : nullptr;
  if (!bodyRegion)
    bodyRegion = resolveNativeThunkProcessRegion(state);
  if (!bodyRegion || bodyRegion->empty()) {
    requestDeopt("empty_body_region", /*restoreSnapshot=*/true);
    return true;
  }

  // Keep a stable token for resumable wait state machines.
  if (thunkState.resumeToken != state.jitThunkResumeToken ||
      state.jitThunkResumeToken != 0) {
    requestDeopt("resume_token_mismatch", /*restoreSnapshot=*/true);
    return true;
  }
  bool containsWaitConditionCall = hasMooreWaitConditionCall(*bodyRegion);

  if (state.destBlock) {
    state.currentBlock = state.destBlock;
    if (!state.resumeAtCurrentOp)
      state.currentOp = state.currentBlock->begin();
    for (auto [arg, val] :
         llvm::zip(state.currentBlock->getArguments(), state.destOperands)) {
      state.valueMap[arg] = val;
      if (isa<llhd::RefType>(arg.getType()))
        state.refBlockArgSources.erase(arg);
    }
    state.waiting = false;
    state.destBlock = nullptr;
    state.destOperands.clear();
    state.resumeAtCurrentOp = false;
  } else if (state.waiting) {
    if (containsWaitConditionCall && state.waitConditionRestartBlock &&
        !state.halted) {
      // Ignore spurious triggers while wait_condition poll state is active.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    state.waiting = false;
  }

  if (containsWaitConditionCall && state.waitConditionRestartBlock) {
    if (state.waitConditionRestartBlock->getParent() != bodyRegion) {
      requestDeopt("invalid_wait_condition_restart_block",
                   /*restoreSnapshot=*/true);
      return true;
    }
    state.currentBlock = state.waitConditionRestartBlock;
    state.currentOp = state.waitConditionRestartOp;
    for (Value value : state.waitConditionValuesToInvalidate)
      state.valueMap.erase(value);
  } else if (!state.currentBlock ||
             state.currentBlock->getParent() != bodyRegion ||
             state.currentOp == state.currentBlock->end()) {
    state.currentBlock = &bodyRegion->front();
    state.currentOp = state.currentBlock->begin();
  }

  ProcessId savedActiveProcessId = activeProcessId;
  ProcessExecutionState *savedActiveProcessState = activeProcessState;
  activeProcessId = procId;
  activeProcessState = &state;
  auto restoreActive = llvm::make_scope_exit([&]() {
    activeProcessId = savedActiveProcessId;
    activeProcessState = savedActiveProcessState;
  });

  CallStackResumeResult callStackResume =
      resumeSavedCallStackFrames(procId, state);
  if (callStackResume == CallStackResumeResult::Failed) {
    if (state.halted) {
      thunkState.halted = true;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    requestDeopt("call_stack_resume_failed", /*restoreSnapshot=*/false);
    return true;
  }
  if (callStackResume == CallStackResumeResult::Suspended) {
    thunkState.halted = state.halted;
    thunkState.waiting = state.waiting;
    thunkState.resumeToken = state.jitThunkResumeToken;
    return true;
  }

  if (callStackResume == CallStackResumeResult::Completed &&
      (!state.currentBlock || state.currentBlock->getParent() != bodyRegion ||
       state.currentOp == state.currentBlock->end())) {
    state.currentBlock = &bodyRegion->front();
    state.currentOp = state.currentBlock->begin();
  }

  size_t maxSteps =
      std::max<size_t>(8192, countRegionOps(*bodyRegion) * 256);
  bool reachedStepLimit = true;
  for (size_t steps = 0; steps < maxSteps; ++steps) {
    if (!executeStep(procId)) {
      reachedStepLimit = false;
      break;
    }
  }
  if (reachedStepLimit) {
    requestDeopt("step_limit_reached", /*restoreSnapshot=*/false);
    return true;
  }

  auto post = processStates.find(procId);
  if (post == processStates.end()) {
    requestDeopt("process_state_missing_post_exec",
                 /*restoreSnapshot=*/false);
    return true;
  }
  if (!post->second.halted && !post->second.waiting) {
    requestDeopt("post_exec_not_halted_or_waiting",
                 /*restoreSnapshot=*/false);
    return true;
  }

  if (post->second.waiting && post->second.pendingDelayFs > 0) {
    SimTime currentTime = scheduler.getCurrentTime();
    SimTime targetTime = currentTime.advanceTime(post->second.pendingDelayFs);
    LLVM_DEBUG(llvm::dbgs()
               << "  Scheduling __moore_delay resumption (native thunk): "
               << post->second.pendingDelayFs << " fs from time "
               << currentTime.realTime << " to " << targetTime.realTime
               << "\n");
    post->second.pendingDelayFs = 0;
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, procId]() { resumeProcess(procId); }));
  }

  post->second.jitThunkResumeToken = 0;
  thunkState.halted = post->second.halted;
  thunkState.waiting = post->second.waiting;
  thunkState.resumeToken = post->second.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::executeResumableWaitThenHaltNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isResumableWaitThenHaltNativeThunkCandidate(procId, state, nullptr))
    return false;

  auto processOp = state.getProcessOp();
  Block &entry = processOp.getBody().front();
  auto waitIt = std::prev(entry.end());
  auto waitOp = dyn_cast<llhd::WaitOp>(*waitIt);
  if (!waitOp) {
    thunkState.deoptRequested = true;
    return true;
  }
  llvm::SmallVector<Operation *, 8> preWaitOps;
  llvm::SmallDenseSet<Value, 8> preWaitResults;
  for (auto it = entry.begin(); it != waitIt; ++it) {
    Operation *op = &*it;
    if (!isPureResumableWaitPreludeOp(op)) {
      thunkState.deoptRequested = true;
      return true;
    }
    preWaitOps.push_back(op);
    for (Value result : op->getResults())
      preWaitResults.insert(result);
  }
  auto observed = waitOp.getObserved();
  if (preWaitOps.empty()) {
    if (!observed.empty()) {
      thunkState.deoptRequested = true;
      return true;
    }
  } else {
    for (Value obs : observed) {
      if (!preWaitResults.contains(obs)) {
        thunkState.deoptRequested = true;
        return true;
      }
    }
  }
  Block *terminalBlock = waitOp.getDest();
  auto opIt = terminalBlock->begin();
  sim::PrintFormattedProcOp printOp;
  sim::TerminateOp terminateOp;
  if (auto maybePrint = dyn_cast<sim::PrintFormattedProcOp>(*opIt)) {
    printOp = maybePrint;
    ++opIt;
  }
  if (auto maybeTerminate = dyn_cast<sim::TerminateOp>(*opIt)) {
    terminateOp = maybeTerminate;
    ++opIt;
  }
  auto haltOp = dyn_cast<llhd::HaltOp>(*opIt);
  if (!haltOp) {
    thunkState.deoptRequested = true;
    return true;
  }

  // Guard the compiled state machine token before any side effects.
  if (thunkState.resumeToken != state.jitThunkResumeToken) {
    thunkState.deoptRequested = true;
    return true;
  }

  // Token 0: first activation, execute pre-wait probes, then suspend on wait.
  if (state.jitThunkResumeToken == 0) {
    if (state.waiting || state.destBlock) {
      thunkState.deoptRequested = true;
      return true;
    }
    for (Operation *op : preWaitOps) {
      if (failed(interpretOperation(procId, op))) {
        thunkState.deoptRequested = true;
        return true;
      }
    }
    if (failed(interpretWait(procId, waitOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    state.jitThunkResumeToken = 1;
    thunkState.waiting = state.waiting;
    thunkState.halted = state.halted;
    thunkState.resumeToken = state.jitThunkResumeToken;
    return true;
  }

  // Token 1: resumed activation, run optional print and halt.
  if (state.jitThunkResumeToken == 1) {
    if (state.destBlock != terminalBlock) {
      thunkState.deoptRequested = true;
      return true;
    }

    if (state.destOperands.size() != terminalBlock->getNumArguments()) {
      thunkState.deoptRequested = true;
      return true;
    }
    for (auto [arg, val] :
         llvm::zip(terminalBlock->getArguments(), state.destOperands)) {
      state.valueMap[arg] = val;
      if (isa<llhd::RefType>(arg.getType()))
        state.refBlockArgSources.erase(arg);
    }
    state.destOperands.clear();
    state.destBlock = nullptr;
    state.resumeAtCurrentOp = false;
    state.waiting = false;
    if (printOp && failed(interpretProcPrint(procId, printOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    if (terminateOp && failed(interpretTerminate(procId, terminateOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    if (state.halted) {
      state.jitThunkResumeToken = 0;
      thunkState.halted = true;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (state.waiting) {
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }
    if (failed(interpretHalt(procId, haltOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    auto post = processStates.find(procId);
    if (post != processStates.end()) {
      post->second.jitThunkResumeToken = post->second.halted ? 0 : 1;
      thunkState.halted = post->second.halted;
      thunkState.waiting = post->second.waiting;
      thunkState.resumeToken = post->second.jitThunkResumeToken;
    }
    return true;
  }

  // Any unexpected shape/state transition requests deopt bridge fallback.
  thunkState.deoptRequested = true;
  return true;
}

bool LLHDProcessInterpreter::executeCombinationalNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  if (!isCombinationalNativeThunkCandidate(procId, state, nullptr))
    return false;

  auto combOp = state.getCombinationalOp();
  if (!combOp) {
    thunkState.deoptRequested = true;
    return true;
  }

  Block &entry = combOp.getBody().front();

  // Compile-mode combinational thunk expects a stable token.
  if (thunkState.resumeToken != state.jitThunkResumeToken ||
      state.jitThunkResumeToken != 0) {
    thunkState.deoptRequested = true;
    return true;
  }

  // If waking from a prior yield, normalize to block-entry execution.
  if (state.destBlock) {
    if (state.destBlock != &entry || !state.destOperands.empty()) {
      thunkState.deoptRequested = true;
      return true;
    }
  }
  state.currentBlock = &entry;
  state.currentOp = state.currentBlock->begin();
  state.destBlock = nullptr;
  state.destOperands.clear();
  state.resumeAtCurrentOp = false;

  // Combinational processes recompute from fresh inputs each activation.
  state.valueMap.clear();
  state.refBlockArgSources.clear();
  state.waiting = false;
  state.halted = false;

  ProcessId savedActiveProcessId = activeProcessId;
  ProcessExecutionState *savedActiveProcessState = activeProcessState;
  activeProcessId = procId;
  activeProcessState = &state;
  auto restoreActive = llvm::make_scope_exit([&]() {
    activeProcessId = savedActiveProcessId;
    activeProcessState = savedActiveProcessState;
  });

  size_t totalBodyOps = 0;
  for (Block &block : combOp.getBody())
    totalBodyOps += block.getOperations().size();
  size_t maxSteps = std::max<size_t>(64, totalBodyOps * 16);

  bool reachedStepLimit = true;
  for (size_t steps = 0; steps < maxSteps; ++steps) {
    if (!executeStep(procId)) {
      reachedStepLimit = false;
      break;
    }
  }
  if (reachedStepLimit && !state.waiting && !state.halted) {
    thunkState.deoptRequested = true;
    return true;
  }

  if (!state.waiting || state.halted) {
    // Native combinational thunks must suspend on llhd.yield.
    thunkState.deoptRequested = true;
    return true;
  }

  if (state.currentBlock == nullptr || state.currentOp == state.currentBlock->end()) {
    if (!state.destBlock) {
      thunkState.deoptRequested = true;
      return true;
    }
  }

  thunkState.halted = state.halted;
  thunkState.waiting = state.waiting;
  thunkState.resumeToken = state.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::tryBuildPeriodicToggleClockThunkSpec(
    const ProcessExecutionState &state,
    PeriodicToggleClockThunkSpec &spec) const {
  auto processOp = state.getProcessOp();
  if (!processOp)
    return false;
  Region &bodyRegion = processOp.getBody();
  if (bodyRegion.empty())
    return false;

  spec = PeriodicToggleClockThunkSpec{};

  Block &entry = bodyRegion.front();
  if (entry.empty())
    return false;
  auto entryIt = entry.begin();
  if (auto initDrive = dyn_cast<llhd::DriveOp>(*entryIt)) {
    spec.hasInitialDrive = true;
    spec.initialDriveOp = initDrive;
    ++entryIt;
    if (entryIt == entry.end())
      return false;
  }

  auto entryBr = dyn_cast<mlir::cf::BranchOp>(*entryIt);
  if (!entryBr || std::next(entryIt) != entry.end() ||
      !entryBr.getDestOperands().empty())
    return false;

  Block *loopBlock = entryBr.getDest();
  if (!loopBlock || loopBlock == &entry)
    return false;
  if (!llvm::hasNItemsOrMore(*loopBlock, 2) ||
      llvm::hasNItemsOrMore(*loopBlock, 3))
    return false;

  auto loopIt = loopBlock->begin();
  auto intToTimeOp = dyn_cast<llhd::IntToTimeOp>(*loopIt++);
  auto waitOp = dyn_cast<llhd::WaitOp>(*loopIt);
  if (!intToTimeOp || !waitOp || waitOp.getDelay() != intToTimeOp.getResult() ||
      !waitOp.getObserved().empty() || !waitOp.getYieldOperands().empty() ||
      !waitOp.getDestOperands().empty())
    return false;

  auto extractConstU64 = [&](Value value, uint64_t &out) -> bool {
    out = 0;
    if (auto hwConst = value.getDefiningOp<hw::ConstantOp>()) {
      out = hwConst.getValue().getZExtValue();
      return true;
    }
    if (auto arithConst = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
        out = intAttr.getValue().getZExtValue();
        return true;
      }
      return false;
    }
    if (auto llvmConst = value.getDefiningOp<LLVM::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(llvmConst.getValue())) {
        out = intAttr.getValue().getZExtValue();
        return true;
      }
      return false;
    }
    return false;
  };

  uint64_t delayFs = 0;
  if (!extractConstU64(intToTimeOp.getInput(), delayFs) || delayFs == 0)
    return false;

  Block *toggleBlock = waitOp.getDest();
  if (!toggleBlock || toggleBlock == loopBlock)
    return false;
  if (!llvm::hasNItemsOrMore(*toggleBlock, 4) ||
      llvm::hasNItemsOrMore(*toggleBlock, 5))
    return false;

  auto toggleIt = toggleBlock->begin();
  auto probeOp = dyn_cast<llhd::ProbeOp>(*toggleIt++);
  Operation *toggleOp = &*toggleIt++;
  auto toggleDriveOp = dyn_cast<llhd::DriveOp>(*toggleIt++);
  auto toggleBr = dyn_cast<mlir::cf::BranchOp>(*toggleIt);
  if (!probeOp || !toggleDriveOp || !toggleBr ||
      std::next(toggleIt) != toggleBlock->end() ||
      toggleBr.getDest() != loopBlock || !toggleBr.getDestOperands().empty())
    return false;

  if (toggleOp->getNumResults() != 1 ||
      toggleDriveOp.getValue() != toggleOp->getResult(0))
    return false;
  if (probeOp.getSignal() != toggleDriveOp.getSignal())
    return false;
  if (spec.hasInitialDrive &&
      spec.initialDriveOp.getSignal() != toggleDriveOp.getSignal())
    return false;
  if (isa<llhd::WaitOp, llhd::HaltOp, mlir::cf::BranchOp,
          mlir::cf::CondBranchOp, mlir::func::CallOp,
          mlir::func::CallIndirectOp, LLVM::CallOp>(toggleOp))
    return false;

  spec.intToTimeResult = intToTimeOp.getResult();
  spec.waitOp = waitOp;
  spec.waitDestBlock = toggleBlock;
  spec.probeOp = probeOp;
  spec.toggleOp = toggleOp;
  spec.toggleDriveOp = toggleDriveOp;
  spec.delayFs = delayFs;
  return true;
}

bool LLHDProcessInterpreter::executePeriodicToggleClockNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  auto specIt = periodicToggleClockThunkSpecs.find(procId);
  if (specIt == periodicToggleClockThunkSpecs.end())
    return false;
  auto &spec = specIt->second;

  if (state.halted) {
    state.jitThunkResumeToken = 0;
    thunkState.halted = true;
    thunkState.waiting = false;
    thunkState.resumeToken = 0;
    return true;
  }

  // Guard the compiled state machine token before any side effects.
  if (thunkState.resumeToken != state.jitThunkResumeToken) {
    thunkState.deoptRequested = true;
    return true;
  }

  auto scheduleNextWait = [&]() -> bool {
    unsigned delayWidth = std::max(1u, getTypeWidth(spec.intToTimeResult.getType()));
    setValue(procId, spec.intToTimeResult,
             InterpretedValue(llvm::APInt(delayWidth, spec.delayFs)));
    if (failed(interpretWait(procId, spec.waitOp))) {
      thunkState.deoptRequested = true;
      return false;
    }
    state.jitThunkResumeToken = 1;
    thunkState.halted = state.halted;
    thunkState.waiting = state.waiting;
    thunkState.resumeToken = state.jitThunkResumeToken;
    return true;
  };

  // Token 0: first activation executes optional initial drive, then waits.
  if (state.jitThunkResumeToken == 0) {
    if (state.waiting || state.destBlock) {
      thunkState.deoptRequested = true;
      return true;
    }
    if (spec.hasInitialDrive &&
        failed(interpretDrive(procId, spec.initialDriveOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    (void)scheduleNextWait();
    return true;
  }

  // Token 1: resumed activation, toggle drive value, then wait again.
  if (state.jitThunkResumeToken == 1) {
    if (state.destBlock != spec.waitDestBlock) {
      thunkState.deoptRequested = true;
      return true;
    }
    state.waiting = false;
    if (failed(interpretProbe(procId, spec.probeOp)) ||
        failed(interpretOperation(procId, spec.toggleOp)) ||
        failed(interpretDrive(procId, spec.toggleDriveOp))) {
      thunkState.deoptRequested = true;
      return true;
    }
    (void)scheduleNextWait();
    return true;
  }

  thunkState.deoptRequested = true;
  return true;
}

LLHDProcessInterpreter::CallStackResumeResult
LLHDProcessInterpreter::resumeSavedCallStackFrames(
    ProcessId procId, ProcessExecutionState &state) {
  if (state.callStack.empty())
    return CallStackResumeResult::NoFrames;

  LLVM_DEBUG(llvm::dbgs()
             << "  Process has " << state.callStack.size()
             << " saved call stack frame(s), resuming from innermost\n");

  // Preserve the outermost call site for restoring process-level position
  // once all nested frames have completed.
  Operation *outermostCallOp = state.callStack.back().callOp;
  while (!state.callStack.empty()) {
    CallStackFrame frame = std::move(state.callStack.front());
    state.callStack.erase(state.callStack.begin());

    // Remaining old outer frames before resuming this frame.
    size_t oldFrameCount = state.callStack.size();

    llvm::StringRef frameName =
        frame.isLLVM() ? frame.llvmFuncOp.getName() : frame.funcOp.getName();
    LLVM_DEBUG(llvm::dbgs()
               << "    Resuming " << (frame.isLLVM() ? "LLVM " : "")
               << "function '" << frameName << "' (remaining old frames: "
               << oldFrameCount << ")\n");

    llvm::SmallVector<InterpretedValue, 4> results;
    ++state.callDepth;
    LogicalResult funcResult =
        frame.isLLVM()
            ? interpretLLVMFuncBody(procId, frame.llvmFuncOp, frame.args,
                                    results, frame.callOperands, frame.callOp,
                                    frame.resumeBlock, frame.resumeOp)
            : interpretFuncBody(procId, frame.funcOp, frame.args, results,
                                frame.callOp, frame.resumeBlock, frame.resumeOp);
    --state.callDepth;

    if (failed(funcResult)) {
      LLVM_DEBUG(llvm::dbgs() << "    Function '" << frameName
                              << "' failed during resume\n");
      finalizeProcess(procId, /*killed=*/false);
      return CallStackResumeResult::Failed;
    }

    // Function suspended again: rotate newly created inner frames to front so
    // the next resume continues innermost-first.
    if (state.waiting) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Function '" << frameName
                 << "' suspended again during resume, "
                 << "callStack size=" << state.callStack.size()
                 << " (old outer: " << oldFrameCount << ")\n");
      if (oldFrameCount > 0 && state.callStack.size() > oldFrameCount) {
        std::rotate(state.callStack.begin(),
                    state.callStack.begin() + oldFrameCount,
                    state.callStack.end());
        LLVM_DEBUG(llvm::dbgs()
                   << "    Rotated stack: moved " << oldFrameCount
                   << " old frames after "
                   << (state.callStack.size() - oldFrameCount)
                   << " new frames\n");
      }

      if (state.pendingDelayFs > 0) {
        SimTime currentTime = scheduler.getCurrentTime();
        SimTime targetTime = currentTime.advanceTime(state.pendingDelayFs);
        LLVM_DEBUG(llvm::dbgs()
                   << "    Scheduling delay " << state.pendingDelayFs
                   << " fs from function suspend\n");
        state.pendingDelayFs = 0;
        scheduler.getEventScheduler().schedule(
            targetTime, SchedulingRegion::Active,
            Event([this, procId]() { resumeProcess(procId); }));
      }
      return CallStackResumeResult::Suspended;
    }

    // Function completed: write return values to the call operation results.
    if (frame.callOp) {
      if (auto callIndirectOp = dyn_cast<mlir::func::CallIndirectOp>(frame.callOp)) {
        for (auto [result, retVal] : llvm::zip(callIndirectOp.getResults(), results))
          setValue(procId, result, retVal);
      } else if (auto callOp = dyn_cast<mlir::func::CallOp>(frame.callOp)) {
        for (auto [result, retVal] : llvm::zip(callOp.getResults(), results))
          setValue(procId, result, retVal);
      } else if (auto llvmCallOp = dyn_cast<LLVM::CallOp>(frame.callOp)) {
        for (auto [result, retVal] : llvm::zip(llvmCallOp.getResults(), results))
          setValue(procId, result, retVal);
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "    Function '" << frameName
               << "' completed, continuing to next frame\n");

    // Drop stale frames added during the completed frame's execution.
    if (state.callStack.size() > oldFrameCount) {
      size_t newFrames = state.callStack.size() - oldFrameCount;
      LLVM_DEBUG(llvm::dbgs()
                 << "    Removing " << newFrames
                 << " stale frames added by completed function\n");
      state.callStack.resize(oldFrameCount);
    }
  }

  if (state.waitConditionSavedBlock) {
    state.currentBlock = state.waitConditionSavedBlock;
    state.currentOp = state.waitConditionSavedOp;
    state.waitConditionSavedBlock = nullptr;
  } else if (outermostCallOp) {
    state.currentBlock = outermostCallOp->getBlock();
    state.currentOp = std::next(outermostCallOp->getIterator());
  }
  state.waitConditionRestartBlock = nullptr;

  LLVM_DEBUG(llvm::dbgs()
             << "  Call stack frames exhausted, continuing process\n");
  return CallStackResumeResult::Completed;
}
