//===- LLHDProcessInterpreterNativeThunkExec.cpp - Native thunk execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "JITBlockCompiler.h"
#include "JITCompileManager.h"
#include "JITSchedulerRuntime.h"
#include "circt/Dialect/Comb/CombOps.h"
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

static bool isDirectResumableWaitSelfLoopPreludeOp(Operation *op) {
  if (!op)
    return false;

  if (isa<llhd::ProbeOp, llhd::DriveOp, LLVM::StoreOp>(op))
    return true;

  if (isa<llhd::WaitOp, llhd::YieldOp, llhd::HaltOp, mlir::cf::BranchOp,
          mlir::cf::CondBranchOp, mlir::func::CallOp,
          mlir::func::CallIndirectOp, LLVM::CallOp, sim::PrintFormattedProcOp,
          sim::SimForkOp, sim::SimJoinOp, sim::SimJoinAnyOp,
          sim::SimWaitForkOp, sim::SimDisableForkOp,
          sim::SimForkTerminatorOp>(op))
    return false;

  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;

  if (isa<LLVM::AllocaOp, LLVM::GEPOp, LLVM::LoadOp, LLVM::AddressOfOp>(op))
    return true;

  if (op->getNumResults() == 0)
    return false;
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

    // Block-level JIT: identify and compile a hot block, but DON'T add to
    // kindMask yet. The first activation must be handled by the interpreter
    // (or another thunk) to execute the entry block. The JIT kind will be
    // activated on the second invocation once jitThunkResumeToken >= 1.
    if (blockJITEnabled && jitBlockCompiler) {
      auto processOp = state.getProcessOp();
      if (processOp) {
        JITBlockSpec blockSpec;
        if (jitBlockCompiler->identifyHotBlock(processOp, valueToSignal,
                                               scheduler, blockSpec)) {
          if (jitBlockCompiler->compileBlock(blockSpec, state.getProcessOp()
                                                 ->getParentOfType<mlir::ModuleOp>())) {
            jitBlockSpecs[procId] = std::make_unique<JITBlockSpec>(std::move(blockSpec));
            LLVM_DEBUG(llvm::dbgs()
                       << "[JIT] Compiled block for proc=" << procId
                       << " (deferred activation)\n");
          }
        }
      }
    }

    directProcessFastPathKinds[procId] = kindMask;
  } else {
    kindMask = kindIt->second;

    // Deferred JIT activation: after the first activation (handled by
    // interpreter or other thunks), add JIT kind if we have a compiled block
    // and the process has been through at least one wait cycle.
    if (!(kindMask &
          toFastPathMask(DirectProcessFastPathKind::JITCompiledBlock)) &&
        state.jitThunkResumeToken >= 1) {
      auto specIt = jitBlockSpecs.find(procId);
      if (specIt != jitBlockSpecs.end() && specIt->second &&
          specIt->second->nativeFunc) {
        kindMask |=
            toFastPathMask(DirectProcessFastPathKind::JITCompiledBlock);
        directProcessFastPathKinds[procId] = kindMask;
        LLVM_DEBUG(llvm::dbgs()
                   << "[JIT] Activated JIT block for proc=" << procId << "\n");
      }
    }
  }

  if (kindMask == 0)
    return false;

  auto clearKind = [&](DirectProcessFastPathKind kind) {
    kindMask &= ~toFastPathMask(kind);
    directProcessFastPathKinds[procId] = kindMask;
    if ((kind == DirectProcessFastPathKind::JITCompiledBlock) &&
        ((kindMask &
          toFastPathMask(DirectProcessFastPathKind::JITCompiledBlock)) == 0))
      jitBlockSpecs.erase(procId);
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

  // Try PeriodicToggleClock first — it's a specialized pattern that
  // bypasses the interpreter entirely with direct signal memory access.
  if (kindMask & toFastPathMask(DirectProcessFastPathKind::PeriodicToggleClock))
    if (tryKind(DirectProcessFastPathKind::PeriodicToggleClock,
                [this](ProcessId id, ProcessExecutionState &s,
                       ProcessThunkExecutionState &thunkState) {
                  return executePeriodicToggleClockNativeThunk(id, s,
                                                               thunkState);
                }))
      return true;

  // JIT-compiled block: general native code with scheduler callbacks.
  if (kindMask & toFastPathMask(DirectProcessFastPathKind::JITCompiledBlock))
    if (tryKind(DirectProcessFastPathKind::JITCompiledBlock,
                [this](ProcessId id, ProcessExecutionState &s,
                       ProcessThunkExecutionState &thunkState) {
                  return executeJITCompiledBlockNativeThunk(id, s, thunkState);
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
  static bool traceThunkGuards = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_JIT_THUNK_GUARDS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (forceJitThunkDeoptRequests) {
    thunkState.deoptRequested = true;
    return;
  }

  auto it = processStates.find(procId);
  if (it == processStates.end() || it->second.halted)
    return;

  auto requestTrivialDeopt = [&](StringRef reason) {
    if (traceThunkGuards) {
      llvm::errs() << "[JIT-THUNK-GUARD] proc=" << procId;
      if (auto *proc = scheduler.getProcess(procId))
        llvm::errs() << " name=" << proc->getName();
      llvm::errs() << " shape=trivial reason=" << reason;
      llvm::errs() << " state{halted=" << (it->second.halted ? 1 : 0)
                   << " waiting=" << (it->second.waiting ? 1 : 0)
                   << " resume_token=" << it->second.jitThunkResumeToken
                   << " call_stack=" << it->second.callStack.size()
                   << " seq_retry="
                   << (it->second.sequencerGetRetryCallOp ? 1 : 0)
                   << " parent=" << it->second.parentProcessId << "}\n";
    }
    thunkState.deoptRequested = true;
    thunkState.deoptDetail = (Twine("trivial_thunk:") + reason).str();
  };

  auto guardIt = jitProcessThunkIndirectSiteGuards.find(procId);
  if (guardIt != jitProcessThunkIndirectSiteGuards.end()) {
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

  // If resumable thunk shapes did not consume an active call stack, defer to
  // interpreter replay rather than running trivial inline execution.
  if (!it->second.callStack.empty()) {
    requestTrivialDeopt("call_stack_active");
    return;
  }

  if (auto processOp = it->second.getProcessOp()) {
    Block &body = processOp.getBody().front();
    if (body.empty()) {
      requestTrivialDeopt("process_body_empty");
      return;
    }

    auto opIt = body.begin();
    if (auto haltOp = dyn_cast<llhd::HaltOp>(*opIt)) {
      if (std::next(opIt) != body.end()) {
        requestTrivialDeopt("process_halt_extra_ops");
        return;
      }
      (void)interpretHalt(procId, haltOp);
      auto post = processStates.find(procId);
      if (post != processStates.end()) {
        thunkState.halted = post->second.halted;
        thunkState.waiting = post->second.waiting;
      }
      return;
    }

    if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*opIt)) {
      auto nextIt = std::next(opIt);
      if (nextIt == body.end() || !isa<llhd::HaltOp>(*nextIt) ||
          std::next(nextIt) != body.end()) {
        requestTrivialDeopt("process_print_halt_shape_mismatch");
        return;
      }
      (void)interpretProcPrint(procId, printOp);
      auto haltOp = cast<llhd::HaltOp>(*nextIt);
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
    if (!body || body->empty()) {
      requestTrivialDeopt("initial_body_empty");
      return;
    }

    auto yieldOp = dyn_cast<seq::YieldOp>(body->back());
    if (!yieldOp || !yieldOp.getOperands().empty()) {
      requestTrivialDeopt("initial_yield_shape_mismatch");
      return;
    }

    sim::PrintFormattedProcOp printOp = nullptr;
    bool validInitialShape = true;
    if (!llvm::hasSingleElement(*body)) {
      auto printIt = std::prev(body->end());
      --printIt;
      printOp = dyn_cast<sim::PrintFormattedProcOp>(*printIt);
      if (!printOp) {
        validInitialShape = false;
      } else {
        for (auto it = body->begin(), e = printIt; it != e; ++it) {
          Operation *op = &*it;
          if (op->getName().getStringRef().starts_with("sim.fmt."))
            continue;
          if (isa<hw::ConstantOp, arith::ConstantOp, LLVM::ConstantOp>(op))
            continue;
          validInitialShape = false;
          break;
        }
      }
    }
    if (!validInitialShape) {
      requestTrivialDeopt("initial_print_yield_shape_mismatch");
      return;
    }

    if (printOp)
      (void)interpretProcPrint(procId, printOp);
    (void)interpretSeqYield(procId, yieldOp);
    auto post = processStates.find(procId);
    if (post != processStates.end()) {
      thunkState.halted = post->second.halted;
      thunkState.waiting = post->second.waiting;
    }
    return;
  }

  requestTrivialDeopt("fallback_shape");
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
    if (!state.callStack.empty()) {
      // Suspension through nested call/call_indirect replay can schedule this
      // process with waiting still set. Mirror single-block thunk behavior and
      // resume from the saved call-stack frame.
      state.waiting = false;
    } else if (state.sequencerGetRetryCallOp) {
      // Sequencer retries are resumed by rewinding to the saved call op.
      state.waiting = false;
    } else if (isAwaitingProcessCompletion(procId)) {
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isWaitingOnForkJoinChildren(state)) {
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isWaitingOnObjectionWaitFor(procId, state)) {
      // wait_for-style objection polling is resumed by scheduled callbacks.
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    } else if (isResumingAfterForkJoinWait(state)) {
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
  bool waitingOnSavedCallStack =
      post->second.waiting && !post->second.callStack.empty() &&
      !post->second.halted;
  bool waitingOnSequencerRetry =
      post->second.waiting && post->second.sequencerGetRetryCallOp &&
      !post->second.halted;
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
      !waitingOnObjectionWaitFor && !waitingOnSavedCallStack &&
      !waitingOnSequencerRetry && !waitingOnProcessAwaitQueue &&
      !waitingOnSemaphoreGet) {
    requestDeopt("post_exec_not_halted", /*restoreSnapshot=*/false);
    return true;
  }
  if (post->second.waiting && !waitingOnForkJoinChildren &&
      !waitingOnObjectionWaitFor && !waitingOnSavedCallStack &&
      !waitingOnSequencerRetry && !waitingOnProcessAwaitQueue &&
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
  Block *loopBlock = nullptr;
  Region &bodyRegion = processOp.getBody();
  if (bodyRegion.hasOneBlock()) {
    loopBlock = &bodyRegion.front();
  } else {
    Block &entry = bodyRegion.front();
    if (!llvm::hasSingleElement(entry)) {
      thunkState.deoptRequested = true;
      return true;
    }
    auto entryBranch = dyn_cast<mlir::cf::BranchOp>(entry.back());
    if (!entryBranch || !entryBranch.getDestOperands().empty()) {
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
  if (!waitOp || waitOp.getDest() != loopBlock ||
      waitOp.getDestOperands().size() != loopBlock->getNumArguments() ||
      !waitOp.getYieldOperands().empty()) {
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

  // Common hot path: one loop block with non-suspending prelude ops ending in
  // a self-looping llhd.wait. Execute directly without executeStep() per-op
  // dispatch to reduce overhead in edge-triggered mirror loops.
  bool canExecuteDirectLinearLoop = true;
  for (auto it = loopBlock->begin(), e = std::prev(loopBlock->end()); it != e;
       ++it) {
    if (!isDirectResumableWaitSelfLoopPreludeOp(&*it)) {
      canExecuteDirectLinearLoop = false;
      break;
    }
  }

  if (canExecuteDirectLinearLoop) {
    state.currentBlock = loopBlock;
    for (auto it = loopBlock->begin(), e = std::prev(loopBlock->end()); it != e;
         ++it) {
      state.currentOp = it;
      state.lastOp = &*it;
      if (failed(interpretOperation(procId, &*it))) {
        thunkState.deoptRequested = true;
        thunkState.restoreSnapshotOnDeopt = false;
        return true;
      }
    }

    state.currentOp = std::prev(loopBlock->end());
    state.lastOp = &*state.currentOp;
    if (failed(interpretWait(procId, waitOp))) {
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

  // Pre-resolve signal ID, toggle constant, and drive delay for the native
  // fast path. This enables bypassing the interpreter entirely at execution.
  spec.signalId = getSignalId(probeOp.getSignal());
  if (spec.signalId != 0) {
    spec.signalWidth = scheduler.getSignalValue(spec.signalId).getWidth();
    // Extract the XOR constant from the toggle op operands.
    bool xorResolved = false;
    if (isa<comb::XorOp>(toggleOp) && toggleOp->getNumOperands() == 2) {
      Value constOperand =
          (toggleOp->getOperand(0) == probeOp.getResult())
              ? toggleOp->getOperand(1) : toggleOp->getOperand(0);
      uint64_t xorConst = 0;
      if (extractConstU64(constOperand, xorConst) && spec.signalWidth <= 64) {
        spec.toggleXorConstant = xorConst;
        xorResolved = true;
      }
    }
    // Extract the drive delay from the drive op (NOT the wait delay).
    // Clock toggles typically use epsilon delay for the drive.
    bool driveDelayResolved = false;
    if (xorResolved && toggleDriveOp.getTime()) {
      if (auto constTime =
              toggleDriveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>()) {
        SimTime driveDelay = convertTime(constTime.getValueAttr());
        spec.driveRealFs = driveDelay.realTime;
        spec.driveDelta = driveDelay.deltaStep;
        spec.driveEpsilon = 0; // Already folded into deltaStep by convertTime
        driveDelayResolved = true;
      }
    }
    spec.nativePathAvailable = xorResolved && driveDelayResolved;
  }

  return true;
}

bool LLHDProcessInterpreter::executeJITCompiledBlockNativeThunk(
    ProcessId procId, ProcessExecutionState &state,
    ProcessThunkExecutionState &thunkState) {
  auto specIt = jitBlockSpecs.find(procId);
  if (specIt == jitBlockSpecs.end() || !specIt->second)
    return false;
  auto &spec = *specIt->second;

  if (!spec.nativeFunc) {
    thunkState.deoptRequested = true;
    thunkState.deoptDetail = "jit_compiled_block:null_func";
    return true;
  }

  if (state.halted) {
    thunkState.halted = true;
    thunkState.waiting = false;
    return true;
  }

  // The first activation is always handled by the interpreter (deferred
  // activation). The JIT thunk only handles subsequent activations where the
  // process has resumed from a wait into the hot block.
  if (state.destBlock != spec.hotBlock) {
    thunkState.deoptRequested = true;
    thunkState.deoptDetail = "jit_compiled_block:dest_block_mismatch";
    return true;
  }

  state.waiting = false;

  // Build the argument array for the packed calling convention.
  // Arguments are: [read_handle_0, ..., read_handle_N, drive_handle_0, ..., drive_handle_M]
  llvm::SmallVector<void *, 8> argPtrs;
  // Pack signal handles as pointers.
  llvm::SmallVector<void *, 8> handleStorage;
  for (SignalId sigId : spec.signalReads)
    handleStorage.push_back(
        reinterpret_cast<void *>(static_cast<uintptr_t>(sigId)));
  for (SignalId sigId : spec.signalDrives)
    handleStorage.push_back(
        reinterpret_cast<void *>(static_cast<uintptr_t>(sigId)));

  // For the packed calling convention, each argument is passed as void**.
  for (auto &h : handleStorage)
    argPtrs.push_back(&h);

  // Set up the JIT runtime context.
  JITRuntimeContext ctx;
  ctx.scheduler = &scheduler;
  ctx.processId = procId;
  setJITRuntimeContext(&ctx);

  // Call the JIT-compiled native function.
  // The packed convention passes an array of void* pointers to arguments.
  LLVM_DEBUG(llvm::dbgs() << "[JIT] Executing " << spec.funcName
                          << " proc=" << procId
                          << " reads=" << spec.signalReads.size()
                          << " drives=" << spec.signalDrives.size()
                          << " nargs=" << argPtrs.size() << "\n");
  spec.nativeFunc(argPtrs.data());

  clearJITRuntimeContext();

  // Re-enter wait state. Set up the delay value for the wait op.
  if (spec.waitOp.getDelay()) {
    Value delayVal = spec.waitOp.getDelay();
    if (auto constTime =
            delayVal.getDefiningOp<llhd::ConstantTimeOp>()) {
      SimTime delay = convertTime(constTime.getValueAttr());
      unsigned delayWidth =
          std::max(1u, getTypeWidth(delayVal.getType()));
      setValue(procId, delayVal,
               InterpretedValue(llvm::APInt(delayWidth, delay.realTime)));
    } else if (auto intToTime =
                   delayVal.getDefiningOp<llhd::IntToTimeOp>()) {
      // IntToTimeOp: get the integer operand value from the interpreter.
      Value intVal = intToTime.getOperand();
      if (auto constOp = intVal.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = llvm::cast<mlir::IntegerAttr>(constOp.getValue());
        unsigned delayWidth =
            std::max(1u, getTypeWidth(delayVal.getType()));
        setValue(procId, delayVal,
                 InterpretedValue(llvm::APInt(delayWidth,
                     intAttr.getValue().getZExtValue())));
      }
    }
  }
  if (failed(interpretWait(procId, spec.waitOp))) {
    thunkState.deoptRequested = true;
    thunkState.deoptDetail = "jit_compiled_block:wait_failed";
    return true;
  }

  state.jitThunkResumeToken = 1;
  thunkState.halted = state.halted;
  thunkState.waiting = state.waiting;
  thunkState.resumeToken = state.jitThunkResumeToken;
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

    // NATIVE FAST PATH: bypass interpreter entirely for direct signal access.
    // Reads signal value from scheduler's flat array, XORs with pre-resolved
    // constant, schedules drive event and process wake without any DenseMap
    // lookups, APInt allocation, or MLIR op interpretation.
    //
    // Lazy resolution: if signalId wasn't resolved at spec-build time
    // (common in compile mode where thunks install before signal registration),
    // resolve it now on first use.
    if (!spec.nativePathAvailable && spec.signalId == 0) {
      spec.signalId = getSignalId(spec.probeOp.getSignal());
      if (spec.signalId != 0) {
        spec.signalWidth =
            scheduler.getSignalValue(spec.signalId).getWidth();
        bool xorOk = false;
        if (isa<comb::XorOp>(spec.toggleOp) &&
            spec.toggleOp->getNumOperands() == 2 &&
            spec.signalWidth <= 64) {
          Value constOperand =
              (spec.toggleOp->getOperand(0) == spec.probeOp.getResult())
                  ? spec.toggleOp->getOperand(1)
                  : spec.toggleOp->getOperand(0);
          uint64_t xorConst = 0;
          if (auto hwConst = constOperand.getDefiningOp<hw::ConstantOp>()) {
            spec.toggleXorConstant = hwConst.getValue().getZExtValue();
            xorOk = true;
          } else if (auto arithConst =
                         constOperand.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
              spec.toggleXorConstant = intAttr.getValue().getZExtValue();
              xorOk = true;
            }
          }
        }
        // Also resolve drive delay.
        bool driveOk = false;
        if (xorOk && spec.toggleDriveOp.getTime()) {
          if (auto constTime = spec.toggleDriveOp.getTime()
                                   .getDefiningOp<llhd::ConstantTimeOp>()) {
            SimTime dd = convertTime(constTime.getValueAttr());
            spec.driveRealFs = dd.realTime;
            spec.driveDelta = dd.deltaStep;
            spec.driveEpsilon = 0;
            driveOk = true;
          }
        }
        spec.nativePathAvailable = xorOk && driveOk;
      }
    }
    if (spec.nativePathAvailable) {
      // Cache Process pointer on first use for O(1) scheduling.
      if (!spec.cachedProcess)
        spec.cachedProcess = scheduler.getProcessDirect(procId);

      // FAST PATH: batched clock toggle.
      // When no other processes are sensitive to this clock signal, we can
      // execute N half-cycles in a tight loop without Event/TimeWheel overhead.
      SimTime currentTime = scheduler.getCurrentTime();
      auto *schedPtr = &scheduler;
      SignalId sigId = spec.signalId;
      uint32_t sigWidth = spec.signalWidth;
      uint64_t rawVal = scheduler.readSignalValueFast(sigId);

      // Determine batch size. Batch only when this clock process is the ONLY
      // active process in the entire simulation. This ensures no other process
      // could become sensitive to the clock during the batch period.
      size_t batchCount = 1;
      uint32_t apc = scheduler.getActiveProcessCount();
      if (apc <= 1) {
        // Only the clock process is alive. Safe to batch.
        batchCount = 4096;
        // Limit batch to not exceed max simulation time.
        uint64_t maxSimTime = scheduler.getMaxSimTime();
        if (maxSimTime > 0 && currentTime.realTime < maxSimTime) {
          uint64_t remaining = maxSimTime - currentTime.realTime;
          size_t maxBatch = remaining / spec.delayFs;
          if (maxBatch < batchCount)
            batchCount = std::max(maxBatch, (size_t)1);
        }
      }
      // Execute N-1 toggles inline. Since no process is sensitive, we
      // only need to update the raw signal value — no triggerSensitiveProcesses,
      // no event scheduling, no signalChangeCallback. Just flip the bit.
      uint64_t toggled = rawVal;
      uint64_t timeFs = currentTime.realTime;
      for (size_t i = 0; i < batchCount - 1; ++i) {
        toggled ^= spec.toggleXorConstant;
        timeFs += spec.delayFs;
      }
      // Write the accumulated pre-final value to the signal state.
      // Since we verified no one is listening, use writeSignalValueRaw
      // which skips triggerSensitiveProcesses and callbacks entirely.
      if (batchCount > 1) {
        scheduler.writeSignalValueRaw(sigId, toggled);
      }
      // Final toggle — this one schedules events normally.
      toggled ^= spec.toggleXorConstant;

      if (spec.driveRealFs == 0 && spec.driveDelta == 0 &&
          spec.driveEpsilon == 0) {
        scheduler.updateSignalFast(sigId, toggled, sigWidth);
      } else {
        SimTime driveTime(timeFs + spec.delayFs, 0, 0);
        if (spec.driveRealFs > 0)
          driveTime = SimTime(timeFs, 0, 0).advanceTime(spec.driveRealFs);
        uint32_t combinedDelta = spec.driveDelta + spec.driveEpsilon;
        if (combinedDelta > 0 && spec.driveRealFs == 0)
          driveTime.deltaStep = combinedDelta;
        driveTime.region = static_cast<uint8_t>(SchedulingRegion::NBA);
        scheduler.getEventScheduler().schedule(
            driveTime, SchedulingRegion::NBA,
            Event([schedPtr, sigId, toggled, sigWidth]() {
              schedPtr->updateSignalFast(sigId, toggled, sigWidth);
            }));
      }
      // Schedule process resumption at the final wake time.
      SimTime wakeTime(timeFs + spec.delayFs, 0, 0);
      Process *cachedProc = spec.cachedProcess;
      scheduler.getEventScheduler().schedule(
          wakeTime, SchedulingRegion::Active,
          Event([schedPtr, procId, cachedProc]() {
            schedPtr->scheduleProcessDirect(procId, cachedProc);
          }));
      state.destBlock = spec.waitDestBlock;
      state.destOperands.clear();
      state.waiting = true;
      state.jitThunkResumeToken = 1;
      thunkState.halted = state.halted;
      thunkState.waiting = state.waiting;
      thunkState.resumeToken = state.jitThunkResumeToken;
      return true;
    }

    // Fallback: use interpreter for non-XOR toggle ops or unresolved signals.
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
  static bool traceI3CCallStack = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_CALLSTACK");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  if (state.callStack.empty()) {
    state.callStackOutermostCallOp = nullptr;
    return CallStackResumeResult::NoFrames;
  }

  LLVM_DEBUG(llvm::dbgs()
             << "  Process has " << state.callStack.size()
             << " saved call stack frame(s), resuming from innermost\n");

  // Preserve the outermost call site for restoring process-level position
  // once all nested frames have completed.
  Operation *outermostCallOp = state.callStackOutermostCallOp
                                   ? state.callStackOutermostCallOp
                                   : state.callStack.back().callOp;
  enum class FastResumeAction : uint8_t {
    NotHandled,
    Continue,
    Suspended,
    Failed,
  };
  static bool traceBaudFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_BAUD_FASTPATH");
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
  static bool traceTailWrapperFastPath = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_TAIL_WRAPPER_FASTPATH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto tryResumeGenerateBaudClkFrameFastPath =
      [&](CallStackFrame &frame,
          size_t oldFrameCount) -> FastResumeAction {
    if (frame.isLLVM() || !frame.funcOp || !frame.resumeBlock)
      return FastResumeAction::NotHandled;
    if (!frame.funcOp.getName().ends_with("::GenerateBaudClk"))
      return FastResumeAction::NotHandled;
    if (frame.resumeOp == frame.resumeBlock->end())
      return FastResumeAction::NotHandled;

    auto baudCallOp = dyn_cast<mlir::func::CallOp>(*frame.resumeOp);
    if (!baudCallOp || !baudCallOp.getCallee().ends_with("::BaudClkGenerator"))
      return FastResumeAction::NotHandled;

    auto nextOp = std::next(frame.resumeOp);
    if (nextOp == frame.resumeBlock->end() ||
        !isa<mlir::func::ReturnOp>(*nextOp))
      return FastResumeAction::NotHandled;

    auto *symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(
        baudCallOp.getOperation(), baudCallOp.getCalleeAttr());
    auto baudFuncOp = dyn_cast_or_null<mlir::func::FuncOp>(symbolOp);
    if (!baudFuncOp || baudFuncOp.isExternal())
      return FastResumeAction::NotHandled;

    llvm::SmallVector<InterpretedValue, 4> args;
    args.reserve(baudCallOp.getNumOperands());
    for (Value operand : baudCallOp.getOperands())
      args.push_back(getValue(procId, operand));

    if (!handleBaudClkGeneratorFastPath(procId, baudCallOp, baudFuncOp, args,
                                        baudCallOp.getCallee()))
      return FastResumeAction::NotHandled;
    if (!state.waiting)
      return FastResumeAction::NotHandled;

    if (traceBaudFastPath) {
      static unsigned generateResumeFastPathHits = 0;
      if (generateResumeFastPathHits < 50) {
        ++generateResumeFastPathHits;
        llvm::errs() << "[BAUD-GEN-FP] resume-hit proc=" << procId
                     << " callee=" << frame.funcOp.getName() << "\n";
      }
    }

    if (frame.callOp &&
        (nextOp != frame.resumeBlock->end() ||
         frame.resumeBlock != &frame.funcOp.getBody().front())) {
      CallStackFrame resumedFrame(frame.funcOp, frame.resumeBlock, nextOp,
                                  frame.callOp);
      resumedFrame.args.assign(frame.args.begin(), frame.args.end());
      state.callStack.push_back(std::move(resumedFrame));
    }

    if (oldFrameCount > 0 && state.callStack.size() > oldFrameCount) {
      std::rotate(state.callStack.begin(),
                  state.callStack.begin() + oldFrameCount,
                  state.callStack.end());
    }

    if (state.pendingDelayFs > 0) {
      SimTime currentTime = scheduler.getCurrentTime();
      SimTime targetTime = currentTime.advanceTime(state.pendingDelayFs);
      state.pendingDelayFs = 0;
      scheduler.getEventScheduler().schedule(
          targetTime, SchedulingRegion::Active,
          Event([this, procId]() { resumeProcess(procId); }));
    }

    return FastResumeAction::Suspended;
  };
  auto collapseTailWrapperFrame = [&](CallStackFrame &frame,
                                      size_t &oldFrameCount) {
    if (frame.isLLVM() || !frame.funcOp || state.callStack.empty())
      return;

    CallStackFrame &outerFrame = state.callStack.front();
    if (outerFrame.isLLVM() || !outerFrame.funcOp || !outerFrame.resumeBlock)
      return;
    if (outerFrame.resumeOp == outerFrame.resumeBlock->end())
      return;

    auto returnOp = dyn_cast<mlir::func::ReturnOp>(*outerFrame.resumeOp);
    if (!returnOp || returnOp.getNumOperands() != 0)
      return;
    if (outerFrame.resumeOp == outerFrame.resumeBlock->begin())
      return;

    auto callInWrapperIt = std::prev(outerFrame.resumeOp);
    auto wrapperCallOp = dyn_cast<mlir::func::CallOp>(*callInWrapperIt);
    if (!wrapperCallOp || wrapperCallOp.getNumResults() != 0)
      return;
    if (wrapperCallOp.getCallee() != frame.funcOp.getName())
      return;

    // Tail call-through wrappers can be dropped once execution has suspended in
    // the callee. Keep resume work pinned on the hot inner frame.
    llvm::StringRef outerFrameNameRef = outerFrame.funcOp.getName();
    llvm::StringRef calleeNameRef = wrapperCallOp.getCallee();
    std::string outerFrameName = outerFrameNameRef.str();
    std::string calleeName = calleeNameRef.str();
    state.callStack.erase(state.callStack.begin());
    --oldFrameCount;

    bool emitMonitorDeserializerTrace =
        traceMonitorDeserializerFastPath &&
        outerFrameNameRef.ends_with("::StartMonitoring") &&
        calleeNameRef.ends_with("::Deserializer");
    bool emitDriveSampleTrace = traceDriveSampleFastPath &&
                                outerFrameNameRef.ends_with("::DriveToBfm") &&
                                calleeNameRef.ends_with("::SampleData");

    if (emitMonitorDeserializerTrace || emitDriveSampleTrace ||
        traceTailWrapperFastPath) {
      static unsigned tailWrapperResumeFastPathHits = 0;
      if (tailWrapperResumeFastPathHits < 100) {
        ++tailWrapperResumeFastPathHits;
        if (emitMonitorDeserializerTrace)
          llvm::errs() << "[MON-DESER-FP]";
        else if (emitDriveSampleTrace)
          llvm::errs() << "[DRV-SAMPLE-FP]";
        else
          llvm::errs() << "[TAIL-WRAP-FP]";
        llvm::errs() << " resume-hit proc=" << procId
                     << " wrapper=" << outerFrameName
                     << " callee=" << calleeName << "\n";
      }
    }
  };

  while (!state.callStack.empty()) {
    // Use copy-out instead of move-out for frame extraction. Some compile-mode
    // workloads hit unstable behavior in SmallVector move-assignment for
    // CallStackFrame payloads during nested resume/rotate sequences.
    CallStackFrame frame = state.callStack.front();
    state.callStack.erase(state.callStack.begin());

    // Remaining old outer frames before resuming this frame.
    size_t oldFrameCount = state.callStack.size();

    // Corrupt or synthetic frames should fail gracefully through deopt bridge
    // instead of dereferencing null function metadata.
    if ((!frame.isLLVM() && !frame.funcOp) || !frame.resumeBlock) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Invalid saved call stack frame, cannot resume\n");
      return CallStackResumeResult::Failed;
    }

    llvm::StringRef frameName =
        frame.isLLVM() ? frame.llvmFuncOp.getName() : frame.funcOp.getName();
    if (traceI3CCallStack &&
        (frameName.contains("i3c_target_monitor_bfm::") ||
         frameName.contains("i3c_target_driver_bfm::") ||
         frameName.contains("i3c_controller_monitor_bfm::") ||
         frameName.contains("i3c_controller_driver_bfm::"))) {
      llvm::errs() << "[I3C-CS-RESUME] proc=" << procId
                   << " func=" << frameName << " old_outer=" << oldFrameCount
                   << " stack_remaining=" << state.callStack.size() << "\n";
    }
    LLVM_DEBUG(llvm::dbgs()
               << "    Resuming " << (frame.isLLVM() ? "LLVM " : "")
               << "function '" << frameName << "' (remaining old frames: "
               << oldFrameCount << ")\n");

    collapseTailWrapperFrame(frame, oldFrameCount);

    FastResumeAction fastResumeAction =
        tryResumeGenerateBaudClkFrameFastPath(frame, oldFrameCount);
    if (fastResumeAction == FastResumeAction::Failed)
      return CallStackResumeResult::Failed;
    if (fastResumeAction == FastResumeAction::Suspended)
      return CallStackResumeResult::Suspended;
    if (fastResumeAction == FastResumeAction::Continue)
      continue;

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
      if (traceI3CCallStack &&
          (frameName.contains("i3c_target_monitor_bfm::") ||
           frameName.contains("i3c_target_driver_bfm::") ||
           frameName.contains("i3c_controller_monitor_bfm::") ||
           frameName.contains("i3c_controller_driver_bfm::"))) {
        llvm::errs() << "[I3C-CS-SUSPEND] proc=" << procId
                     << " func=" << frameName
                     << " stack_size=" << state.callStack.size()
                     << " old_outer=" << oldFrameCount << "\n";
      }
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
  state.callStackOutermostCallOp = nullptr;

  LLVM_DEBUG(llvm::dbgs()
             << "  Call stack frames exhausted, continuing process\n");
  return CallStackResumeResult::Completed;
}
