//===- LLHDProcessInterpreterNativeThunkPolicy.cpp - Native thunk policy --===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstdlib>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

static bool isSafeSingleBlockTerminatingPreludeOp(Operation *op);
static bool isBranchTerminatorInRegion(Operation &terminator,
                                       Region &bodyRegion);

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
  // Probe reads are intentionally allowed for event sensitivity setup.
  if (isa<llhd::ProbeOp>(op))
    return true;
  return mlir::isMemoryEffectFree(op);
}

static bool isConfigDbSetWrapperName(StringRef calleeName) {
  if (!calleeName.starts_with("set_") || calleeName.size() <= 4)
    return false;
  for (char c : calleeName.drop_front(4))
    if (!std::isdigit(static_cast<unsigned char>(c)))
      return false;
  return true;
}

static bool isConfigDbSetWrapperCallPrelude(mlir::func::CallOp callOp) {
  if (callOp.getNumResults() != 0 || callOp.getNumOperands() != 4)
    return false;
  if (!isa<LLVM::LLVMPointerType>(callOp.getOperand(0).getType()) ||
      !isa<LLVM::LLVMPointerType>(callOp.getOperand(3).getType()))
    return false;
  return isConfigDbSetWrapperName(callOp.getCallee());
}

static bool isInterceptedNonSuspendingFuncCallPrelude(mlir::func::CallOp callOp) {
  StringRef calleeName = callOp.getCallee();
  if (calleeName == "m_execute_scheduled_forks" ||
      calleeName == "m_process_guard" ||
      calleeName == "uvm_pkg::run_test" ||
      calleeName == "uvm_pkg::uvm_create_random_seed" ||
      calleeName == "uvm_pkg::uvm_get_report_object" ||
      calleeName == "uvm_pkg::uvm_root::find_all" ||
      calleeName.ends_with("::get_automatic_phase_objection") ||
      calleeName.ends_with("::m_safe_raise_starting_phase") ||
      calleeName.ends_with("::m_safe_drop_starting_phase") ||
      calleeName.ends_with("::m_killed") ||
      calleeName.ends_with("::set_report_id_verbosity") ||
      calleeName.ends_with("::get_report_action") ||
      calleeName.ends_with("::get_report_verbosity_level") ||
      calleeName.ends_with("::set_report_verbosity_level") ||
      calleeName.ends_with("::uvm_get_report_object") ||
      calleeName.ends_with("::m_process_guard") ||
      calleeName == "get_global_hopper" ||
      calleeName.ends_with("::get_global_hopper"))
    return true;

  if (isConfigDbSetWrapperCallPrelude(callOp))
    return true;

  return false;
}

static bool isMooreWaitConditionCall(LLVM::CallOp callOp) {
  auto callee = callOp.getCallee();
  return callee && *callee == "__moore_wait_condition";
}

static bool isMooreDelayCall(LLVM::CallOp callOp) {
  auto callee = callOp.getCallee();
  return callee && *callee == "__moore_delay";
}

static bool isMooreProcessAwaitCall(LLVM::CallOp callOp) {
  auto callee = callOp.getCallee();
  return callee && *callee == "__moore_process_await";
}

static bool isPotentialResumableMultiblockSuspendOp(Operation *op) {
  if (!op)
    return false;
  if (isa<sim::SimForkOp, mlir::func::CallIndirectOp>(op))
    return true;
  if (auto callOp = dyn_cast<LLVM::CallOp>(op))
    return isMooreWaitConditionCall(callOp) || isMooreDelayCall(callOp) ||
           isMooreProcessAwaitCall(callOp);
  return false;
}

static bool isSafeStructuredControlPreludeRegion(Region &region) {
  if (region.empty())
    return false;

  bool sawStructuredTerminator = false;
  for (Block &block : region) {
    if (block.empty())
      return false;
    Operation &terminator = block.back();
    bool isStructuredTerminator =
        isa<mlir::scf::YieldOp, mlir::scf::ConditionOp>(terminator);
    bool isBranch = isBranchTerminatorInRegion(terminator, region);
    if (!isStructuredTerminator && !isBranch)
      return false;
    sawStructuredTerminator |= isStructuredTerminator;
    for (auto it = block.begin(), e = std::prev(block.end()); it != e; ++it) {
      if (!isSafeSingleBlockTerminatingPreludeOp(&*it))
        return false;
    }
  }
  return sawStructuredTerminator;
}

static bool isSafeStructuredControlPreludeOp(Operation *op) {
  if (!op)
    return false;
  if (auto forOp = dyn_cast<mlir::scf::ForOp>(op))
    return isSafeStructuredControlPreludeRegion(forOp.getRegion());
  if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
    if (!isSafeStructuredControlPreludeRegion(ifOp.getThenRegion()))
      return false;
    if (!ifOp.getElseRegion().empty() &&
        !isSafeStructuredControlPreludeRegion(ifOp.getElseRegion()))
      return false;
    return true;
  }
  if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op))
    return isSafeStructuredControlPreludeRegion(whileOp.getBefore()) &&
           isSafeStructuredControlPreludeRegion(whileOp.getAfter());
  return false;
}

static bool isSafeForkPreludeRegion(Region &region) {
  if (region.empty())
    return false;
  bool sawTerminator = false;
  for (Block &block : region) {
    if (block.empty())
      return false;
    Operation &terminator = block.back();
    bool isTerminal = isa<sim::SimForkTerminatorOp>(terminator);
    bool isBranch = isBranchTerminatorInRegion(terminator, region);
    if (!isTerminal && !isBranch)
      return false;
    sawTerminator |= isTerminal;
  }
  return sawTerminator;
}

static bool isSafeForkPreludeOp(sim::SimForkOp forkOp) {
  StringRef joinType = forkOp.getJoinType();
  if (!joinType.equals_insensitive("join") &&
      !joinType.equals_insensitive("join_any") &&
      !joinType.equals_insensitive("join_none") && !joinType.empty())
    return false;
  for (Region &branch : forkOp.getBranches()) {
    if (!isSafeForkPreludeRegion(branch))
      return false;
  }
  return true;
}

static bool isSafeSingleBlockTerminatingPreludeOp(Operation *op) {
  if (!op)
    return false;
  if (isa<sim::PrintFormattedProcOp>(op))
    return true;
  if (isa<llhd::ProbeOp>(op))
    return true;
  if (isa<sim::SimDisableForkOp>(op))
    return true;
  if (auto forkOp = dyn_cast<sim::SimForkOp>(op))
    return isSafeForkPreludeOp(forkOp);
  if (isSafeStructuredControlPreludeOp(op))
    return true;
  if (auto callOp = dyn_cast<mlir::func::CallOp>(op))
    return isInterceptedNonSuspendingFuncCallPrelude(callOp);
  if (isa<mlir::func::CallIndirectOp>(op))
    return true;
  if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
    auto callee = callOp.getCallee();
    if (!callee)
      return false;
    llvm::StringRef calleeName = *callee;
    if (calleeName == "__moore_process_self" ||
        calleeName == "__moore_process_await" ||
        calleeName == "__moore_process_srandom" ||
        calleeName == "__moore_is_rand_enabled" ||
        calleeName == "__moore_int_to_string" ||
        calleeName == "__moore_string_itoa" ||
        calleeName == "__moore_string_concat" ||
        calleeName == "__moore_randomize_basic" ||
        calleeName == "__moore_randomize_with_range" ||
        calleeName == "__moore_randomize_with_ranges" ||
        calleeName == "__moore_randomize_with_dist" ||
        calleeName == "__moore_packed_string_to_string" ||
        calleeName == "__moore_string_cmp" ||
        calleeName == "__moore_assoc_get_ref" ||
        calleeName == "__moore_uvm_report_info" ||
        calleeName == "__moore_queue_push_front" ||
        calleeName == "__moore_queue_pop_front_ptr")
      return true;
    if (calleeName == "__moore_wait_condition") {
      Region *parentRegion = callOp->getBlock() ? callOp->getBlock()->getParent()
                                                : nullptr;
      return parentRegion && parentRegion->hasOneBlock();
    }
    return false;
  }
  if (isa<LLVM::StoreOp>(op))
    return true;
  if (op->getNumResults() == 0)
    return false;
  if (isa<LLVM::AllocaOp, LLVM::GEPOp, LLVM::LoadOp>(op))
    return true;
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<llhd::WaitOp, llhd::YieldOp, llhd::HaltOp, sim::SimForkOp,
          sim::SimJoinOp, sim::SimJoinAnyOp, sim::SimWaitForkOp,
          sim::SimDisableForkOp, sim::SimForkTerminatorOp, mlir::func::CallOp,
          LLVM::CallOp, mlir::cf::BranchOp, mlir::cf::CondBranchOp>(op))
    return false;
  return mlir::isMemoryEffectFree(op);
}

static bool isSafeResumableMultiblockWaitPreludeOp(Operation *op) {
  if (!op)
    return false;
  if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
    // wait_condition, __moore_delay, and process::await can suspend/resume;
    // resumable multiblock thunks handle these state-machine paths explicitly.
    if (isMooreWaitConditionCall(callOp) || isMooreDelayCall(callOp) ||
        isMooreProcessAwaitCall(callOp))
      return true;
  }
  // Multiblock wait loops commonly drive signals between waits.
  if (isa<llhd::DriveOp>(op))
    return true;
  return isSafeSingleBlockTerminatingPreludeOp(op);
}

static bool isBranchTerminatorInRegion(Operation &terminator, Region &bodyRegion) {
  if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
    if (!branchOp.getDest() || branchOp.getDest()->getParent() != &bodyRegion)
      return false;
    return branchOp.getDestOperands().size() ==
           branchOp.getDest()->getNumArguments();
  }
  if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (!condBrOp.getTrueDest() ||
        condBrOp.getTrueDest()->getParent() != &bodyRegion ||
        !condBrOp.getFalseDest() ||
        condBrOp.getFalseDest()->getParent() != &bodyRegion)
      return false;
    return condBrOp.getTrueDestOperands().size() ==
               condBrOp.getTrueDest()->getNumArguments() &&
           condBrOp.getFalseDestOperands().size() ==
               condBrOp.getFalseDest()->getNumArguments();
  }
  return false;
}

static Value stripCallIndirectCalleeCasts(Value value) {
  while (auto castOp =
             value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() != 1)
      break;
    value = castOp.getInputs().front();
  }
  return value;
}

static std::string
getStaticCallIndirectTargetSymbol(mlir::func::CallIndirectOp callOp) {
  Value callee = stripCallIndirectCalleeCasts(callOp.getCallee());
  if (auto constOp = callee.getDefiningOp<mlir::func::ConstantOp>())
    return constOp.getValue().str();
  return {};
}

static std::string formatUnsupportedProcessOpDetail(Operation &op) {
  if (auto llvmCall = dyn_cast<LLVM::CallOp>(op)) {
    if (auto callee = llvmCall.getCallee())
      return (Twine("first_op:llvm.call:") + *callee).str();
  }
  if (auto funcCall = dyn_cast<mlir::func::CallOp>(op))
    return (Twine("first_op:func.call:") + funcCall.getCallee()).str();
  if (auto callIndirect = dyn_cast<mlir::func::CallIndirectOp>(op)) {
    std::string symbol = getStaticCallIndirectTargetSymbol(callIndirect);
    if (!symbol.empty())
      return (Twine("first_op:func.call_indirect:") + symbol).str();
    return "first_op:func.call_indirect";
  }
  return (Twine("first_op:") + op.getName().getStringRef()).str();
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

LLHDProcessInterpreter::ProcessThunkInstallResult
LLHDProcessInterpreter::tryInstallProcessThunk(ProcessId procId,
                                               ProcessExecutionState &state,
                                               std::string *deoptDetail) {
  if (!jitCompileManager)
    return ProcessThunkInstallResult::MissingThunk;

  auto traceUnsupported = []() {
    const char *env =
        std::getenv("CIRCT_SIM_TRACE_JIT_UNSUPPORTED_SHAPES");
    return env && env[0] != '\0' && env[0] != '0';
  }();

  auto dumpUnsupportedShape = [&](StringRef detail) {
    if (!traceUnsupported || !state.getProcessOp())
      return;
    auto processOp = state.getProcessOp();
    Region *bodyRegion = resolveNativeThunkProcessRegion(state);
    Region &body = bodyRegion ? *bodyRegion : processOp.getBody();
    auto *proc = scheduler.getProcess(procId);
    StringRef procName = proc ? proc->getName() : "<unknown>";
    llvm::errs() << "[JIT-UNSUPPORTED] proc=" << procId << " name=" << procName
                 << " detail=" << detail
                 << " blocks=" << static_cast<unsigned>(body.getBlocks().size())
                 << "\n";
    for (auto [blockIdx, block] : llvm::enumerate(body)) {
      if (block.empty()) {
        llvm::errs() << "  [block " << blockIdx << "] <empty>\n";
        continue;
      }
      Operation &terminator = block.back();
      llvm::errs() << "  [block " << blockIdx
                   << "] terminator=" << terminator.getName().getStringRef();
      if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
        llvm::errs() << " wait(delay=" << (waitOp.getDelay() ? "1" : "0")
                     << ", observed=" << waitOp.getObserved().size()
                     << ", dest="
                     << (waitOp.getDest() ? "set" : "null") << ")";
      }
      llvm::errs() << "\n";
      for (Operation &op : block) {
        llvm::errs() << "    op=" << op.getName().getStringRef();
        if (auto callOp = dyn_cast<mlir::func::CallOp>(op))
          llvm::errs() << " callee=" << callOp.getCallee();
        else if (auto llvmCall = dyn_cast<LLVM::CallOp>(op)) {
          if (auto callee = llvmCall.getCallee())
            llvm::errs() << " callee=" << *callee;
        } else if (auto forkOp = dyn_cast<sim::SimForkOp>(op)) {
          llvm::errs() << " join=" << forkOp.getJoinType();
          for (auto [branchIdx, branch] : llvm::enumerate(forkOp.getBranches())) {
            llvm::errs() << "\n      [fork-branch " << branchIdx
                         << "] blocks="
                         << static_cast<unsigned>(branch.getBlocks().size());
            for (auto [nestedBlockIdx, nestedBlock] : llvm::enumerate(branch)) {
              if (nestedBlock.empty()) {
                llvm::errs() << "\n        [block " << nestedBlockIdx
                             << "] <empty>";
                continue;
              }
              llvm::errs() << "\n        [block " << nestedBlockIdx
                           << "] terminator="
                           << nestedBlock.back().getName().getStringRef();
              for (Operation &nestedOp : nestedBlock) {
                llvm::errs() << "\n          op="
                             << nestedOp.getName().getStringRef();
                if (auto nestedFuncCall = dyn_cast<mlir::func::CallOp>(nestedOp))
                  llvm::errs() << " callee=" << nestedFuncCall.getCallee();
                else if (auto nestedLLVMCall = dyn_cast<LLVM::CallOp>(nestedOp))
                  if (auto nestedCallee = nestedLLVMCall.getCallee())
                    llvm::errs() << " callee=" << *nestedCallee;
              }
            }
          }
        } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
          auto dumpIfRegion = [&](StringRef tag, Region &region) {
            llvm::errs() << "\n      [scf.if " << tag
                         << "] blocks="
                         << static_cast<unsigned>(region.getBlocks().size());
            for (auto [nestedBlockIdx, nestedBlock] : llvm::enumerate(region)) {
              if (nestedBlock.empty()) {
                llvm::errs() << "\n        [block " << nestedBlockIdx
                             << "] <empty>";
                continue;
              }
              llvm::errs() << "\n        [block " << nestedBlockIdx
                           << "] terminator="
                           << nestedBlock.back().getName().getStringRef();
              for (Operation &nestedOp : nestedBlock) {
                llvm::errs() << "\n          op="
                             << nestedOp.getName().getStringRef();
                if (auto nestedFuncCall = dyn_cast<mlir::func::CallOp>(nestedOp))
                  llvm::errs() << " callee=" << nestedFuncCall.getCallee();
                else if (auto nestedLLVMCall = dyn_cast<LLVM::CallOp>(nestedOp))
                  if (auto nestedCallee = nestedLLVMCall.getCallee())
                    llvm::errs() << " callee=" << *nestedCallee;
              }
            }
          };
          dumpIfRegion("then", ifOp.getThenRegion());
          if (!ifOp.getElseRegion().empty())
            dumpIfRegion("else", ifOp.getElseRegion());
        }
        llvm::errs() << "\n";
      }
    }
  };

  auto compileAttempt =
      jitCompileManager->classifyProcessCompileAttempt(procId);
  if (compileAttempt != JITCompileManager::CompileAttemptDecision::Proceed) {
    if (deoptDetail)
      *deoptDetail =
          JITCompileManager::getCompileAttemptDecisionName(compileAttempt)
              .str();
    return ProcessThunkInstallResult::MissingThunk;
  }

  PeriodicToggleClockThunkSpec periodicSpec;
  bool isPeriodicToggleClock =
      tryBuildPeriodicToggleClockThunkSpec(state, periodicSpec);
  bool isTrivialCandidate =
      !isPeriodicToggleClock && isTrivialNativeThunkCandidate(state);
  if (!isPeriodicToggleClock && !isTrivialCandidate) {
    std::string detail = getUnsupportedThunkDeoptDetail(state);
    if (deoptDetail)
      *deoptDetail = detail;
    dumpUnsupportedShape(detail);
    return ProcessThunkInstallResult::UnsupportedOperation;
  }

  bool installed =
      jitCompileManager->installProcessThunk(procId, [this, procId](
                                                         ProcessThunkExecutionState
                                                             &thunkState) {
        executeTrivialNativeThunk(procId, thunkState);
      });
  if (!installed) {
    if (deoptDetail)
      *deoptDetail = "install_failed";
    return ProcessThunkInstallResult::MissingThunk;
  }

  if (isPeriodicToggleClock)
    periodicToggleClockThunkSpecs[procId] = std::move(periodicSpec);
  else
    periodicToggleClockThunkSpecs.erase(procId);

  jitCompileManager->noteCompile();
  return ProcessThunkInstallResult::Installed;
}

std::string LLHDProcessInterpreter::getUnsupportedThunkDeoptDetail(
    const ProcessExecutionState &state) const {
  if (auto combOp = state.getCombinationalOp()) {
    Region &body = combOp.getBody();
    if (body.empty())
      return "combinational_empty";
    bool sawYield = false;
    for (Block &block : body) {
      if (block.empty())
        return "combinational_empty_block";
      Operation &terminator = block.back();
      if (!isa<llhd::YieldOp, mlir::cf::BranchOp, mlir::cf::CondBranchOp>(
              terminator)) {
        return (Twine("combinational_unsupported_terminator:") +
                terminator.getName().getStringRef())
            .str();
      }
      for (Operation &op : block) {
        if (isa<llhd::WaitOp, llhd::HaltOp>(op))
          return (Twine("combinational_unsupported:") +
                  op.getName().getStringRef())
              .str();
        if (isa<llhd::YieldOp>(op))
          sawYield = true;
      }
    }
    if (!sawYield)
      return "combinational_no_yield";
    return (Twine("combinational_unsupported:first_op:") +
            body.front().front().getName().getStringRef())
        .str();
  }

  if (auto processOp = state.getProcessOp()) {
    const Region *bodyRegion = resolveNativeThunkProcessRegion(state);
    Region &body =
        bodyRegion ? *const_cast<Region *>(bodyRegion) : processOp.getBody();
    if (!body.empty()) {
      if (body.hasOneBlock()) {
        Block &singleBlock = body.front();
        if (!singleBlock.empty() &&
            isa<llhd::HaltOp, sim::SimForkTerminatorOp>(singleBlock.back())) {
          for (auto it = singleBlock.begin(), e = std::prev(singleBlock.end());
               it != e; ++it) {
            if (!isSafeSingleBlockTerminatingPreludeOp(&*it))
              return formatUnsupportedProcessOpDetail(*it);
          }
        }
      } else {
        bool sawTerminal = false;
        bool sawSuspend = false;
        for (Block &block : body) {
          if (block.empty())
            return "multiblock_empty_block";
          Operation &terminator = block.back();
          if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
            if (!waitOp.getDest() || waitOp.getDest()->getParent() != &body ||
                waitOp.getDestOperands().size() !=
                    waitOp.getDest()->getNumArguments()) {
              return (Twine("multiblock_unsupported_terminator:") +
                      terminator.getName().getStringRef())
                  .str();
            }
            sawSuspend = true;
          } else if (isa<llhd::HaltOp, sim::SimForkTerminatorOp>(terminator)) {
            sawTerminal = true;
          } else if (!isBranchTerminatorInRegion(terminator, body)) {
            return (Twine("multiblock_unsupported_terminator:") +
                    terminator.getName().getStringRef())
                .str();
          }
          for (auto it = block.begin(), e = std::prev(block.end()); it != e;
               ++it) {
            if (!isSafeResumableMultiblockWaitPreludeOp(&*it))
              return formatUnsupportedProcessOpDetail(*it);
            sawSuspend |= isPotentialResumableMultiblockSuspendOp(&*it);
          }
        }
        if (!sawTerminal && !sawSuspend)
          return "multiblock_no_terminal";
      }
      Block &entry = body.front();
      if (!entry.empty()) {
        auto waitIt = std::prev(entry.end());
        if (auto waitOp = dyn_cast<llhd::WaitOp>(*waitIt)) {
          (void)waitOp;
          for (auto it = entry.begin(); it != waitIt; ++it) {
            Operation *op = &*it;
            if (!isPureResumableWaitPreludeOp(op))
              return (Twine("prewait_impure:") +
                      op->getName().getStringRef())
                  .str();
          }
        }
      }
    }
    for (Block &block : body) {
      for (Operation &op : block)
        return formatUnsupportedProcessOpDetail(op);
    }
    return "process_empty";
  }

  if (auto initialOp = state.getInitialOp()) {
    Block *body = initialOp.getBodyBlock();
    if (!body || body->empty())
      return "initial_empty";
    return (Twine("initial_first_op:") + body->front().getName().getStringRef())
        .str();
  }

  return "unknown_shape";
}

bool LLHDProcessInterpreter::snapshotJITDeoptState(
    ProcessId procId, JITDeoptStateSnapshot &snapshot) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return false;
  ProcessExecutionState &state = it->second;
  snapshot.currentBlock = state.currentBlock;
  snapshot.currentOp = state.currentOp;
  snapshot.halted = state.halted;
  snapshot.waiting = state.waiting;
  snapshot.destBlock = state.destBlock;
  snapshot.resumeAtCurrentOp = state.resumeAtCurrentOp;
  snapshot.destOperands = state.destOperands;
  snapshot.callStack = state.callStack;
  snapshot.jitThunkResumeToken = state.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::restoreJITDeoptState(
    ProcessId procId, const JITDeoptStateSnapshot &snapshot) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return false;
  ProcessExecutionState &state = it->second;
  state.currentBlock = snapshot.currentBlock;
  state.currentOp = snapshot.currentOp;
  state.halted = snapshot.halted;
  state.waiting = snapshot.waiting;
  state.destBlock = snapshot.destBlock;
  state.resumeAtCurrentOp = snapshot.resumeAtCurrentOp;
  state.destOperands = snapshot.destOperands;
  state.callStack = snapshot.callStack;
  state.jitThunkResumeToken = snapshot.jitThunkResumeToken;
  return true;
}

bool LLHDProcessInterpreter::isTrivialNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  if (isCombinationalNativeThunkCandidate(state))
    return true;

  if (isSingleBlockTerminatingNativeThunkCandidate(state))
    return true;

  if (isMultiBlockTerminatingNativeThunkCandidate(state))
    return true;

  if (isResumableWaitSelfLoopNativeThunkCandidate(state))
    return true;

  if (isResumableMultiblockWaitNativeThunkCandidate(state))
    return true;

  if (isResumableWaitThenHaltNativeThunkCandidate(state))
    return true;

  if (state.getProcessOp()) {
    auto processOp = state.getProcessOp();
    if (!processOp.getBody().hasOneBlock())
      return false;
    Block &body = processOp.getBody().front();
    if (body.empty())
      return false;
    auto it = body.begin();
    if (auto haltOp = dyn_cast<llhd::HaltOp>(*it)) {
      (void)haltOp;
      return std::next(it) == body.end();
    }
    if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*it)) {
      (void)printOp;
      ++it;
      if (it == body.end())
        return false;
      if (!dyn_cast<llhd::HaltOp>(*it))
        return false;
      return std::next(it) == body.end();
    }
    return false;
  }

  if (state.getInitialOp()) {
    auto initialOp = state.getInitialOp();
    Block *body = initialOp.getBodyBlock();
    if (!body || body->empty())
      return false;
    Operation &lastOp = body->back();
    auto yieldOp = dyn_cast<seq::YieldOp>(lastOp);
    if (!yieldOp || !yieldOp.getOperands().empty())
      return false;
    if (llvm::hasSingleElement(*body))
      return true;

    auto rit = body->rbegin();
    ++rit;
    auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*rit);
    if (!printOp)
      return false;

    for (auto it = body->begin(), e = std::prev(std::prev(body->end())); it != e;
         ++it) {
      Operation *op = &*it;
      if (op->getName().getStringRef().starts_with("sim.fmt."))
        continue;
      if (isa<hw::ConstantOp, arith::ConstantOp, LLVM::ConstantOp>(op))
        continue;
      return false;
    }
    return true;
  }

  return false;
}

bool LLHDProcessInterpreter::isSingleBlockTerminatingNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  Region *bodyRegion =
      resolveNativeThunkProcessRegion(const_cast<ProcessExecutionState &>(state));
  if (!bodyRegion || !bodyRegion->hasOneBlock())
    return false;

  Block &body = bodyRegion->front();
  if (body.empty())
    return false;

  Operation &terminator = body.back();
  if (!isa<llhd::HaltOp, sim::SimForkTerminatorOp>(terminator))
    return false;

  for (auto it = body.begin(), e = std::prev(body.end()); it != e; ++it) {
    Operation *op = &*it;
    if (!isSafeSingleBlockTerminatingPreludeOp(op))
      return false;
  }
  return true;
}

bool LLHDProcessInterpreter::isMultiBlockTerminatingNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  Region *bodyRegion =
      resolveNativeThunkProcessRegion(const_cast<ProcessExecutionState &>(state));
  if (!bodyRegion || bodyRegion->empty() || bodyRegion->hasOneBlock())
    return false;

  bool sawTerminal = false;
  for (Block &block : *bodyRegion) {
    if (block.empty())
      return false;

    Operation &terminator = block.back();
    bool isTerminal = isa<llhd::HaltOp, sim::SimForkTerminatorOp>(terminator);
    bool isBranch = isBranchTerminatorInRegion(terminator, *bodyRegion);
    if (!isTerminal && !isBranch)
      return false;
    sawTerminal |= isTerminal;

    for (auto it = block.begin(), e = std::prev(block.end()); it != e; ++it) {
      if (!isSafeSingleBlockTerminatingPreludeOp(&*it))
        return false;
    }
  }

  return sawTerminal;
}

bool LLHDProcessInterpreter::isCombinationalNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  auto combOp = state.getCombinationalOp();
  if (!combOp)
    return false;
  Region &body = combOp.getBody();
  if (body.empty())
    return false;
  bool sawYield = false;
  for (Block &block : body) {
    if (block.empty())
      return false;
    Operation &terminator = block.back();
    if (!isa<llhd::YieldOp, mlir::cf::BranchOp, mlir::cf::CondBranchOp>(
            terminator))
      return false;
    for (Operation &op : block) {
      if (isa<llhd::WaitOp, llhd::HaltOp>(op))
        return false;
      if (isa<llhd::YieldOp>(op))
        sawYield = true;
    }
  }
  return sawYield;
}

bool LLHDProcessInterpreter::isResumableWaitSelfLoopNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  auto processOp = state.getProcessOp();
  if (!processOp)
    return false;

  Region &bodyRegion = processOp.getBody();
  if (bodyRegion.empty() || llvm::hasNItemsOrMore(bodyRegion, 3))
    return false;

  Block *loopBlock = nullptr;
  if (bodyRegion.hasOneBlock()) {
    loopBlock = &bodyRegion.front();
  } else {
    Block &entry = bodyRegion.front();
    if (entry.empty())
      return false;
    auto entryBranch = dyn_cast<mlir::cf::BranchOp>(entry.back());
    if (!entryBranch || !entryBranch.getDestOperands().empty())
      return false;
    for (auto it = entry.begin(), e = std::prev(entry.end()); it != e; ++it) {
      if (!isSafeSingleBlockTerminatingPreludeOp(&*it))
        return false;
    }
    loopBlock = entryBranch.getDest();
  }

  if (!loopBlock || loopBlock->empty())
    return false;
  auto waitOp = dyn_cast<llhd::WaitOp>(loopBlock->back());
  if (!waitOp || (!waitOp.getDelay() && waitOp.getObserved().empty()))
    return false;
  if (waitOp.getDest() != loopBlock)
    return false;
  if (waitOp.getDestOperands().size() != loopBlock->getNumArguments())
    return false;

  for (auto it = loopBlock->begin(), e = std::prev(loopBlock->end()); it != e;
       ++it) {
    if (!isSafeSingleBlockTerminatingPreludeOp(&*it))
      return false;
  }
  return true;
}

bool LLHDProcessInterpreter::isResumableMultiblockWaitNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  Region *bodyRegion =
      resolveNativeThunkProcessRegion(const_cast<ProcessExecutionState &>(state));
  if (!bodyRegion || bodyRegion->empty())
    return false;

  bool sawSuspendSource = false;
  for (Block &block : *bodyRegion) {
    if (block.empty())
      return false;

    Operation &terminator = block.back();
    if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
      if (!waitOp.getDest() || waitOp.getDest()->getParent() != bodyRegion)
        return false;
      if (waitOp.getDestOperands().size() != waitOp.getDest()->getNumArguments())
        return false;
      sawSuspendSource = true;
    } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
      if (!branchOp.getDest() || branchOp.getDest()->getParent() != bodyRegion)
        return false;
      if (branchOp.getDestOperands().size() != branchOp.getDest()->getNumArguments())
        return false;
    } else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      if (!condBrOp.getTrueDest() ||
          condBrOp.getTrueDest()->getParent() != bodyRegion ||
          !condBrOp.getFalseDest() ||
          condBrOp.getFalseDest()->getParent() != bodyRegion)
        return false;
      if (condBrOp.getTrueDestOperands().size() !=
              condBrOp.getTrueDest()->getNumArguments() ||
          condBrOp.getFalseDestOperands().size() !=
              condBrOp.getFalseDest()->getNumArguments())
        return false;
    } else if (!isa<llhd::HaltOp, sim::SimForkTerminatorOp>(terminator)) {
      return false;
    }

    for (auto it = block.begin(), e = std::prev(block.end()); it != e; ++it) {
      if (!isSafeResumableMultiblockWaitPreludeOp(&*it))
        return false;
      sawSuspendSource |= isPotentialResumableMultiblockSuspendOp(&*it);
    }
  }

  return sawSuspendSource;
}

bool LLHDProcessInterpreter::isResumableWaitThenHaltNativeThunkCandidate(
    const ProcessExecutionState &state) const {
  auto processOp = state.getProcessOp();
  if (!processOp)
    return false;
  Region &bodyRegion = processOp.getBody();
  if (!llvm::hasNItemsOrMore(bodyRegion, 2) || llvm::hasNItemsOrMore(bodyRegion, 3))
    return false;
  Block &entry = bodyRegion.front();
  if (entry.empty())
    return false;
  auto waitIt = std::prev(entry.end());
  auto waitOp = dyn_cast<llhd::WaitOp>(*waitIt);
  if (!waitOp || (!waitOp.getDelay() && waitOp.getObserved().empty()))
    return false;
  llvm::SmallVector<Operation *, 8> preWaitOps;
  llvm::SmallDenseSet<Value, 8> preWaitResults;
  for (auto it = entry.begin(); it != waitIt; ++it) {
    Operation *op = &*it;
    if (!isPureResumableWaitPreludeOp(op))
      return false;
    preWaitOps.push_back(op);
    for (Value result : op->getResults())
      preWaitResults.insert(result);
  }
  auto observed = waitOp.getObserved();
  if (preWaitOps.empty()) {
    if (!observed.empty())
      return false;
  } else {
    for (Value obs : observed) {
      if (!preWaitResults.contains(obs))
        return false;
    }
  }
  Block *dest = waitOp.getDest();
  if (!dest || dest == &entry || dest->empty())
    return false;
  auto opIt = dest->begin();
  if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(*opIt)) {
    (void)printOp;
    ++opIt;
  }
  if (opIt == dest->end())
    return false;
  auto haltOp = dyn_cast<llhd::HaltOp>(*opIt);
  if (!haltOp)
    return false;
  ++opIt;
  return opIt == dest->end();
}
