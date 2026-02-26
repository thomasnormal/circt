// RUN: env CIRCT_SIM_TRACE_WAIT_CONDITION=1 circt-sim %s --max-time=4000000000 2>&1 | FileCheck %s
//
// When wait(condition) is lowered from uvm_phase::wait_for_state and depends on
// a memory load, the interpreter arms a memory-event waiter plus a timed poll.
// The timed poll must still re-enter __moore_wait_condition even when the memory
// location is unchanged; otherwise the process strands with waiting=0/callStack>0.
//
// CHECK: [WAITCOND] proc=
// CHECK-SAME: func=uvm_pkg::uvm_phase::wait_for_state
// CHECK-SAME: condition=false
// CHECK: [WAITCOND] proc=
// CHECK-SAME: func=uvm_pkg::uvm_phase::wait_for_state
// CHECK-SAME: condition=false

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_wait_condition(i32)

  func.func @"uvm_pkg::uvm_phase::wait_for_state"(%phaseState: !llvm.ptr) {
    %c1 = hw.constant 1 : i32
    %state = llvm.load %phaseState : !llvm.ptr -> i32
    %cond = comb.icmp eq %state, %c1 : i32
    %condI32 = llvm.zext %cond : i1 to i32
    llvm.call @__moore_wait_condition(%condI32) : (i32) -> ()
    return
  }

  hw.module @top() {
    %c0_i32 = hw.constant 0 : i32
    %c4_i64 = hw.constant 4 : i64
    %c3500000000_i64 = hw.constant 3500000000 : i64

    // Wait-for-state process. The pointed state remains 0 forever, so the
    // condition stays false and must be re-polled by timed callbacks.
    llhd.process {
      %ptr = llvm.call @malloc(%c4_i64) : (i64) -> !llvm.ptr
      llvm.store %c0_i32, %ptr : i32, !llvm.ptr
      func.call @"uvm_pkg::uvm_phase::wait_for_state"(%ptr) : (!llvm.ptr) -> ()
      llhd.halt
    }

    // End the simulation after enough real time for multiple polls.
    llhd.process {
      %delay = llhd.int_to_time %c3500000000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
