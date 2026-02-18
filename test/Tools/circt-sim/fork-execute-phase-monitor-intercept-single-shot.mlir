// RUN: env CIRCT_SIM_TRACE_FORK_JOIN=1 circt-sim %s --skip-passes --max-time=100000000 --max-process-steps=5000 2>&1 | FileCheck %s
//
// CHECK-COUNT-1: [FORK-INTERCEPT] proc=
// CHECK: done
// CHECK: [circt-sim] Simulation completed

module {
  func.func @__moore_wait_condition(%cond: i32) {
    return
  }

  func.func @__moore_delay(%delay_fs: i64) {
    return
  }

  func.func @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(
      %phase: i64) {
    return
  }

  hw.module @top() {
    %zero_i32 = arith.constant 0 : i32
    %zero_i64 = arith.constant 0 : i64
    %phase = arith.constant 1 : i64
    %fmt = sim.fmt.literal "done\0A"

    llhd.process {
      %h = sim.fork join_type "join_any" {
        func.call @__moore_wait_condition(%zero_i32) : (i32) -> ()
        sim.fork.terminator
      }, {
        func.call @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(%phase) : (i64) -> ()
        sim.fork.terminator
      }, {
        func.call @__moore_delay(%zero_i64) : (i64) -> ()
        sim.fork.terminator
      }
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
