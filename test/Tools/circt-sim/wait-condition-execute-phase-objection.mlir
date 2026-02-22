// RUN: env CIRCT_SIM_TRACE_WAIT_CONDITION=1 circt-sim %s --max-time=1000000000 --max-process-steps=300 2>&1 | FileCheck %s
// Regression: execute_phase can hit wait(condition) with no analyzable SSA deps
// (constant/opaque condition). It must use objection-backed waiting instead of
// tight delta polling so time can advance and other processes can run.

// CHECK: [WAITCOND] proc=
// CHECK: func=uvm_pkg::uvm_phase_hopper::execute_phase
// CHECK: objectionWaitHandle={{[0-9]+}}
// CHECK: Trigger terminate
// CHECK-NOT: ERROR(PROCESS_STEP_OVERFLOW)

module {
  llvm.func @__moore_wait_condition(i32)

  llvm.func @"uvm_pkg::uvm_phase_hopper::execute_phase"(%self: i64, %phase: i64) {
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llvm.call @__moore_wait_condition(%c0_i32) : (i32) -> ()
    llvm.return
  }

  hw.module @top() {
    %c0_i64 = hw.constant 0 : i64
    %c42_i64 = hw.constant 42 : i64
    %c100000000_i64 = hw.constant 100000000 : i64

    %fmt_start = sim.fmt.literal "A start\0A"
    %fmt_terminate = sim.fmt.literal "Trigger terminate\0A"

    // Process A runs execute_phase, which blocks in wait(condition) false.
    llhd.process {
      sim.proc.print %fmt_start
      llvm.call @"uvm_pkg::uvm_phase_hopper::execute_phase"(%c0_i64, %c42_i64) : (i64, i64) -> ()
      llhd.halt
    }

    // Process B advances time and terminates simulation.
    llhd.process {
      %delay = llhd.int_to_time %c100000000_i64
      llhd.wait delay %delay, ^bb2
    ^bb2:
      sim.proc.print %fmt_terminate
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
