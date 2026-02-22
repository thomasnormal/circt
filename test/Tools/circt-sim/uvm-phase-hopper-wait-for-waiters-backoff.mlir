// RUN: circt-sim %s --max-deltas=64 --max-time=3000000 2>&1 | FileCheck %s
// RUN: circt-sim %s --max-deltas=64 --max-time=3000000 --max-process-steps=40 2>&1 | FileCheck %s --check-prefix=CHECK-STEPS
//
// uvm_phase_hopper::wait_for_waiters is lowered as a zero-delay yield. Keep
// progress bounded so this polling shape cannot overflow deltas at one time.
//
// CHECK-NOT: ERROR(DELTA_OVERFLOW)
// CHECK: [circt-sim] Simulation completed at time 2000000 fs
// CHECK-STEPS-NOT: ERROR(PROCESS_STEP_OVERFLOW)
// CHECK-STEPS: [circt-sim] Simulation completed at time 2000000 fs

module {
  llvm.mlir.global internal @hopper_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_phase_hopper::wait_for_waiters"]]
  } : !llvm.array<1 x ptr>

  llvm.func @__moore_delay(i64)

  func.func @"uvm_pkg::uvm_phase_hopper::wait_for_waiters"(
      %self: !llvm.ptr, %phase: !llvm.ptr, %state: i32) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    llvm.call @__moore_delay(%zero) : (i64) -> ()
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero32 = llvm.mlir.constant(0 : i32) : i32
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %t2ns = llhd.constant_time <2ns, 0d, 0e>

    // Repeatedly call wait_for_waiters through vtable dispatch after time > 0.
    llhd.process {
      %hopper = llvm.mlir.addressof @hopper_vtable : !llvm.ptr
      %slot = llvm.getelementptr %hopper[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fn = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %cast = builtin.unrealized_conversion_cast %fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.wait delay %t1ns, ^loop
    ^loop:
      func.call_indirect %cast(%hopper, %null, %zero32) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      cf.br ^loop
    }

    // Bound the test to 2 ns.
    llhd.process {
      llhd.wait delay %t2ns, ^bb1
    ^bb1:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
