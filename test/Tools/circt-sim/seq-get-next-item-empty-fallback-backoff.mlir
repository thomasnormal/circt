// RUN: circt-sim %s --max-deltas=8 --max-time=1000000 2>&1 | FileCheck %s
//
// Empty sequencer get_next_item retries should switch to sparse timed polling
// before exhausting the scheduler delta budget.
//
// CHECK-NOT: ERROR(DELTA_OVERFLOW)
// CHECK: [circt-sim] Simulation completed at time 1000000 fs

module {
  llvm.mlir.global internal @port_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"]]} : !llvm.array<1 x ptr>

  func.func @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"(%port: !llvm.ptr, %ref: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t1ns = llhd.constant_time <1ns, 0d, 0e>

    // This call never gets an item, so the interpreter's get_next_item retry
    // path handles progress while remaining blocked.
    llhd.process {
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %slot = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fn = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %cast = builtin.unrealized_conversion_cast %fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %cast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      llhd.halt
    }

    // Force completion at 1ns so the test remains bounded.
    llhd.process {
      llhd.wait delay %t1ns, ^bb1
    ^bb1:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
