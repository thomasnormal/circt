// RUN: env CIRCT_SIM_TRACE_SEQ=1 circt-sim %s --max-time 7000000 2>&1 | FileCheck %s
//
// Empty get_next_item waiters must remain parked until sequencer push wakeup.
// Unrelated signal events must not re-activate the blocked process.
//
// CHECK-COUNT-1: [SEQ-CI] wait port=
// CHECK: PASS: remained blocked on empty get_next_item
// CHECK-NOT: FAIL:
// CHECK-NOT: ERROR(DELTA_OVERFLOW)

module {
  llvm.mlir.global internal @port_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"]]
  } : !llvm.array<1 x ptr>

  func.func @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"(%port: !llvm.ptr, %ref: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %zero = hw.constant false
    %one = hw.constant true
    %tick = llhd.sig %zero : i1

    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %t2 = llhd.constant_time <2ns, 0d, 0e>
    %t3 = llhd.constant_time <3ns, 0d, 0e>
    %t4 = llhd.constant_time <4ns, 0d, 0e>
    %t5 = llhd.constant_time <5ns, 0d, 0e>
    %t6 = llhd.constant_time <6ns, 0d, 0e>

    %fmtPass = sim.fmt.literal "PASS: remained blocked on empty get_next_item\0A"
    %fmtFail = sim.fmt.literal "FAIL: get_next_item unexpectedly resumed\0A"

    // Arm on first tick edge, then block in empty get_next_item.
    llhd.process {
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %one64 = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one64 x !llvm.ptr : (i64) -> !llvm.ptr
      %tick_val = llhd.prb %tick : i1
      llhd.wait (%tick_val : i1), ^armed
    ^armed:
      %slot = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fn = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %cast = builtin.unrealized_conversion_cast %fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %cast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmtFail
      sim.terminate failure, quiet
      llhd.halt
    }

    // Generate unrelated events after the waiter is armed.
    llhd.process {
      llhd.drv %tick, %one after %t1 : i1
      llhd.drv %tick, %zero after %t2 : i1
      llhd.drv %tick, %one after %t3 : i1
      llhd.drv %tick, %zero after %t4 : i1
      llhd.drv %tick, %one after %t5 : i1
      llhd.halt
    }

    // End the test after enough unrelated events have fired.
    llhd.process {
      llhd.wait delay %t6, ^done
    ^done:
      sim.proc.print %fmtPass
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
