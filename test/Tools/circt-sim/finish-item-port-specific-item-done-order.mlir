// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s --max-time 100000000 2>&1 | FileCheck %s --check-prefix=CHECK-SUMMARY
//
// Regression: when one driver process dequeues items from multiple pull ports,
// item_done(self) must resolve by the calling port first (not just by process
// FIFO order). Otherwise item_done(portB) can incorrectly complete portA's
// item.
//
// CHECK: driver: item_done B
// CHECK: seq2: finish_item returned
// CHECK: driver: item_done A
// CHECK: seq1: finish_item returned
// CHECK: [circt-sim] Simulation completed
//
// CHECK-SUMMARY: [circt-sim] UVM sequencer native state:
// CHECK-SUMMARY-SAME: done_pending=0 last_dequeued=0

module {
  llvm.mlir.global internal @item_storage_1(0 : i64) : i64
  llvm.mlir.global internal @item_storage_2(0 : i64) : i64
  llvm.mlir.global internal @sequencer_addr(99 : i64) : i64

  llvm.mlir.global internal @seq_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]} : !llvm.array<2 x ptr>
  llvm.mlir.global internal @port_a_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]} : !llvm.array<3 x ptr>
  llvm.mlir.global internal @port_b_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]} : !llvm.array<3 x ptr>

  func.func @"uvm_pkg::uvm_sequence_base::start_item"(%self: !llvm.ptr, %item: !llvm.ptr, %pri: i32, %sqr: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_sequence_base::finish_item"(%self: !llvm.ptr, %item: !llvm.ptr, %pri: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"(%port: !llvm.ptr, %ref: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::item_done"(%port: !llvm.ptr, %rsp: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %t3ns = llhd.constant_time <3ns, 0d, 0e>
    %fmt_s1_ret = sim.fmt.literal "seq1: finish_item returned\0A"
    %fmt_s2_ret = sim.fmt.literal "seq2: finish_item returned\0A"
    %fmt_done_a = sim.fmt.literal "driver: item_done A\0A"
    %fmt_done_b = sim.fmt.literal "driver: item_done B\0A"

    // Sequence 1 (first item in FIFO).
    llhd.process {
      cf.br ^start
    ^start:
      %item = llvm.mlir.addressof @item_storage_1 : !llvm.ptr
      %sqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %si_slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %si_fn = llvm.load %si_slot : !llvm.ptr -> !llvm.ptr
      %si_cast = builtin.unrealized_conversion_cast %si_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %si_cast(%null, %item, %neg1, %sqr) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %fi_slot = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fi_fn = llvm.load %fi_slot : !llvm.ptr -> !llvm.ptr
      %fi_cast = builtin.unrealized_conversion_cast %fi_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()
      func.call_indirect %fi_cast(%null, %item, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      sim.proc.print %fmt_s1_ret
      llhd.halt
    }

    // Sequence 2 (second item in FIFO, delayed for deterministic order).
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t1ns, ^push
    ^push:
      %item = llvm.mlir.addressof @item_storage_2 : !llvm.ptr
      %sqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %si_slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %si_fn = llvm.load %si_slot : !llvm.ptr -> !llvm.ptr
      %si_cast = builtin.unrealized_conversion_cast %si_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %si_cast(%null, %item, %neg1, %sqr) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %fi_slot = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fi_fn = llvm.load %fi_slot : !llvm.ptr -> !llvm.ptr
      %fi_cast = builtin.unrealized_conversion_cast %fi_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()
      func.call_indirect %fi_cast(%null, %item, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      sim.proc.print %fmt_s2_ret
      llhd.halt
    }

    // Driver process: dequeue from two different pull ports and complete B
    // first, then A.
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t3ns, ^run
    ^run:
      %port_a = llvm.mlir.addressof @port_a_vtable : !llvm.ptr
      %port_b = llvm.mlir.addressof @port_b_vtable : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref_a = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %ref_b = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr

      %gni_slot_a = llvm.getelementptr %port_a[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %gni_fn_a = llvm.load %gni_slot_a : !llvm.ptr -> !llvm.ptr
      %gni_cast_a = builtin.unrealized_conversion_cast %gni_fn_a : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %gni_cast_a(%port_a, %ref_a) : (!llvm.ptr, !llvm.ptr) -> ()

      %gni_slot_b = llvm.getelementptr %port_b[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %gni_fn_b = llvm.load %gni_slot_b : !llvm.ptr -> !llvm.ptr
      %gni_cast_b = builtin.unrealized_conversion_cast %gni_fn_b : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %gni_cast_b(%port_b, %ref_b) : (!llvm.ptr, !llvm.ptr) -> ()

      %id_slot_b = llvm.getelementptr %port_b[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %id_fn_b = llvm.load %id_slot_b : !llvm.ptr -> !llvm.ptr
      %id_cast_b = builtin.unrealized_conversion_cast %id_fn_b : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %id_cast_b(%port_b, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_done_b

      llhd.wait delay %t1ns, ^done_a
    ^done_a:
      %id_slot_a = llvm.getelementptr %port_a[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %id_fn_a = llvm.load %id_slot_a : !llvm.ptr -> !llvm.ptr
      %id_cast_a = builtin.unrealized_conversion_cast %id_fn_a : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %id_cast_a(%port_a, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_done_a
      llhd.halt
    }

    hw.output
  }
}
