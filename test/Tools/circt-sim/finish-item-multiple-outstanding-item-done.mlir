// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s --max-time 100000000 2>&1 | FileCheck %s --check-prefix=CHECK-SUMMARY
//
// Regression: multiple get_next_item calls can dequeue more than one item from
// the same pull-port before item_done is issued. item_done must still resolve
// the oldest outstanding dequeued item and release both finish_item waiters.
//
// CHECK: seq1: finish_item returned
// CHECK: seq2: finish_item returned
// CHECK: [circt-sim] Simulation completed
//
// CHECK-SUMMARY: [circt-sim] UVM sequencer native state: item_map_live=0 item_map_peak=1 item_map_stores=2 item_map_erases=2
// CHECK-SUMMARY-SAME: done_pending=0 last_dequeued=0

module {
  llvm.mlir.global internal @item_storage_1(0 : i64) : i64
  llvm.mlir.global internal @item_storage_2(0 : i64) : i64
  llvm.mlir.global internal @sequencer_addr(42 : i64) : i64

  llvm.mlir.global internal @seq_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]} : !llvm.array<2 x ptr>
  llvm.mlir.global internal @port_vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]} : !llvm.array<3 x ptr>

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
    %t2ns = llhd.constant_time <2ns, 0d, 0e>
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %fmt_s1_ret = sim.fmt.literal "seq1: finish_item returned\0A"
    %fmt_s2_ret = sim.fmt.literal "seq2: finish_item returned\0A"
    %fmt_done1 = sim.fmt.literal "driver: item_done #1\0A"
    %fmt_done2 = sim.fmt.literal "driver: item_done #2\0A"

    // Sequence 1: enqueue item #1 and block in finish_item.
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

    // Sequence 2: enqueue item #2 and block in finish_item.
    llhd.process {
      cf.br ^start
    ^start:
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

    // Driver: dequeue two items before signaling either completion.
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t2ns, ^run
    ^run:
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref1 = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %ref2 = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr

      %gni_slot = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %gni_fn = llvm.load %gni_slot : !llvm.ptr -> !llvm.ptr
      %gni_cast = builtin.unrealized_conversion_cast %gni_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %gni_cast(%port, %ref1) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %gni_cast(%port, %ref2) : (!llvm.ptr, !llvm.ptr) -> ()

      llhd.wait delay %t1ns, ^done
    ^done:
      %id_slot = llvm.getelementptr %port[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %id_fn = llvm.load %id_slot : !llvm.ptr -> !llvm.ptr
      %id_cast = builtin.unrealized_conversion_cast %id_fn : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %id_cast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_done1
      func.call_indirect %id_cast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.proc.print %fmt_done2
      llhd.halt
    }

    hw.output
  }
}
