// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Verify that reconnecting a seq_item_pull_port updates native routing for
// get_next_item. This specifically guards against stale provider selection when
// a prior get cached/observed an older connection.
//
// CHECK: PASS: reconnect routed get_next_item to newest provider
// CHECK-NOT: FAIL:

module {
  // Distinct item objects.
  llvm.mlir.global internal @item_a1_storage(11 : i64) : i64
  llvm.mlir.global internal @item_a2_storage(22 : i64) : i64
  llvm.mlir.global internal @item_b1_storage(33 : i64) : i64

  // Distinct sequencer/export identities.
  llvm.mlir.global internal @sequencer_a_addr(101 : i64) : i64
  llvm.mlir.global internal @sequencer_b_addr(202 : i64) : i64

  // Sequence vtable: slot 0 = start_item, slot 1 = finish_item.
  llvm.mlir.global internal @seq_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]
  } : !llvm.array<2 x ptr>

  // Pull-port vtable: slot 0 = get_next_item, slot 2 = item_done.
  llvm.mlir.global internal @port_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]
  } : !llvm.array<3 x ptr>

  // Intercept targets (dummy bodies; native intercepts run before body).
  func.func @"uvm_pkg::uvm_sequence_base::start_item"(
      %self: !llvm.ptr, %item: !llvm.ptr, %pri: i32, %sqr: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_sequence_base::finish_item"(
      %self: !llvm.ptr, %item: !llvm.ptr, %pri: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"(
      %port: !llvm.ptr, %ref: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_seq_item_pull_port::item_done"(
      %port: !llvm.ptr, %rsp: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_port_base::connect"(
      %self: !llvm.ptr, %provider: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %t2ns = llhd.constant_time <2ns, 0d, 0e>
    %fmtPass = sim.fmt.literal "PASS: reconnect routed get_next_item to newest provider\0A"
    %fmtFail = sim.fmt.literal "FAIL: reconnect routing mismatch\0A"

    // Sequence A pushes item A1, then item A2 on sequencer A.
    llhd.process {
      cf.br ^start
    ^start:
      %itemA1 = llvm.mlir.addressof @item_a1_storage : !llvm.ptr
      %itemA2 = llvm.mlir.addressof @item_a2_storage : !llvm.ptr
      %sqrA = llvm.mlir.addressof @sequencer_a_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %slotStart = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnStart = llvm.load %slotStart : !llvm.ptr -> !llvm.ptr
      %startCast = builtin.unrealized_conversion_cast %fnStart : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %slotFinish = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnFinish = llvm.load %slotFinish : !llvm.ptr -> !llvm.ptr
      %finishCast = builtin.unrealized_conversion_cast %fnFinish : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()

      func.call_indirect %startCast(%null, %itemA1, %neg1, %sqrA) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %itemA1, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      func.call_indirect %startCast(%null, %itemA2, %neg1, %sqrA) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %itemA2, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Sequence B pushes item B1 on sequencer B.
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t1ns, ^push
    ^push:
      %itemB1 = llvm.mlir.addressof @item_b1_storage : !llvm.ptr
      %sqrB = llvm.mlir.addressof @sequencer_b_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %slotStart = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnStart = llvm.load %slotStart : !llvm.ptr -> !llvm.ptr
      %startCast = builtin.unrealized_conversion_cast %fnStart : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %slotFinish = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnFinish = llvm.load %slotFinish : !llvm.ptr -> !llvm.ptr
      %finishCast = builtin.unrealized_conversion_cast %fnFinish : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()

      func.call_indirect %startCast(%null, %itemB1, %neg1, %sqrB) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %itemB1, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Driver: connect to A, consume A1, reconnect to B, and verify B1 is next.
    llhd.process {
      cf.br ^start
    ^start:
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %sqrA = llvm.mlir.addressof @sequencer_a_addr : !llvm.ptr
      %sqrB = llvm.mlir.addressof @sequencer_b_addr : !llvm.ptr
      %itemA1 = llvm.mlir.addressof @item_a1_storage : !llvm.ptr
      %itemB1 = llvm.mlir.addressof @item_b1_storage : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr

      %slotGet = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnGet = llvm.load %slotGet : !llvm.ptr -> !llvm.ptr
      %getCast = builtin.unrealized_conversion_cast %fnGet : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      %slotDone = llvm.getelementptr %port[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnDone = llvm.load %slotDone : !llvm.ptr -> !llvm.ptr
      %doneCast = builtin.unrealized_conversion_cast %fnDone : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()

      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %sqrA) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %getCast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %firstItem = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %firstIsA1 = llvm.icmp "eq" %firstItem, %itemA1 : !llvm.ptr
      cf.cond_br %firstIsA1, ^afterFirst, ^fail

    ^afterFirst:
      func.call_indirect %doneCast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      llhd.wait delay %t2ns, ^secondGet

    ^secondGet:
      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %sqrB) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %getCast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %secondItem = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %secondIsB1 = llvm.icmp "eq" %secondItem, %itemB1 : !llvm.ptr
      cf.cond_br %secondIsB1, ^pass, ^fail

    ^pass:
      sim.proc.print %fmtPass
      func.call_indirect %doneCast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.terminate success, quiet
      llhd.halt

    ^fail:
      sim.proc.print %fmtFail
      func.call_indirect %doneCast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.terminate failure, quiet
      llhd.halt
    }

    hw.output
  }
}
