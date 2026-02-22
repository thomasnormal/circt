// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Verify that try_next_item on an unresolved/unconnected pull port does not
// dequeue from unrelated sequencer FIFOs.
//
// CHECK: PASS: try_next_item did not steal unrelated sequencer items
// CHECK-NOT: FAIL:

module {
  llvm.mlir.global internal @item_a_storage(11 : i64) : i64
  llvm.mlir.global internal @item_b_storage(22 : i64) : i64
  llvm.mlir.global internal @sequencer_a_addr(101 : i64) : i64
  llvm.mlir.global internal @sequencer_b_addr(202 : i64) : i64

  // Sequence vtable: slot 0 = start_item, slot 1 = finish_item.
  llvm.mlir.global internal @seq_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]
  } : !llvm.array<2 x ptr>

  // Pull-port vtable: slot 0 = get_next_item, slot 2 = item_done.
  llvm.mlir.global internal @port_get_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::get_next_item"], [2, @"uvm_pkg::uvm_seq_item_pull_port::item_done"]]
  } : !llvm.array<3 x ptr>

  // Pull-port vtable: slot 0 = try_next_item.
  llvm.mlir.global internal @port_try_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"]]
  } : !llvm.array<1 x ptr>

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
  func.func @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(
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
    %fmtPass = sim.fmt.literal "PASS: try_next_item did not steal unrelated sequencer items\0A"
    %fmtFail = sim.fmt.literal "FAIL: try_next_item dequeued unrelated item\0A"

    // Sequence A pushes item A on sequencer A.
    llhd.process {
      cf.br ^start
    ^start:
      %itemA = llvm.mlir.addressof @item_a_storage : !llvm.ptr
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
      func.call_indirect %startCast(%null, %itemA, %neg1, %sqrA) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %itemA, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Sequence B pushes item B on sequencer B.
    llhd.process {
      cf.br ^start
    ^start:
      %itemB = llvm.mlir.addressof @item_b_storage : !llvm.ptr
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
      func.call_indirect %startCast(%null, %itemB, %neg1, %sqrB) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %itemB, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Driver:
    // 1) try_next_item on an unconnected port must return null and dequeue
    //    nothing.
    // 2) connect to sequencer A and consume item A.
    // 3) reconnect to sequencer B and consume item B.
    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t1ns, ^run
    ^run:
      %itemA = llvm.mlir.addressof @item_a_storage : !llvm.ptr
      %itemB = llvm.mlir.addressof @item_b_storage : !llvm.ptr
      %sqrA = llvm.mlir.addressof @sequencer_a_addr : !llvm.ptr
      %sqrB = llvm.mlir.addressof @sequencer_b_addr : !llvm.ptr
      %portGet = llvm.mlir.addressof @port_get_vtable : !llvm.ptr
      %portTry = llvm.mlir.addressof @port_try_vtable : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %null, %ref : !llvm.ptr, !llvm.ptr

      %slotTry = llvm.getelementptr %portTry[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fnTry = llvm.load %slotTry : !llvm.ptr -> !llvm.ptr
      %tryCast = builtin.unrealized_conversion_cast %fnTry : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()

      %slotGet = llvm.getelementptr %portGet[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnGet = llvm.load %slotGet : !llvm.ptr -> !llvm.ptr
      %getCast = builtin.unrealized_conversion_cast %fnGet : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()

      %slotDone = llvm.getelementptr %portGet[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnDone = llvm.load %slotDone : !llvm.ptr -> !llvm.ptr
      %doneCast = builtin.unrealized_conversion_cast %fnDone : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()

      func.call_indirect %tryCast(%portTry, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %tryItem = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %tryIsNull = llvm.icmp "eq" %tryItem, %null : !llvm.ptr

      func.call @"uvm_pkg::uvm_port_base::connect"(%portGet, %sqrA) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %getCast(%portGet, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %firstItem = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %firstIsA = llvm.icmp "eq" %firstItem, %itemA : !llvm.ptr
      func.call_indirect %doneCast(%portGet, %null) : (!llvm.ptr, !llvm.ptr) -> ()

      func.call @"uvm_pkg::uvm_port_base::connect"(%portGet, %sqrB) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %getCast(%portGet, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %secondItem = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %secondIsB = llvm.icmp "eq" %secondItem, %itemB : !llvm.ptr
      func.call_indirect %doneCast(%portGet, %null) : (!llvm.ptr, !llvm.ptr) -> ()

      %ok0 = comb.and %tryIsNull, %firstIsA : i1
      %ok = comb.and %ok0, %secondIsB : i1
      cf.cond_br %ok, ^pass, ^fail

    ^pass:
      sim.proc.print %fmtPass
      sim.terminate success, quiet
      llhd.halt

    ^fail:
      sim.proc.print %fmtFail
      sim.terminate failure, quiet
      llhd.halt
    }

    hw.output
  }
}
