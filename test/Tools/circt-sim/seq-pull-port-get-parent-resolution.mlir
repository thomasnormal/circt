// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Verify pull-port queue resolution uses get_parent() on a terminal pull-imp.
// This maps port->imp native connections to the owning sequencer queue without
// opportunistic queue stealing.
//
// CHECK: PASS: get_parent resolved pull-imp to sequencer queue
// CHECK-NOT: FAIL:

module {
  llvm.mlir.global internal @item_storage(11 : i64) : i64
  llvm.mlir.global internal @sequencer_addr(101 : i64) : i64
  llvm.mlir.global internal @pull_port_addr(202 : i64) : i64
  llvm.mlir.global internal @pull_imp_addr(303 : i64) : i64

  llvm.mlir.global internal @seq_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_sequence_base::start_item"], [1, @"uvm_pkg::uvm_sequence_base::finish_item"]]
  } : !llvm.array<2 x ptr>

  func.func @"uvm_pkg::uvm_port_base::connect"(%self: !llvm.ptr, %provider: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_port_base::get_parent"(%self: !llvm.ptr) -> !llvm.ptr {
    %seqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
    return %seqr : !llvm.ptr
  }

  func.func @"uvm_pkg::uvm_sequence_base::start_item"(
      %self: !llvm.ptr, %item: !llvm.ptr, %pri: i32, %sqr: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_sequence_base::finish_item"(
      %self: !llvm.ptr, %item: !llvm.ptr, %pri: i32) {
    return
  }

  func.func @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(
      %port: !llvm.ptr, %out: !llvm.ptr) {
    return
  }

  func.func @"uvm_pkg::uvm_seq_item_pull_port::item_done"(
      %port: !llvm.ptr, %rsp: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t1ns = llhd.constant_time <1ns, 0d, 0e>
    %fmtPass = sim.fmt.literal "PASS: get_parent resolved pull-imp to sequencer queue\0A"
    %fmtFail = sim.fmt.literal "FAIL: pull-imp queue resolution failed\0A"

    llhd.process {
      cf.br ^start
    ^start:
      %item = llvm.mlir.addressof @item_storage : !llvm.ptr
      %seqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %slotStart = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnStart = llvm.load %slotStart : !llvm.ptr -> !llvm.ptr
      %startCast = builtin.unrealized_conversion_cast %fnStart : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %slotFinish = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnFinish = llvm.load %slotFinish : !llvm.ptr -> !llvm.ptr
      %finishCast = builtin.unrealized_conversion_cast %fnFinish : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()
      func.call_indirect %startCast(%null, %item, %neg1, %seqr) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %item, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    llhd.process {
      cf.br ^start
    ^start:
      llhd.wait delay %t1ns, ^run
    ^run:
      %item = llvm.mlir.addressof @item_storage : !llvm.ptr
      %port = llvm.mlir.addressof @pull_port_addr : !llvm.ptr
      %imp = llvm.mlir.addressof @pull_imp_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      llvm.store %null, %ref : !llvm.ptr, !llvm.ptr

      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %imp) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call @"uvm_pkg::uvm_seq_item_pull_port::try_next_item"(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()
      %got = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %ok = llvm.icmp "eq" %got, %item : !llvm.ptr
      cf.cond_br %ok, ^pass, ^fail

    ^pass:
      func.call @"uvm_pkg::uvm_seq_item_pull_port::item_done"(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
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
