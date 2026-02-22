// RUN: circt-sim %s --max-deltas=8 --max-time=3000000000 2>&1 | FileCheck %s
//
// Empty get_next_item should block on a queue waiter and wake on finish_item
// push, rather than spin in delta-cycle polling.
//
// CHECK: PASS: get_next_item resumed on fifo push
// CHECK-NOT: FAIL:
// CHECK-NOT: TIMEOUT:
// CHECK-NOT: ERROR(DELTA_OVERFLOW)

module {
  llvm.mlir.global internal @item_storage(1234 : i64) : i64
  llvm.mlir.global internal @sequencer_addr(777 : i64) : i64

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

  // Intercept targets (native intercepts run before these bodies).
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
    %fmtPass = sim.fmt.literal "PASS: get_next_item resumed on fifo push\0A"
    %fmtFail = sim.fmt.literal "FAIL: unexpected item value\0A"
    %fmtTimeout = sim.fmt.literal "TIMEOUT: get_next_item did not wake\0A"

    // Producer pushes one item after 1ns.
    llhd.process {
      llhd.wait delay %t1ns, ^bb1
    ^bb1:
      %item = llvm.mlir.addressof @item_storage : !llvm.ptr
      %sqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %neg1 = llvm.mlir.constant(-1 : i32) : i32
      %vt = llvm.mlir.addressof @seq_vtable : !llvm.ptr
      %slotStart = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnStart = llvm.load %slotStart : !llvm.ptr -> !llvm.ptr
      %startCast = builtin.unrealized_conversion_cast %fnStart : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      %slotFinish = llvm.getelementptr %vt[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fnFinish = llvm.load %slotFinish : !llvm.ptr -> !llvm.ptr
      %finishCast = builtin.unrealized_conversion_cast %fnFinish : !llvm.ptr to (!llvm.ptr, !llvm.ptr, i32) -> ()

      func.call_indirect %startCast(%null, %item, %neg1, %sqr) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
      func.call_indirect %finishCast(%null, %item, %neg1) : (!llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    // Consumer should block in get_next_item until producer pushes.
    llhd.process {
      %port = llvm.mlir.addressof @port_vtable : !llvm.ptr
      %sqr = llvm.mlir.addressof @sequencer_addr : !llvm.ptr
      %item = llvm.mlir.addressof @item_storage : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      %one = llvm.mlir.constant(1 : i64) : i64
      %ref = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr

      %slotGet = llvm.getelementptr %port[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnGet = llvm.load %slotGet : !llvm.ptr -> !llvm.ptr
      %getCast = builtin.unrealized_conversion_cast %fnGet : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()
      %slotDone = llvm.getelementptr %port[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x ptr>
      %fnDone = llvm.load %slotDone : !llvm.ptr -> !llvm.ptr
      %doneCast = builtin.unrealized_conversion_cast %fnDone : !llvm.ptr to (!llvm.ptr, !llvm.ptr) -> ()

      func.call @"uvm_pkg::uvm_port_base::connect"(%port, %sqr) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call_indirect %getCast(%port, %ref) : (!llvm.ptr, !llvm.ptr) -> ()

      %got = llvm.load %ref : !llvm.ptr -> !llvm.ptr
      %ok = llvm.icmp "eq" %got, %item : !llvm.ptr
      cf.cond_br %ok, ^pass, ^fail

    ^pass:
      sim.proc.print %fmtPass
      func.call_indirect %doneCast(%port, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      sim.terminate success, quiet
      llhd.halt

    ^fail:
      sim.proc.print %fmtFail
      sim.terminate failure, quiet
      llhd.halt
    }

    // Watchdog for regressions that fail to wake the blocked consumer.
    llhd.process {
      llhd.wait delay %t2ns, ^bb1
    ^bb1:
      sim.proc.print %fmtTimeout
      sim.terminate failure, quiet
      llhd.halt
    }

    hw.output
  }
}
