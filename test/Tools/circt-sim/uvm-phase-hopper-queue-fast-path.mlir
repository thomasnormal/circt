// RUN: circt-sim %s | FileCheck %s

// Verify the targeted native queue fast-path for uvm_phase_hopper::try_put/get.
// The function bodies below intentionally do the wrong thing; the simulator
// interceptors should bypass those bodies and transfer the queued phase pointer.

module {
  func.func private @"uvm_pkg::uvm_phase_hopper::try_put"(%hopper: !llvm.ptr, %phase: !llvm.ptr) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func private @"uvm_pkg::uvm_phase_hopper::get"(%hopper: !llvm.ptr, %out: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.store %null, %out : !llvm.ptr, !llvm.ptr
    return
  }

  func.func private @"uvm_pkg::uvm_phase_hopper::try_get"(%hopper: !llvm.ptr, %out: !llvm.ptr) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func private @"uvm_pkg::uvm_phase_hopper::get_objection"(%hopper: !llvm.ptr) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }

  func.func private @"uvm_pkg::uvm_objection::get_objection_total"(%obj: !llvm.ptr) -> i32 {
    %minus1 = hw.constant -1 : i32
    return %minus1 : i32
  }

  hw.module @main() {
    %fmtPrefix = sim.fmt.literal "match = "
    %fmtOk = sim.fmt.literal " ok = "
    %fmtCount = sim.fmt.literal " count = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %hopper_i64 = llvm.mlir.constant(4096 : i64) : i64
      %phase_i64_const = llvm.mlir.constant(4660 : i64) : i64
      %hopper = llvm.inttoptr %hopper_i64 : i64 to !llvm.ptr
      %phase = llvm.inttoptr %phase_i64_const : i64 to !llvm.ptr
      %out = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      llvm.store %null, %out : !llvm.ptr, !llvm.ptr

      %ignore = func.call @"uvm_pkg::uvm_phase_hopper::try_put"(%hopper, %phase) : (!llvm.ptr, !llvm.ptr) -> i1
      // First get uses an unwritable output pointer. The fast-path must
      // decline without draining the queue.
      func.call @"uvm_pkg::uvm_phase_hopper::get"(%hopper, %null) : (!llvm.ptr, !llvm.ptr) -> ()
      %ok = func.call @"uvm_pkg::uvm_phase_hopper::try_get"(%hopper, %out) : (!llvm.ptr, !llvm.ptr) -> i1

      %got = llvm.load %out : !llvm.ptr -> !llvm.ptr
      %got_i64 = llvm.ptrtoint %got : !llvm.ptr to i64
      %match = comb.icmp eq %got_i64, %phase_i64_const : i64
      %ok_i32 = arith.extui %ok : i1 to i32
      %match_i32 = arith.extui %match : i1 to i32
      %okFmt = sim.fmt.dec %ok_i32 : i32
      %matchFmt = sim.fmt.dec %match_i32 : i32
      %obj = func.call @"uvm_pkg::uvm_phase_hopper::get_objection"(%hopper) : (!llvm.ptr) -> !llvm.ptr
      %count = func.call @"uvm_pkg::uvm_objection::get_objection_total"(%obj) : (!llvm.ptr) -> i32
      %countFmt = sim.fmt.dec %count : i32
      %line = sim.fmt.concat (%fmtPrefix, %matchFmt, %fmtOk, %okFmt, %fmtCount, %countFmt, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: match = 1 ok = 1 count = 0
