// RUN: circt-sim %s | FileCheck %s

// Verify function-body hopper fast paths that catch non-canonical symbol names.
// The names intentionally do not end with "::try_put"/"::get", so call-site
// interceptors should miss and function-entry fast paths should handle them.

module {
  func.func private @"uvm_pkg::uvm_phase_hopper::try_put_shim"(%hopper: !llvm.ptr, %phase: !llvm.ptr) -> i1 {
    %false = hw.constant false
    return %false : i1
  }

  func.func private @"uvm_pkg::uvm_phase_hopper::get_shim"(%hopper: !llvm.ptr, %out: !llvm.ptr) {
    %null = llvm.mlir.zero : !llvm.ptr
    llvm.store %null, %out : !llvm.ptr, !llvm.ptr
    return
  }

  hw.module @main() {
    %fmtPrefix = sim.fmt.literal "ok/match = "
    %fmtSlash = sim.fmt.literal "/"
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %hopper_i64 = llvm.mlir.constant(8192 : i64) : i64
      %phase_i64_const = llvm.mlir.constant(8738 : i64) : i64
      %hopper = llvm.inttoptr %hopper_i64 : i64 to !llvm.ptr
      %phase = llvm.inttoptr %phase_i64_const : i64 to !llvm.ptr
      %out = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr
      llvm.store %null, %out : !llvm.ptr, !llvm.ptr

      %ok = func.call @"uvm_pkg::uvm_phase_hopper::try_put_shim"(%hopper, %phase) : (!llvm.ptr, !llvm.ptr) -> i1
      func.call @"uvm_pkg::uvm_phase_hopper::get_shim"(%hopper, %out) : (!llvm.ptr, !llvm.ptr) -> ()

      %got = llvm.load %out : !llvm.ptr -> !llvm.ptr
      %got_i64 = llvm.ptrtoint %got : !llvm.ptr to i64
      %match = comb.icmp eq %got_i64, %phase_i64_const : i64

      %ok_i32 = arith.extui %ok : i1 to i32
      %okFmt = sim.fmt.dec %ok_i32 : i32
      %match_i32 = arith.extui %match : i1 to i32
      %matchFmt = sim.fmt.dec %match_i32 : i32
      %line = sim.fmt.concat (%fmtPrefix, %okFmt, %fmtSlash, %matchFmt, %fmtNl)
      sim.proc.print %line

      llhd.halt
    }

    hw.output
  }
}

// CHECK: ok/match = 1/1
