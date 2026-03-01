// RUN: circt-sim %s | FileCheck %s
//
// Regression: sequencer send_request fast-path must not silently swallow calls
// when no valid sequencer queue can be resolved. In that case we must fall back
// to canonical function-body execution.
//
// CHECK: send_request_fallback_counter = 1
// CHECK-NOT: send_request_fallback_counter = 0

module {
  llvm.mlir.global internal @counter(0 : i32) : i32

  func.func @"uvm_pkg::uvm_sequencer_base::send_request"(%sqr: !llvm.ptr,
                                                         %seq: !llvm.ptr,
                                                         %item: !llvm.ptr) {
    %addr = llvm.mlir.addressof @counter : !llvm.ptr
    %cur = llvm.load %addr : !llvm.ptr -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %cur, %one : i32
    llvm.store %next, %addr : i32, !llvm.ptr
    return
  }

  hw.module @top() {
    %prefix = sim.fmt.literal "send_request_fallback_counter = "
    %nl = sim.fmt.literal "\0A"
    llhd.process {
      %one64 = llvm.mlir.constant(1 : i64) : i64
      %zero64 = llvm.mlir.constant(0 : i64) : i64
      %null = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %item = llvm.alloca %one64 x i8 : (i64) -> !llvm.ptr

      // Null sequencer pointer: fast-path cannot resolve a queue and must
      // fall back to the canonical function body.
      func.call @"uvm_pkg::uvm_sequencer_base::send_request"(%null, %null, %item) :
          (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

      %addr = llvm.mlir.addressof @counter : !llvm.ptr
      %cur = llvm.load %addr : !llvm.ptr -> i32
      %curDec = sim.fmt.dec %cur signed : i32
      %line = sim.fmt.concat (%prefix, %curDec, %nl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
