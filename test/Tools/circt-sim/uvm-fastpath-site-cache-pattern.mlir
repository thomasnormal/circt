// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify per-callsite UVM action cache hits on repeated executions of one
// func.call site.
//
// CHECK: site-cache sum = 0

module {
  // If this body executes, it returns 123. Fast-path should force zero.
  func.func private @"uvm_pkg::custom_printer::uvm_printer::print_field_int_sitecache"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  // One callsite executed twice in a loop.
  func.func private @drive(%self: !llvm.ptr, %value: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c0idx = arith.constant 0 : index
    %c1idx = arith.constant 1 : index
    %c2idx = arith.constant 2 : index
    %sum = scf.for %i = %c0idx to %c2idx step %c1idx iter_args(%acc = %c0) -> (i32) {
      %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_sitecache"(%self, %value) :
          (!llvm.ptr, i32) -> i32
      %next = arith.addi %acc, %v : i32
      scf.yield %next : i32
    }
    return %sum : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "site-cache sum = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %value = arith.constant 9 : i32

      %sum = func.call @drive(%self, %value) : (!llvm.ptr, i32) -> i32
      %sumFmt = sim.fmt.dec %sum signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %sumFmt, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
