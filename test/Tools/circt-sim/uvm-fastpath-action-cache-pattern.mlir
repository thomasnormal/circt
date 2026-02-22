// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify non-exact UVM fast-path symbols use cached pattern resolution.
//
// CHECK: wrapper sum = 0

module {
  // If this body executes, it returns 123. The printer fast-path should no-op
  // and force zero results instead.
  func.func private @"uvm_pkg::custom_printer::uvm_printer::print_field_int_wrapper"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  // Two callsites force a callee-name cache hit on the second resolve while
  // bypassing per-site cache hits.
  func.func private @drive_a(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_wrapper"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  func.func private @drive_b(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_wrapper"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "wrapper sum = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %value = arith.constant 9 : i32

      %a = func.call @drive_a(%self, %value) : (!llvm.ptr, i32) -> i32
      %b = func.call @drive_b(%self, %value) : (!llvm.ptr, i32) -> i32
      %sum = arith.addi %a, %b : i32

      %sumFmt = sim.fmt.dec %sum signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %sumFmt, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
