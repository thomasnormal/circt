// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 CIRCT_SIM_UVM_JIT_FASTPATH_SITE_CACHE_MAX_ENTRIES=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify per-callsite action cache cap without eviction.
//
// CHECK: site-cap sum = 0

module {
  // If executed, this returns non-zero. Fast-path should zero it.
  func.func private @"uvm_pkg::custom_printer::uvm_printer::print_field_int_site_cap"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c77 = arith.constant 77 : i32
    return %c77 : i32
  }

  // Distinct callsites to exercise cap behavior.
  func.func private @drive_a(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_site_cap"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  func.func private @drive_b(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_site_cap"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "site-cap sum = "
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
