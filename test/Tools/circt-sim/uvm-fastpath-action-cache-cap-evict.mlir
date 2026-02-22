// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 CIRCT_SIM_UVM_JIT_FASTPATH_ACTION_CACHE_MAX_ENTRIES=1 CIRCT_SIM_UVM_JIT_FASTPATH_ACTION_CACHE_EVICT_ON_CAP=1 circt-sim %s 2>&1 | FileCheck %s
//
// Verify callee-name action cache cap with eviction.
//
// CHECK: action-evict sum = 0

module {
  // If these execute, they return non-zero; fast-path should zero.
  func.func private @"uvm_pkg::custom_printer::uvm_printer::print_field_int_action_evict_a"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c33 = arith.constant 33 : i32
    return %c33 : i32
  }

  func.func private @"uvm_pkg::custom_printer::uvm_printer::print_field_int_action_evict_b"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c44 = arith.constant 44 : i32
    return %c44 : i32
  }

  func.func private @drive_a(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_action_evict_a"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  func.func private @drive_b1(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_action_evict_b"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  func.func private @drive_b2(%self: !llvm.ptr, %value: i32) -> i32 {
    %v = func.call @"uvm_pkg::custom_printer::uvm_printer::print_field_int_action_evict_b"(%self, %value) :
        (!llvm.ptr, i32) -> i32
    return %v : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "action-evict sum = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %value = arith.constant 9 : i32

      %a = func.call @drive_a(%self, %value) : (!llvm.ptr, i32) -> i32
      %b1 = func.call @drive_b1(%self, %value) : (!llvm.ptr, i32) -> i32
      %b2 = func.call @drive_b2(%self, %value) : (!llvm.ptr, i32) -> i32
      %ab = arith.addi %a, %b1 : i32
      %sum = arith.addi %ab, %b2 : i32

      %sumFmt = sim.fmt.dec %sum signed : i32
      %line = sim.fmt.concat (%fmtPrefix, %sumFmt, %fmtNl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
