// RUN: circt-sim %s | FileCheck %s
//
// Verify targeted UVM printer func.call fast-path:
// print_field_int bypasses callee execution and zeroes the result.

module {
  // If this executes, it returns 123. The fast-path must suppress it.
  func.func private @"uvm_pkg::uvm_printer::print_field_int"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  hw.module @main() {
    %fmtField = sim.fmt.literal "print_field_int direct result = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %value = arith.constant 7 : i32

      %result = func.call @"uvm_pkg::uvm_printer::print_field_int"(%self, %value) :
          (!llvm.ptr, i32) -> i32

      %fieldDec = sim.fmt.dec %result signed : i32
      %fieldLine = sim.fmt.concat (%fmtField, %fieldDec, %fmtNl)
      sim.proc.print %fieldLine

      llhd.halt
    }
    hw.output
  }
}

// CHECK: print_field_int direct result = 0
