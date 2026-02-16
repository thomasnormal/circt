// RUN: circt-sim %s | FileCheck %s

// Verify targeted UVM printer call_indirect fast-paths:
// 1) adjust_name returns the incoming name argument (passthrough)
// 2) print_field_int bypasses callee execution (zeroed result)

module {
  // If this executes, it returns 99. The fast-path must return %id instead.
  func.func private @"uvm_pkg::uvm_printer::adjust_name"(
      %self: !llvm.ptr, %id: i32, %scope_separator: i8) -> i32 {
    %c99 = arith.constant 99 : i32
    return %c99 : i32
  }

  // If this executes, it returns 123. The fast-path must suppress it.
  func.func private @"uvm_pkg::uvm_printer::print_field_int"(
      %self: !llvm.ptr, %value: i32) -> i32 {
    %c123 = arith.constant 123 : i32
    return %c123 : i32
  }

  llvm.mlir.global internal @"uvm_printer_fastpath::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [
      [0, @"uvm_pkg::uvm_printer::adjust_name"],
      [1, @"uvm_pkg::uvm_printer::print_field_int"]
    ]
  } : !llvm.array<2 x ptr>

  hw.module @main() {
    %fmtAdjust = sim.fmt.literal "adjust_name result = "
    %fmtField = sim.fmt.literal "print_field_int result = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %zero64 = arith.constant 0 : i64
      %self = llvm.inttoptr %zero64 : i64 to !llvm.ptr
      %id = arith.constant 7 : i32
      %scopeSep = arith.constant 46 : i8

      %vtableAddr = llvm.mlir.addressof @"uvm_printer_fastpath::__vtable__" :
          !llvm.ptr

      %slot0 = llvm.getelementptr %vtableAddr[0, 0] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn0 = builtin.unrealized_conversion_cast %fptr0 :
          !llvm.ptr to (!llvm.ptr, i32, i8) -> i32
      %adjustResult = func.call_indirect %fn0(%self, %id, %scopeSep) :
          (!llvm.ptr, i32, i8) -> i32

      %slot1 = llvm.getelementptr %vtableAddr[0, 1] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fptr1 = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
      %fn1 = builtin.unrealized_conversion_cast %fptr1 :
          !llvm.ptr to (!llvm.ptr, i32) -> i32
      %fieldResult = func.call_indirect %fn1(%self, %id) :
          (!llvm.ptr, i32) -> i32

      %adjustDec = sim.fmt.dec %adjustResult signed : i32
      %adjustLine = sim.fmt.concat (%fmtAdjust, %adjustDec, %fmtNl)
      sim.proc.print %adjustLine

      %fieldDec = sim.fmt.dec %fieldResult signed : i32
      %fieldLine = sim.fmt.concat (%fmtField, %fieldDec, %fmtNl)
      sim.proc.print %fieldLine

      llhd.halt
    }
    hw.output
  }
}

// CHECK: adjust_name result = 7
// CHECK: print_field_int result = 0
