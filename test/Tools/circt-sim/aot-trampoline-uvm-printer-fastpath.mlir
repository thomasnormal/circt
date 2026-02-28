// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=RUNTIME
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 CIRCT_AOT_DISABLE_ALL=1 circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=DISABLEALL

// Regression: call_indirect from native compiled code can route to a
// non-native trampoline. That trampoline must preserve UVM printer/report
// fast-path semantics (no-op with zero results), instead of executing the
// interpreted body directly.
//
// COMPILE: [circt-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] Generated 1 interpreter trampolines
//
// RUNTIME: Trampoline calls:                 1
// RUNTIME: indirect_result=0
// RUNTIME-NOT: BAD_TRAMP
//
// DISABLEALL: [circt-sim] CIRCT_AOT_DISABLE_ALL: all native dispatch disabled
// DISABLEALL: indirect_result=0
// DISABLEALL-NOT: BAD_TRAMP

func.func @driver_call_indirect(%a: i32) -> i32 {
  %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
  %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
  %r = func.call_indirect %fn(%a) : (i32) -> i32
  return %r : i32
}

func.func @"uvm_pkg::uvm_printer::print_string"(%x: i32) -> i32 {
  %c100 = arith.constant 100 : i32
  %sum = arith.addi %x, %c100 : i32
  %fmt_bad = sim.fmt.literal "BAD_TRAMP\0A"
  sim.proc.print %fmt_bad
  return %sum : i32
}

// Keep one non-intercepted helper alive so AOT codegen still emits native code.
func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_printer::print_string"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "indirect_result="
  %fmt_nl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @driver_call_indirect(%c5) : (i32) -> i32
    %fmt_v = sim.fmt.dec %r signed : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_v, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
