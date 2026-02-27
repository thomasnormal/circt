// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: count non-native tagged entry-table hits.
//
// We force a vtable target to be non-native by compiling with
// CIRCT_AOT_INTERCEPT_ALL_UVM=1. The process performs func.call_indirect
// through a tagged vtable slot; runtime dispatches via the entry table to a
// trampoline callback into the interpreter.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// COMPILED: Loaded 1 compiled functions: 1 native-dispatched, 0 not-native-dispatched, 0 intercepted
// COMPILED: Entry table: 1 entries for tagged-FuncId dispatch (0 native, 1 non-native)
// COMPILED: Trampoline calls:                 1
// COMPILED: Entry-table native calls:         0
// COMPILED: Entry-table trampoline calls:     1
// COMPILED: Hot uncompiled FuncIds (top 50):
// COMPILED: [circt-sim]{{[[:space:]]+}}1x fid=0 uvm_pkg::uvm_demo::add42
// COMPILED: indirect_uvm(5) = 47

func.func @"uvm_pkg::uvm_demo::add42"(%a: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %a, %c42 : i32
  return %r : i32
}

// Keep one non-intercepted compiled function alive so AOT emission succeeds
// even when the vtable target above is demoted to a trampoline.
func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_demo::add42"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "indirect_uvm(5) = "
  %fmt_nl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
    %r = func.call_indirect %fn(%c5) : (i32) -> i32
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
