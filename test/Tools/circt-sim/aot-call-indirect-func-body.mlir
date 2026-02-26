// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: keep func.func bodies with func.call_indirect compilable.
//
// COMPILE: [circt-sim-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-sim-compile] 3 functions + 0 processes ready for codegen
// COMPILE: [circt-sim-compile] Initialized 1 vtable globals with tagged FuncIds
//
// SIM: caller_indirect(5) = 47
//
// COMPILED: Loaded 3 compiled functions: 3 native-dispatched, 0 not-native-dispatched, 0 intercepted
// COMPILED: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// COMPILED: Entry-table native calls:         0
// COMPILED: Entry-table trampoline calls:     0
// COMPILED: caller_indirect(5) = 47

func.func private @"math::add42"(%a: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %a, %c42 : i32
  return %r : i32
}

func.func @caller_indirect(%fptr: !llvm.ptr, %x: i32) -> i32 {
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
  %r = func.call_indirect %fn(%x) : (i32) -> i32
  return %r : i32
}

func.func @invoke_from_vtable(%x: i32) -> i32 {
  %vtable = llvm.mlir.addressof @"math::__vtable__" : !llvm.ptr
  %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
  %r = func.call @caller_indirect(%fptr, %x) : (!llvm.ptr, i32) -> i32
  return %r : i32
}

llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::add42"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "caller_indirect(5) = "
  %fmt_nl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @invoke_from_vtable(%c5) : (i32) -> i32
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
