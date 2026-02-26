// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so | FileCheck %s --check-prefix=COMPILED

// Regression: tagged call_indirect lowering must not call null entry-table
// slots. Missing vtable symbols are represented as null entries in
// @__circt_sim_func_entries and should fail safely instead of crashing.
//
// COMPILE: [circt-sim-compile] LowerTaggedIndirectCalls: lowered 1 indirect calls
// SIM: dyn=x
// COMPILED: dyn=0

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

func.func @invoke_dyn(%sel: i1, %x: i32) -> i32 {
  %vtable = llvm.mlir.addressof @"math::__vtable__" : !llvm.ptr
  %idx = arith.extui %sel : i1 to i64
  %slot = llvm.getelementptr %vtable[0, %idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x ptr>
  %fptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
  %r = func.call @caller_indirect(%fptr, %x) : (!llvm.ptr, i32) -> i32
  return %r : i32
}

llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::add42"],
    [1, @"math::missing"]
  ]
} : !llvm.array<2 x ptr>

hw.module @test() {
  %fmt = sim.fmt.literal "dyn="
  %nl = sim.fmt.literal "\0A"
  %c1 = hw.constant 1 : i1
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64
  %sel_sig = llhd.sig %c1 : i1

  llhd.process {
    %sel = llhd.prb %sel_sig : i1
    %r = func.call @invoke_dyn(%sel, %c5) : (i1, i32) -> i32
    %fv = sim.fmt.dec %r signed : i32
    %out = sim.fmt.concat (%fmt, %fv, %nl)
    sim.proc.print %out
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
