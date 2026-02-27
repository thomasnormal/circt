// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so | FileCheck %s --check-prefix=COMPILED

// Regression: invalid tagged FuncIds in lowered call_indirect paths must not
// dispatch through out-of-bounds entry-table slots.
//
// COMPILE: [circt-compile] LowerTaggedIndirectCalls: lowered 1 indirect calls
// SIM: bad=0
// COMPILED: bad=0

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

func.func @invoke_bad_tag(%x: i32) -> i32 {
  %bad_i64 = hw.constant 4026531970 : i64 // 0xF0000082
  %bad = llvm.inttoptr %bad_i64 : i64 to !llvm.ptr
  %r = func.call @caller_indirect(%bad, %x) : (!llvm.ptr, i32) -> i32
  return %r : i32
}

// Keep a tagged vtable in the module so this test exercises the same tagged
// pointer lowering path used by vtable-dispatched calls.
llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::add42"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "bad="
  %fmt_nl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @invoke_bad_tag(%c5) : (i32) -> i32
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
