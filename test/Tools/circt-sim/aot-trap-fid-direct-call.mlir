// RUN: circt-compile %s -o %t.so
// RUN: circt-sim %s --compiled=%t.so | FileCheck %s --check-prefix=OK
// RUN: not --crash env CIRCT_AOT_TRAP_FID=0 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=TRAP

// OK: out=47
// TRAP: [AOT TRAP] func.call fid=0 name=math::add42

func.func private @"math::add42"(%x: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %r = arith.addi %x, %c42 : i32
  return %r : i32
}

// Name chosen to match default demotion policy (interpreted function body).
func.func @"dummy_seq::body"(%x: i32) -> i32 {
  %r = func.call @"math::add42"(%x) : (i32) -> i32
  return %r : i32
}

llvm.mlir.global internal @"math::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"math::add42"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %r = func.call @"dummy_seq::body"(%c5) : (i32) -> i32
    %fmtV = sim.fmt.dec %r signed : i32
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %fmtOut
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
