// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Regression: compiled call_indirect dispatch must not read past provided
// argument values when the resolved callee expects more args than the call-site
// function type provides.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// RUNTIME-NOT: Assertion
// RUNTIME: r=5

func.func @add2(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"demo2::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [[0, @add2]]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt = sim.fmt.literal "r="
  %nl = sim.fmt.literal "\0A"

  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %vt = llvm.mlir.addressof @"demo2::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    // Deliberately pass 1 arg while @add2 expects 2.
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
    %r = func.call_indirect %fn(%c5) : (i32) -> i32
    %d = sim.fmt.dec %r signed : i32
    %out = sim.fmt.concat (%fmt, %d, %nl)
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
