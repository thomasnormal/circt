// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Regression: compiled call_indirect dispatch must not read past produced
// return values when the call site's function type requests more results than
// the resolved callee actually returns.
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// RUNTIME-NOT: Assertion
// RUNTIME: r0=5 r1=0

func.func @ret1(%a: i32) -> i32 {
  return %a : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"demo::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [[0, @ret1]]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt0 = sim.fmt.literal "r0="
  %fmt1 = sim.fmt.literal " r1="
  %fmtn = sim.fmt.literal "\0A"

  %c5 = hw.constant 5 : i32
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %vt = llvm.mlir.addressof @"demo::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    // Deliberately request 2 results even though @ret1 returns 1.
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> (i32, i32)
    %r0, %r1 = func.call_indirect %fn(%c5) : (i32) -> (i32, i32)
    %d0 = sim.fmt.dec %r0 signed : i32
    %d1 = sim.fmt.dec %r1 signed : i32
    %out = sim.fmt.concat (%fmt0, %d0, %fmt1, %d1, %fmtn)
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
