// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Regression: interpreted call_indirect fallback must not reuse stale block-arg
// values when a later call omits arguments via a mismatched function cast.
//
// The i65 signature forces call_indirect to skip native entry dispatch and
// execute the callee in interpreter fallback.
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// RUNTIME-NOT: r=12
// RUNTIME: r=5

func.func @sum2_i65(%a: i65, %b: i65) -> i65 {
  %sum = arith.addi %a, %b : i65
  return %sum : i65
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

llvm.mlir.global internal @"stale::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [[0, @sum2_i65]]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %fmt = sim.fmt.literal "r="
  %nl = sim.fmt.literal "\0A"

  %a10 = hw.constant 10 : i65
  %b7 = hw.constant 7 : i65
  %a5 = hw.constant 5 : i65
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %vt = llvm.mlir.addressof @"stale::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr

    // Warm call with both args.
    %fn2 = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i65, i65) -> i65
    %warm = func.call_indirect %fn2(%a10, %b7) : (i65, i65) -> i65

    // Mismatched call shape: second argument omitted.
    %fn1 = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i65) -> i65
    %r = func.call_indirect %fn1(%a5) : (i65) -> i65

    %d = sim.fmt.dec %r signed : i65
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
