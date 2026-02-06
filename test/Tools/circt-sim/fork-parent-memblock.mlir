// RUN: circt-sim %s | FileCheck %s

// Test that fork/join child processes can read parent-scope alloca memory
// via findMemoryBlockByAddress's parent-chain walk.
// This previously crashed with a DenseMap iterator invalidation assertion
// when comparing iterators from different DenseMaps (child vs parent valueMap).

// CHECK: result=42

hw.module @test() {
  llhd.process {
    // Parent-scope alloca (shared with forked children)
    %one = llvm.mlir.constant(1 : i64) : i64
    %c42 = hw.constant 42 : i32
    %alloca = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    llvm.store %c42, %alloca : i32, !llvm.ptr

    // Fork a child process that reads the parent's alloca
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %result = llvm.load %alloca : !llvm.ptr -> i32

    %fmt_prefix = sim.fmt.literal "result="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %result : i32
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
