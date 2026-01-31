// RUN: circt-sim %s | FileCheck %s

// Test that llhd.prb and llhd.drv work correctly when the target is an
// llvm.alloca operation (local variable in a function) cast to !llhd.ref.
//
// This tests the fix for a crash that occurred when llhd.prb or llhd.drv
// operations received an UnrealizedConversionCastOp whose input was an
// llvm.alloca. The interpreter now detects this pattern and reads/writes
// directly from the memory block backing the alloca.
//
// Pattern tested:
//   %alloca = llvm.alloca -> unrealized_cast to !llhd.ref -> llhd.prb/llhd.drv

// CHECK: [circt-sim] Starting simulation
// CHECK: Initial value: 0
// CHECK: After drive: 42
// CHECK: After second drive: 99
// CHECK: [circt-sim] Simulation finished successfully

hw.module @test() {
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c42_i32 = llvm.mlir.constant(42 : i32) : i32
  %c99_i32 = llvm.mlir.constant(99 : i32) : i32
  %c10000000_i64 = hw.constant 10000000 : i64

  %fmt_init = sim.fmt.literal "Initial value: "
  %fmt_after1 = sim.fmt.literal "After drive: "
  %fmt_after2 = sim.fmt.literal "After second drive: "
  %fmt_nl = sim.fmt.literal "\0A"

  // LLHD time constants for drives
  %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>

  llhd.process {
    // Allocate a local variable (i32)
    %local_var = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr

    // Initialize with zero using llvm.store
    llvm.store %c0_i32, %local_var : i32, !llvm.ptr

    // Cast the alloca result to !llhd.ref<i32>
    // This is the pattern that occurs when local variables in class methods
    // are accessed via LLHD operations
    %ref = builtin.unrealized_conversion_cast %local_var : !llvm.ptr to !llhd.ref<i32>

    // Test 1: Probe the initial value (should be 0)
    %val0 = llhd.prb %ref : i32
    %fmt_val0 = sim.fmt.dec %val0 : i32
    %fmt_out0 = sim.fmt.concat (%fmt_init, %fmt_val0, %fmt_nl)
    sim.proc.print %fmt_out0

    // Test 2: Drive a new value (42) and probe it back
    llhd.drv %ref, %c42_i32 after %time0 : i32
    %val1 = llhd.prb %ref : i32
    %fmt_val1 = sim.fmt.dec %val1 : i32
    %fmt_out1 = sim.fmt.concat (%fmt_after1, %fmt_val1, %fmt_nl)
    sim.proc.print %fmt_out1

    // Test 3: Drive another value (99) and verify it overwrites
    llhd.drv %ref, %c99_i32 after %time0 : i32
    %val2 = llhd.prb %ref : i32
    %fmt_val2 = sim.fmt.dec %val2 : i32
    %fmt_out2 = sim.fmt.concat (%fmt_after2, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_out2

    // Terminate successfully
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
