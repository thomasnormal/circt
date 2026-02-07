// RUN: circt-sim %s --top=test_llvm_struct --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM struct operations (getelementptr into structs) in LLHD processes.
// This is essential for class field access.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_struct() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Allocate memory for a struct {i32, i32}
    %c1_i64 = arith.constant 1 : i64
    %ptr = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr

    // Store values into struct fields
    %c10_i32 = arith.constant 10 : i32
    %c20_i32 = arith.constant 20 : i32

    // Get pointer to first field (index 0)
    %field0_ptr = llvm.getelementptr %ptr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
    llvm.store %c10_i32, %field0_ptr : i32, !llvm.ptr

    // Get pointer to second field (index 1)
    %field1_ptr = llvm.getelementptr %ptr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
    llvm.store %c20_i32, %field1_ptr : i32, !llvm.ptr

    // Load from second field
    %loaded = llvm.load %field1_ptr : !llvm.ptr -> i32

    // Drive the signal with the loaded value (should be 20)
    llhd.drv %sig, %loaded after %delta : i32
    llhd.halt
  }

  hw.output
}
