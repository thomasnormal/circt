// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// Test that recursive function calls preserve outer call's SSA values.
// This is a regression test for a bug where the inner call's intermediate
// SSA values (GEPs, loads) would overwrite the outer call's values in the
// shared valueMap, causing the outer call to use wrong data after return.

// CHECK: RESULT=42
// CHECK: [circt-sim] Simulation completed

module {
  // Recursive function: at depth 0 it stores 42 into an alloca, recurses,
  // then reads back from the alloca. At depth 1 it stores 99 into its own
  // alloca and returns. Without SSA value preservation, the outer call's
  // alloca pointer gets overwritten by the inner call's alloca pointer.
  func.func @recursive_store_read(%depth: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c42 = arith.constant 42 : i32
    %c99 = arith.constant 99 : i32
    %one = arith.constant 1 : i64

    // Each call level allocates its own storage
    %ptr = llvm.alloca %one x i32 : (i64) -> !llvm.ptr

    // If depth == 0, store 42; if depth > 0, store 99
    %is_zero = arith.cmpi eq, %depth, %c0 : i32
    cf.cond_br %is_zero, ^depth0, ^depth_nonzero

  ^depth0:
    llvm.store %c42, %ptr : i32, !llvm.ptr
    // Recurse with depth=1
    %inner_result = func.call @recursive_store_read(%c1) : (i32) -> i32
    // Read back from OUR alloca - should still be 42, not 99
    %val = llvm.load %ptr : !llvm.ptr -> i32
    return %val : i32

  ^depth_nonzero:
    llvm.store %c99, %ptr : i32, !llvm.ptr
    %val2 = llvm.load %ptr : !llvm.ptr -> i32
    return %val2 : i32
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %c0 = arith.constant 0 : i32
      %result = func.call @recursive_store_read(%c0) : (i32) -> i32

      %lit = sim.fmt.literal "RESULT="
      %val = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %val, %nl)
      sim.proc.print %fmt

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
