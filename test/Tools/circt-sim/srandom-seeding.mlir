// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that srandom() seeding produces reproducible randomization.
// We seed with 42, randomize, capture; seed with 42 again, randomize, compare.

// CHECK: match = 1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  // srandom stub (intercepted by interpreter)
  func.func private @srandom(%seed: i32) {
    return
  }

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Allocate an 8-byte block (two i32 fields: refcount + x)
      %c8 = arith.constant 8 : i64
      %ptr = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr

      // Zero-initialize
      %zero32 = arith.constant 0 : i32
      %f0 = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      llvm.store %zero32, %f0 : i32, !llvm.ptr
      %f1 = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      llvm.store %zero32, %f1 : i32, !llvm.ptr

      // Seed with 42 and randomize -> val1
      %seed42 = arith.constant 42 : i32
      func.call @srandom(%seed42) : (i32) -> ()
      %rc1 = llvm.call @__moore_randomize_basic(%ptr, %c8) : (!llvm.ptr, i64) -> i32
      %val1 = llvm.load %f1 : !llvm.ptr -> i32

      // Re-zero
      llvm.store %zero32, %f1 : i32, !llvm.ptr

      // Seed with 42 again and randomize -> val2
      func.call @srandom(%seed42) : (i32) -> ()
      %rc2 = llvm.call @__moore_randomize_basic(%ptr, %c8) : (!llvm.ptr, i64) -> i32
      %val2 = llvm.load %f1 : !llvm.ptr -> i32

      // Compare
      %eq = arith.cmpi eq, %val1, %val2 : i32
      %match = arith.extui %eq : i1 to i32

      // Print
      %lit = sim.fmt.literal "match = "
      %d = sim.fmt.dec %match signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %d, %nl)
      sim.proc.print %fmt

      sim.terminate success, quiet
      llhd.wait delay %t1, ^start
    }
    hw.output
  }
}
