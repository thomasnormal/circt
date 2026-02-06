// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that __moore_randomize_basic actually fills memory with random bytes
// instead of leaving it zeroed (the old stub behavior).
//
// We allocate a 16-byte block, zero it, call __moore_randomize_basic,
// then check that at least one of the 4-byte words is non-zero.
// The probability of all 16 random bytes being zero is (1/256)^16 ≈ 0.

// CHECK: randomize_returned = 1
// CHECK: at_least_one_nonzero = 1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Allocate 16 bytes
      %c16 = arith.constant 16 : i64
      %ptr = llvm.call @malloc(%c16) : (i64) -> !llvm.ptr

      // Zero-initialize all 16 bytes
      %zero32 = arith.constant 0 : i32
      %f0 = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %zero32, %f0 : i32, !llvm.ptr
      %f1 = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %zero32, %f1 : i32, !llvm.ptr
      %f2 = llvm.getelementptr %ptr[0, 2]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %zero32, %f2 : i32, !llvm.ptr
      %f3 = llvm.getelementptr %ptr[0, 3]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32)>
      llvm.store %zero32, %f3 : i32, !llvm.ptr

      // Call __moore_randomize_basic
      %rc = llvm.call @__moore_randomize_basic(%ptr, %c16) : (!llvm.ptr, i64) -> i32

      // Print return code
      %lit_rc = sim.fmt.literal "randomize_returned = "
      %d_rc = sim.fmt.dec %rc signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt_rc = sim.fmt.concat (%lit_rc, %d_rc, %nl)
      sim.proc.print %fmt_rc

      // Load all 4 words and OR them together - at least one should be nonzero
      %v0 = llvm.load %f0 : !llvm.ptr -> i32
      %v1 = llvm.load %f1 : !llvm.ptr -> i32
      %v2 = llvm.load %f2 : !llvm.ptr -> i32
      %v3 = llvm.load %f3 : !llvm.ptr -> i32
      %or01 = comb.or %v0, %v1 : i32
      %or23 = comb.or %v2, %v3 : i32
      %or_all = comb.or %or01, %or23 : i32

      // Check if nonzero
      %zero = hw.constant 0 : i32
      %c_true = hw.constant 1 : i1
      %is_nonzero = comb.icmp ne %or_all, %zero : i32
      %nz_i32 = comb.concat %c_true : i1
      // Use select to produce 1 if any word is nonzero
      %one32 = arith.constant 1 : i32
      %result = arith.select %is_nonzero, %one32, %zero32 : i32

      %lit_nz = sim.fmt.literal "at_least_one_nonzero = "
      %d_nz = sim.fmt.dec %result signed : i32
      %fmt_nz = sim.fmt.concat (%lit_nz, %d_nz, %nl)
      sim.proc.print %fmt_nz

      llvm.call @free(%ptr) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
