// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that __moore_randomize_with_ranges returns values within the given ranges.
//
// We set up two ranges: [10, 20] and [100, 110].
// The returned value must be in one of these ranges.
// We call 3 times; at least one call should return a value in range.
// (Probability of all 3 falling outside both ranges if the function works
// correctly is 0, since it should always return a value in range.)

// CHECK: result_in_range_0 = 1
// CHECK: result_in_range_1 = 1
// CHECK: result_in_range_2 = 1
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @__moore_randomize_with_ranges(!llvm.ptr, i64) -> i64

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Allocate array for 2 range pairs = 4 x i64 = 32 bytes
      %c32 = arith.constant 32 : i64
      %ptr = llvm.call @malloc(%c32) : (i64) -> !llvm.ptr

      // Store range 0: [10, 20]
      %c10 = arith.constant 10 : i64
      %c20 = arith.constant 20 : i64
      // Store range 1: [100, 110]
      %c100 = arith.constant 100 : i64
      %c110 = arith.constant 110 : i64

      // ranges[0] = 10 (low of range 0)
      %p0 = llvm.getelementptr %ptr[0]
          : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %c10, %p0 : i64, !llvm.ptr

      // ranges[1] = 20 (high of range 0)
      %p1 = llvm.getelementptr %ptr[1]
          : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %c20, %p1 : i64, !llvm.ptr

      // ranges[2] = 100 (low of range 1)
      %p2 = llvm.getelementptr %ptr[2]
          : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %c100, %p2 : i64, !llvm.ptr

      // ranges[3] = 110 (high of range 1)
      %p3 = llvm.getelementptr %ptr[3]
          : (!llvm.ptr) -> !llvm.ptr, i64
      llvm.store %c110, %p3 : i64, !llvm.ptr

      %c2 = arith.constant 2 : i64

      // --- Call 0 ---
      %val0 = llvm.call @__moore_randomize_with_ranges(%ptr, %c2)
          : (!llvm.ptr, i64) -> i64

      // Check: 10 <= val0 <= 20 OR 100 <= val0 <= 110
      %ge10_0 = arith.cmpi sge, %val0, %c10 : i64
      %le20_0 = arith.cmpi sle, %val0, %c20 : i64
      %in_r0_0 = arith.andi %ge10_0, %le20_0 : i1
      %ge100_0 = arith.cmpi sge, %val0, %c100 : i64
      %le110_0 = arith.cmpi sle, %val0, %c110 : i64
      %in_r1_0 = arith.andi %ge100_0, %le110_0 : i1
      %in_range_0 = arith.ori %in_r0_0, %in_r1_0 : i1

      %one32 = arith.constant 1 : i32
      %zero32 = arith.constant 0 : i32
      %res0 = arith.select %in_range_0, %one32, %zero32 : i32

      %lit0 = sim.fmt.literal "result_in_range_0 = "
      %d0 = sim.fmt.dec %res0 signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt0 = sim.fmt.concat (%lit0, %d0, %nl)
      sim.proc.print %fmt0

      // --- Call 1 ---
      %val1 = llvm.call @__moore_randomize_with_ranges(%ptr, %c2)
          : (!llvm.ptr, i64) -> i64

      %ge10_1 = arith.cmpi sge, %val1, %c10 : i64
      %le20_1 = arith.cmpi sle, %val1, %c20 : i64
      %in_r0_1 = arith.andi %ge10_1, %le20_1 : i1
      %ge100_1 = arith.cmpi sge, %val1, %c100 : i64
      %le110_1 = arith.cmpi sle, %val1, %c110 : i64
      %in_r1_1 = arith.andi %ge100_1, %le110_1 : i1
      %in_range_1 = arith.ori %in_r0_1, %in_r1_1 : i1

      %res1 = arith.select %in_range_1, %one32, %zero32 : i32

      %lit1 = sim.fmt.literal "result_in_range_1 = "
      %d1 = sim.fmt.dec %res1 signed : i32
      %fmt1 = sim.fmt.concat (%lit1, %d1, %nl)
      sim.proc.print %fmt1

      // --- Call 2 ---
      %val2 = llvm.call @__moore_randomize_with_ranges(%ptr, %c2)
          : (!llvm.ptr, i64) -> i64

      %ge10_2 = arith.cmpi sge, %val2, %c10 : i64
      %le20_2 = arith.cmpi sle, %val2, %c20 : i64
      %in_r0_2 = arith.andi %ge10_2, %le20_2 : i1
      %ge100_2 = arith.cmpi sge, %val2, %c100 : i64
      %le110_2 = arith.cmpi sle, %val2, %c110 : i64
      %in_r1_2 = arith.andi %ge100_2, %le110_2 : i1
      %in_range_2 = arith.ori %in_r0_2, %in_r1_2 : i1

      %res2 = arith.select %in_range_2, %one32, %zero32 : i32

      %lit2 = sim.fmt.literal "result_in_range_2 = "
      %d2 = sim.fmt.dec %res2 signed : i32
      %fmt2 = sim.fmt.concat (%lit2, %d2, %nl)
      sim.proc.print %fmt2

      llvm.call @free(%ptr) : (!llvm.ptr) -> ()

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
