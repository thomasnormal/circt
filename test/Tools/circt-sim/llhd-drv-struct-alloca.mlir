// RUN: circt-sim %s | FileCheck %s

// Test that llhd.drv correctly handles driving to a struct field when the
// parent signal is backed by an llvm.alloca (memory-backed ref).
//
// This tests the fix for a bug where driving to llhd.sig.struct_extract of
// an alloca-backed !llhd.ref would not work correctly. The interpreter now:
// 1. Reads the current struct value from memory
// 2. Inserts the new field value at the correct bit offset
// 3. Writes the modified value back to memory
//
// Pattern tested:
//   %alloca = llvm.alloca -> unrealized_cast to !llhd.ref<struct>
//   %field_ref = llhd.sig.struct_extract %ref["field"]
//   llhd.drv %field_ref, %value

// CHECK: [circt-sim] Starting simulation
// CHECK: Initial a: 0, b: 0
// CHECK: After driving a=42: a: 42, b: 0
// CHECK: After driving b=99: a: 42, b: 99
// CHECK: After driving a=7: a: 7, b: 99
// CHECK: [circt-sim] Simulation finished successfully

hw.module @test() {
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c42_i32 = llvm.mlir.constant(42 : i32) : i32
  %c99_i32 = llvm.mlir.constant(99 : i32) : i32
  %c7_i32 = llvm.mlir.constant(7 : i32) : i32

  %fmt_init = sim.fmt.literal "Initial a: "
  %fmt_after_a = sim.fmt.literal "After driving a=42: a: "
  %fmt_after_b = sim.fmt.literal "After driving b=99: a: "
  %fmt_after_a2 = sim.fmt.literal "After driving a=7: a: "
  %fmt_comma_b = sim.fmt.literal ", b: "
  %fmt_nl = sim.fmt.literal "\0A"

  // LLHD time constant for drives
  %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>

  llhd.process {
    // Allocate a struct with two i32 fields: {a: i32, b: i32}
    // In LLVM, this is struct<(i32, i32)> - total 64 bits
    %local_struct = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr

    // Initialize the struct to zeros using llvm.store
    %undef = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %s0 = llvm.insertvalue %c0_i32, %undef[0] : !llvm.struct<(i32, i32)>
    %s1 = llvm.insertvalue %c0_i32, %s0[1] : !llvm.struct<(i32, i32)>
    llvm.store %s1, %local_struct : !llvm.struct<(i32, i32)>, !llvm.ptr

    // Cast the alloca result to !llhd.ref<!hw.struct<...>>
    %ref = builtin.unrealized_conversion_cast %local_struct : !llvm.ptr to !llhd.ref<!hw.struct<a: i32, b: i32>>

    // Test 1: Probe initial values (should be 0, 0)
    %val0 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
    %a0 = hw.struct_extract %val0["a"] : !hw.struct<a: i32, b: i32>
    %b0 = hw.struct_extract %val0["b"] : !hw.struct<a: i32, b: i32>
    %fmt_a0 = sim.fmt.dec %a0 : i32
    %fmt_b0 = sim.fmt.dec %b0 : i32
    %fmt_out0 = sim.fmt.concat (%fmt_init, %fmt_a0, %fmt_comma_b, %fmt_b0, %fmt_nl)
    sim.proc.print %fmt_out0

    // Test 2: Drive field "a" to 42 using llhd.sig.struct_extract
    %a_ref = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: i32, b: i32>>
    llhd.drv %a_ref, %c42_i32 after %time0 : i32

    // Probe to verify: should be {a: 42, b: 0}
    %val1 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
    %a1 = hw.struct_extract %val1["a"] : !hw.struct<a: i32, b: i32>
    %b1 = hw.struct_extract %val1["b"] : !hw.struct<a: i32, b: i32>
    %fmt_a1 = sim.fmt.dec %a1 : i32
    %fmt_b1 = sim.fmt.dec %b1 : i32
    %fmt_out1 = sim.fmt.concat (%fmt_after_a, %fmt_a1, %fmt_comma_b, %fmt_b1, %fmt_nl)
    sim.proc.print %fmt_out1

    // Test 3: Drive field "b" to 99, "a" should remain 42
    %b_ref = llhd.sig.struct_extract %ref["b"] : <!hw.struct<a: i32, b: i32>>
    llhd.drv %b_ref, %c99_i32 after %time0 : i32

    // Probe to verify: should be {a: 42, b: 99}
    %val2 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
    %a2 = hw.struct_extract %val2["a"] : !hw.struct<a: i32, b: i32>
    %b2 = hw.struct_extract %val2["b"] : !hw.struct<a: i32, b: i32>
    %fmt_a2 = sim.fmt.dec %a2 : i32
    %fmt_b2 = sim.fmt.dec %b2 : i32
    %fmt_out2 = sim.fmt.concat (%fmt_after_b, %fmt_a2, %fmt_comma_b, %fmt_b2, %fmt_nl)
    sim.proc.print %fmt_out2

    // Test 4: Drive field "a" again to 7, "b" should remain 99
    llhd.drv %a_ref, %c7_i32 after %time0 : i32

    // Probe to verify: should be {a: 7, b: 99}
    %val3 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
    %a3 = hw.struct_extract %val3["a"] : !hw.struct<a: i32, b: i32>
    %b3 = hw.struct_extract %val3["b"] : !hw.struct<a: i32, b: i32>
    %fmt_a3 = sim.fmt.dec %a3 : i32
    %fmt_b3 = sim.fmt.dec %b3 : i32
    %fmt_out3 = sim.fmt.concat (%fmt_after_a2, %fmt_a3, %fmt_comma_b, %fmt_b3, %fmt_nl)
    sim.proc.print %fmt_out3

    // Terminate successfully
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
