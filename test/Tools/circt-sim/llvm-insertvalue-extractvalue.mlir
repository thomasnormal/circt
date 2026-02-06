// RUN: circt-sim %s | FileCheck %s

// Test that LLVM insertvalue and extractvalue operations are correctly handled
// in the interpreter. These ops are used heavily in UVM string formatting code.

// CHECK: [circt-sim] Starting simulation
// CHECK: Extract test: 42
// CHECK: Extract nested: 100
// CHECK: Insert test: 99
// CHECK: Array extract: 3
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %c10000000_i64 = hw.constant 10000000 : i64
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    // Test basic extractvalue from struct
    %undef_struct = llvm.mlir.undef : !llvm.struct<(i32, i64)>
    %c42_i32 = llvm.mlir.constant(42 : i32) : i32
    %c123_i64 = llvm.mlir.constant(123 : i64) : i64

    // Build a struct with insertvalue
    %struct_1 = llvm.insertvalue %c42_i32, %undef_struct[0] : !llvm.struct<(i32, i64)>
    %struct_2 = llvm.insertvalue %c123_i64, %struct_1[1] : !llvm.struct<(i32, i64)>

    // Extract the first field (should be 42)
    %extracted_0 = llvm.extractvalue %struct_2[0] : !llvm.struct<(i32, i64)>

    // Print extracted value
    %fmt_prefix1 = sim.fmt.literal "Extract test: "
    %fmt_val1 = sim.fmt.dec %extracted_0 : i32
    %fmt_line1 = sim.fmt.concat (%fmt_prefix1, %fmt_val1, %fmt_nl)
    sim.proc.print %fmt_line1

    // Test nested struct extractvalue
    %undef_nested = llvm.mlir.undef : !llvm.struct<(i32, !llvm.struct<(i16, i64)>)>
    %c50_i32 = llvm.mlir.constant(50 : i32) : i32
    %c100_i16 = llvm.mlir.constant(100 : i16) : i16
    %c200_i64 = llvm.mlir.constant(200 : i64) : i64

    // Build inner struct
    %undef_inner = llvm.mlir.undef : !llvm.struct<(i16, i64)>
    %inner_1 = llvm.insertvalue %c100_i16, %undef_inner[0] : !llvm.struct<(i16, i64)>
    %inner_2 = llvm.insertvalue %c200_i64, %inner_1[1] : !llvm.struct<(i16, i64)>

    // Build outer struct
    %nested_1 = llvm.insertvalue %c50_i32, %undef_nested[0] : !llvm.struct<(i32, !llvm.struct<(i16, i64)>)>
    %nested_2 = llvm.insertvalue %inner_2, %nested_1[1] : !llvm.struct<(i32, !llvm.struct<(i16, i64)>)>

    // Extract nested field [1, 0] (should be 100)
    %extracted_nested = llvm.extractvalue %nested_2[1, 0] : !llvm.struct<(i32, !llvm.struct<(i16, i64)>)>

    // Print extracted nested value
    %fmt_prefix2 = sim.fmt.literal "Extract nested: "
    %extracted_nested_i32 = llvm.zext %extracted_nested : i16 to i32
    %fmt_val2 = sim.fmt.dec %extracted_nested_i32 : i32
    %fmt_line2 = sim.fmt.concat (%fmt_prefix2, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_line2

    // Test insertvalue modifying existing field
    %c99_i32 = llvm.mlir.constant(99 : i32) : i32
    %struct_modified = llvm.insertvalue %c99_i32, %struct_2[0] : !llvm.struct<(i32, i64)>
    %extracted_mod = llvm.extractvalue %struct_modified[0] : !llvm.struct<(i32, i64)>

    // Print modified value
    %fmt_prefix3 = sim.fmt.literal "Insert test: "
    %fmt_val3 = sim.fmt.dec %extracted_mod : i32
    %fmt_line3 = sim.fmt.concat (%fmt_prefix3, %fmt_val3, %fmt_nl)
    sim.proc.print %fmt_line3

    // Test array extractvalue
    %undef_array = llvm.mlir.undef : !llvm.array<4 x i8>
    %c1_i8 = llvm.mlir.constant(1 : i8) : i8
    %c2_i8 = llvm.mlir.constant(2 : i8) : i8
    %c3_i8 = llvm.mlir.constant(3 : i8) : i8
    %c4_i8 = llvm.mlir.constant(4 : i8) : i8

    %arr_1 = llvm.insertvalue %c1_i8, %undef_array[0] : !llvm.array<4 x i8>
    %arr_2 = llvm.insertvalue %c2_i8, %arr_1[1] : !llvm.array<4 x i8>
    %arr_3 = llvm.insertvalue %c3_i8, %arr_2[2] : !llvm.array<4 x i8>
    %arr_4 = llvm.insertvalue %c4_i8, %arr_3[3] : !llvm.array<4 x i8>

    // Extract element at index 2 (should be 3)
    %extracted_arr = llvm.extractvalue %arr_4[2] : !llvm.array<4 x i8>

    // Print array element
    %fmt_prefix4 = sim.fmt.literal "Array extract: "
    %extracted_arr_i32 = llvm.zext %extracted_arr : i8 to i32
    %fmt_val4 = sim.fmt.dec %extracted_arr_i32 : i32
    %fmt_line4 = sim.fmt.concat (%fmt_prefix4, %fmt_val4, %fmt_nl)
    sim.proc.print %fmt_line4

    // Delay and terminate
    %delay = llhd.int_to_time %c10000000_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
