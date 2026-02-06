// RUN: circt-sim %s | FileCheck %s

// Test that LLVM struct types (e.g., !llvm.struct<(ptr, i64)>) are correctly
// handled during load/store operations in the interpreter.
//
// This tests the fix for a crash that occurred when loading or storing
// values wider than 64 bits. The getTypeWidth function was not handling
// LLVM struct types, returning 1 instead of the correct width (128 for
// struct<(ptr, i64)>), causing an APInt assertion failure.

// CHECK: [circt-sim] Starting simulation
// CHECK: String value: Hello
// CHECK: [circt-sim] Simulation completed

llvm.func @__moore_packed_string_to_string(i64) -> !llvm.struct<(ptr, i64)>

hw.module @test() {
  // Constant for the packed string "Hello" (0x48656C6C6F)
  %c310939249775_i64 = hw.constant 310939249775 : i64
  %c10000000_i64 = hw.constant 10000000 : i64

  %fmt_prefix = sim.fmt.literal "String value: "
  %fmt_nl = sim.fmt.literal "\0A"
  %undef_struct = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %c0_i64 = llvm.mlir.constant(0 : i64) : i64
  %null_ptr = llvm.mlir.zero : !llvm.ptr
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64

  llhd.process {
    // Call the string conversion function
    %str_result = llvm.call @__moore_packed_string_to_string(%c310939249775_i64) : (i64) -> !llvm.struct<(ptr, i64)>

    // Allocate memory for the struct
    %mem = llvm.alloca %c1_i64 x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr

    // Initialize with zeros (tests struct store with wide value)
    %empty_struct_0 = llvm.insertvalue %null_ptr, %undef_struct[0] : !llvm.struct<(ptr, i64)>
    %empty_struct = llvm.insertvalue %c0_i64, %empty_struct_0[1] : !llvm.struct<(ptr, i64)>
    llvm.store %empty_struct, %mem : !llvm.struct<(ptr, i64)>, !llvm.ptr

    // Store the actual string result (tests wide struct store)
    llvm.store %str_result, %mem : !llvm.struct<(ptr, i64)>, !llvm.ptr

    // Load the value back (tests wide struct load - this would crash before fix)
    %loaded = llvm.load %mem : !llvm.ptr -> !llvm.struct<(ptr, i64)>

    // Format and print the string
    %fmt_str = sim.fmt.dyn_string %loaded : !llvm.struct<(ptr, i64)>
    %fmt_full = sim.fmt.concat (%fmt_prefix, %fmt_str, %fmt_nl)
    sim.proc.print %fmt_full

    // Delay and terminate
    %delay = llhd.int_to_time %c10000000_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
