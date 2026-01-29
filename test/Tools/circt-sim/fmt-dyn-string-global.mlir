// RUN: circt-sim %s | FileCheck %s

// Test that sim.fmt.dyn_string correctly retrieves string content from
// LLVM global variables using the reverse address-to-global lookup.
//
// This tests the fix for UVM string formatting where a {ptr, len} struct
// contains a virtual address that must be mapped back to the global name
// to retrieve the string content.

// CHECK: [circt-sim] Starting simulation
// CHECK: Global string: TestString
// CHECK: [circt-sim] Simulation finished successfully

module {
  // Global string constant
  llvm.mlir.global private constant @test_string("TestString") {addr_space = 0 : i32}

  hw.module @test() {
    %c10000000_i64 = hw.constant 10000000 : i64
    %fmt_prefix = sim.fmt.literal "Global string: "
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Get address of the global string
      %str_ptr = llvm.mlir.addressof @test_string : !llvm.ptr

      // Get length of the string (10 characters)
      %str_len = llvm.mlir.constant(10 : i64) : i64

      // Build the {ptr, len} struct that fmt.dyn_string expects
      %undef_struct = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
      %struct_with_ptr = llvm.insertvalue %str_ptr, %undef_struct[0] : !llvm.struct<(ptr, i64)>
      %str_struct = llvm.insertvalue %str_len, %struct_with_ptr[1] : !llvm.struct<(ptr, i64)>

      // Format the dynamic string using reverse address lookup
      %fmt_str = sim.fmt.dyn_string %str_struct : !llvm.struct<(ptr, i64)>
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
}
