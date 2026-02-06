// RUN: circt-sim %s --top test | FileCheck %s
// Test that the findMemoryBlockByAddress fallback works for llvm.store and
// llvm.load when the SSA-based findMemoryBlock fails.
//
// This simulates the real-world scenario where a pointer to an object is
// stored into a "box" (another alloca), then loaded back and used for
// field access. Loading the pointer from memory breaks the SSA chain
// because the pointer no longer comes directly from an alloca - the
// interpreter must fall back to address-based memory lookup.
//
// Object layout: !llvm.struct<"BoxTest", (i32, i32)>
// Box: !llvm.ptr alloca holding the struct address

// CHECK: field0_via_box = 42
// CHECK: field1_via_box = 99
// CHECK: field0_after_store_via_box = 77
// CHECK: [circt-sim] Simulation completed

module {
  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %one = arith.constant 1 : i64

      // Allocate the struct object
      %obj = llvm.alloca %one x !llvm.struct<"BoxTest", (i32, i32)>
          : (i64) -> !llvm.ptr

      // Allocate the "box" - a pointer-sized slot to hold the struct address
      %box = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr

      // Store the struct address into the box
      llvm.store %obj, %box : !llvm.ptr, !llvm.ptr

      // Store values into the struct fields via direct GEP (normal SSA tracing)
      %c42 = arith.constant 42 : i32
      %c99 = arith.constant 99 : i32
      %field0_direct = llvm.getelementptr %obj[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      llvm.store %c42, %field0_direct : i32, !llvm.ptr
      %field1_direct = llvm.getelementptr %obj[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      llvm.store %c99, %field1_direct : i32, !llvm.ptr

      // --- Test 1: Load through box, read field 0 ---
      // Load the struct address FROM the box (breaks SSA chain)
      %loaded_ptr = llvm.load %box : !llvm.ptr -> !llvm.ptr
      // GEP into the loaded address to access field 0
      %field0_via_box = llvm.getelementptr %loaded_ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      // Load the value (requires findMemoryBlockByAddress fallback)
      %val0 = llvm.load %field0_via_box : !llvm.ptr -> i32

      %lit0 = sim.fmt.literal "field0_via_box = "
      %dec0 = sim.fmt.dec %val0 signed : i32
      %nl0 = sim.fmt.literal "\0A"
      %fmt0 = sim.fmt.concat (%lit0, %dec0, %nl0)
      sim.proc.print %fmt0

      // --- Test 2: Load through box, read field 1 ---
      %loaded_ptr2 = llvm.load %box : !llvm.ptr -> !llvm.ptr
      %field1_via_box = llvm.getelementptr %loaded_ptr2[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      %val1 = llvm.load %field1_via_box : !llvm.ptr -> i32

      %lit1 = sim.fmt.literal "field1_via_box = "
      %dec1 = sim.fmt.dec %val1 signed : i32
      %nl1 = sim.fmt.literal "\0A"
      %fmt1 = sim.fmt.concat (%lit1, %dec1, %nl1)
      sim.proc.print %fmt1

      // --- Test 3: Store through loaded address, then read back ---
      %loaded_ptr3 = llvm.load %box : !llvm.ptr -> !llvm.ptr
      %field0_store_via_box = llvm.getelementptr %loaded_ptr3[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      // Store 77 through the loaded-address GEP (requires address fallback)
      %c77 = arith.constant 77 : i32
      llvm.store %c77, %field0_store_via_box : i32, !llvm.ptr
      // Read back through another loaded address to verify
      %loaded_ptr4 = llvm.load %box : !llvm.ptr -> !llvm.ptr
      %field0_readback = llvm.getelementptr %loaded_ptr4[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"BoxTest", (i32, i32)>
      %val2 = llvm.load %field0_readback : !llvm.ptr -> i32

      %lit2 = sim.fmt.literal "field0_after_store_via_box = "
      %dec2 = sim.fmt.dec %val2 signed : i32
      %nl2 = sim.fmt.literal "\0A"
      %fmt2 = sim.fmt.concat (%lit2, %dec2, %nl2)
      sim.proc.print %fmt2

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
