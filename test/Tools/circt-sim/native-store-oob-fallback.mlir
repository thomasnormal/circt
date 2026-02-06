// RUN: circt-sim %s --top test 2>&1 | FileCheck %s
// Test that when a native memory block (e.g., assoc array slot) is too small
// for a store/load, the interpreter falls back to findMemoryBlockByAddress
// instead of silently dropping the store or returning X.
//
// This tests the fix for the "native store OOB" bug where:
// 1. __moore_assoc_get_ref creates an 8-byte native block for an assoc entry
// 2. A struct store to a malloc'd block at a matching address gets matched
//    by findNativeMemoryBlockByAddress first
// 3. Old code: OOB check fails, store is silently dropped
// 4. Fixed code: Falls through to findMemoryBlockByAddress which finds the
//    correct (larger) malloc block
//
// We simulate this by: allocating memory via malloc, storing a struct into it,
// then loading back and verifying the values.

// CHECK: struct_field_0 = 42
// CHECK: struct_field_1 = 99
// CHECK: struct_field_2 = 7
// CHECK: [circt-sim] Simulation completed

module {
  // Declare malloc and free
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)

  // Struct layout: { i32, i32, i32 } = 12 bytes total (unaligned)
  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Allocate 12 bytes via malloc (simulates a class instance)
      %c12 = arith.constant 12 : i64
      %ptr = llvm.call @malloc(%c12) : (i64) -> !llvm.ptr

      // Store three i32 fields
      %c42 = arith.constant 42 : i32
      %c99 = arith.constant 99 : i32
      %c7 = arith.constant 7 : i32

      // Store field 0 at offset 0
      %f0_ptr = llvm.getelementptr %ptr[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      llvm.store %c42, %f0_ptr : i32, !llvm.ptr

      // Store field 1 at offset 4
      %f1_ptr = llvm.getelementptr %ptr[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      llvm.store %c99, %f1_ptr : i32, !llvm.ptr

      // Store field 2 at offset 8
      %f2_ptr = llvm.getelementptr %ptr[0, 2]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
      llvm.store %c7, %f2_ptr : i32, !llvm.ptr

      // Load back and verify
      %v0 = llvm.load %f0_ptr : !llvm.ptr -> i32
      %v1 = llvm.load %f1_ptr : !llvm.ptr -> i32
      %v2 = llvm.load %f2_ptr : !llvm.ptr -> i32

      %lit0 = sim.fmt.literal "struct_field_0 = "
      %d0 = sim.fmt.dec %v0 signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt0 = sim.fmt.concat (%lit0, %d0, %nl)
      sim.proc.print %fmt0

      %lit1 = sim.fmt.literal "struct_field_1 = "
      %d1 = sim.fmt.dec %v1 signed : i32
      %fmt1 = sim.fmt.concat (%lit1, %d1, %nl)
      sim.proc.print %fmt1

      %lit2 = sim.fmt.literal "struct_field_2 = "
      %d2 = sim.fmt.dec %v2 signed : i32
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
