// RUN: circt-sim %s --max-time=1000000 | FileCheck %s

// Test UVM singleton pattern - static class members with initialization
// This tests that:
// 1. Static class members are properly stored as globals
// 2. Global constructors run before processes/initial blocks
// 3. The singleton pattern (if null, create new) works correctly

module {
  // VTable for uvm_root class
  llvm.mlir.global internal @"uvm_root::__vtable__"(#llvm.zero) {addr_space = 0 : i32} : !llvm.array<1 x ptr>
  llvm.func @malloc(i64) -> !llvm.ptr

  // Static singleton member - initialized to null
  llvm.mlir.global internal @"uvm_root::m_inst"(#llvm.zero) {addr_space = 0 : i32} : !llvm.ptr

  hw.module @test() {
    %c42_i32 = hw.constant 42 : i32
    %c1_i32 = hw.constant 1 : i32
    %0 = sim.fmt.literal "Creating singleton...\0A"
    %1 = sim.fmt.literal "Singleton already exists\0A"
    %2 = sim.fmt.literal "First call - created new instance\0A"
    %3 = sim.fmt.literal "Second call - reused instance\0A"
    %4 = sim.fmt.literal "SUCCESS: Singleton pattern works\0A"
    %5 = sim.fmt.literal "FAILURE: Singleton pattern broken\0A"
    %6 = llvm.mlir.addressof @"uvm_root::m_inst" : !llvm.ptr
    %7 = llvm.mlir.constant(16 : i64) : i64
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.addressof @"uvm_root::__vtable__" : !llvm.ptr
    %10 = llvm.mlir.zero : !llvm.ptr

    llhd.process {
      // First call to get() - should create new instance
      %ref = builtin.unrealized_conversion_cast %6 : !llvm.ptr to !llhd.ref<!llvm.ptr>
      %inst1 = llhd.prb %ref : !llvm.ptr
      %is_null1 = llvm.icmp "eq" %inst1, %10 : !llvm.ptr
      cf.cond_br %is_null1, ^create1, ^reuse1

    ^create1:
      sim.proc.print %0
      // Allocate new instance
      %new1 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr
      llvm.store %8, %new1 : i32, !llvm.ptr
      %vtable_ptr1 = llvm.getelementptr %new1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_root", (i32, ptr, i32)>
      llvm.store %9, %vtable_ptr1 : !llvm.ptr, !llvm.ptr
      %field_ptr1 = llvm.getelementptr %new1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_root", (i32, ptr, i32)>
      llvm.store %c42_i32, %field_ptr1 : i32, !llvm.ptr
      // Store in singleton
      llvm.store %new1, %6 : !llvm.ptr, !llvm.ptr
      sim.proc.print %2
      cf.br ^call2(%new1 : !llvm.ptr)

    ^reuse1:
      sim.proc.print %1
      cf.br ^call2(%inst1 : !llvm.ptr)

    ^call2(%root1: !llvm.ptr):
      // Second call to get() - should return existing instance
      %inst2 = llhd.prb %ref : !llvm.ptr
      %is_null2 = llvm.icmp "eq" %inst2, %10 : !llvm.ptr
      cf.cond_br %is_null2, ^create2, ^reuse2

    ^create2:
      sim.proc.print %0
      %new2 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr
      llvm.store %8, %new2 : i32, !llvm.ptr
      %vtable_ptr2 = llvm.getelementptr %new2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_root", (i32, ptr, i32)>
      llvm.store %9, %vtable_ptr2 : !llvm.ptr, !llvm.ptr
      %field_ptr2 = llvm.getelementptr %new2[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"uvm_root", (i32, ptr, i32)>
      llvm.store %c42_i32, %field_ptr2 : i32, !llvm.ptr
      llvm.store %new2, %6 : !llvm.ptr, !llvm.ptr
      cf.br ^check(%root1, %new2 : !llvm.ptr, !llvm.ptr)

    ^reuse2:
      sim.proc.print %3
      cf.br ^check(%root1, %inst2 : !llvm.ptr, !llvm.ptr)

    ^check(%r1: !llvm.ptr, %r2: !llvm.ptr):
      // Check if both references point to the same instance
      %same = llvm.icmp "eq" %r1, %r2 : !llvm.ptr
      %result = arith.select %same, %4, %5 : !sim.fstring
      sim.proc.print %result
      llhd.halt
    }
    hw.output
  }
}

// CHECK: Creating singleton...
// CHECK: First call - created new instance
// CHECK: Second call - reused instance
// CHECK: SUCCESS: Singleton pattern works
