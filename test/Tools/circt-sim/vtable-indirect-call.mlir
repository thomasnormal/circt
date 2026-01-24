// RUN: circt-sim %s 2>&1 | FileCheck %s

// Test vtable initialization and indirect call support in circt-sim.
// This test verifies that:
// 1. LLVM globals with circt.vtable_entries are initialized at startup
// 2. llvm.addressof returns correct addresses for vtable globals
// 3. Loading from vtable memory returns function addresses
// 4. func.call_indirect resolves and calls functions through vtables

// Define a simple function to be called via vtable
func.func private @test_func(%arg0: i32) -> i32 {
  %c42 = arith.constant 42 : i32
  %result = arith.addi %arg0, %c42 : i32
  return %result : i32
}

// Another function for testing different vtable entries
func.func private @another_func(%arg0: i32) -> i32 {
  %c100 = arith.constant 100 : i32
  %result = arith.muli %arg0, %c100 : i32
  return %result : i32
}

// Vtable global with circt.vtable_entries attribute
// This simulates a class vtable with two methods at indices 0 and 1
llvm.mlir.global internal @"test_class::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @test_func],
    [1, @another_func]
  ]
} : !llvm.array<2 x ptr>

hw.module @vtable_test() {
  // Format strings for printing results
  %fmt_prefix1 = sim.fmt.literal "vtable call 1: result = "
  %fmt_prefix2 = sim.fmt.literal "vtable call 2: result = "
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    // Define constants inside the process
    %c5 = arith.constant 5 : i32
    %c3 = arith.constant 3 : i32

    // Get address of the vtable
    %vtable_addr = llvm.mlir.addressof @"test_class::__vtable__" : !llvm.ptr

    // Load function pointer from vtable[0]
    %zero = arith.constant 0 : i64
    %func_ptr_addr = llvm.getelementptr %vtable_addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %func_ptr = llvm.load %func_ptr_addr : !llvm.ptr -> !llvm.ptr

    // Cast to function type and call
    %func = builtin.unrealized_conversion_cast %func_ptr : !llvm.ptr to (i32) -> i32
    %result1 = func.call_indirect %func(%c5) : (i32) -> i32

    // Load function pointer from vtable[1]
    %one = arith.constant 1 : i64
    %func_ptr_addr2 = llvm.getelementptr %vtable_addr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x ptr>
    %func_ptr2 = llvm.load %func_ptr_addr2 : !llvm.ptr -> !llvm.ptr

    // Cast to function type and call
    %func2 = builtin.unrealized_conversion_cast %func_ptr2 : !llvm.ptr to (i32) -> i32
    %result2 = func.call_indirect %func2(%c3) : (i32) -> i32

    // Print results (format: "vtable call 1: result = 47", "vtable call 2: result = 300")
    // Note: 5 + 42 = 47, 3 * 100 = 300
    %fmt_val1 = sim.fmt.dec %result1 signed : i32
    %fmt_str1 = sim.fmt.concat (%fmt_prefix1, %fmt_val1, %fmt_nl)
    sim.proc.print %fmt_str1

    %fmt_val2 = sim.fmt.dec %result2 signed : i32
    %fmt_str2 = sim.fmt.concat (%fmt_prefix2, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_str2

    llhd.halt
  }
  hw.output
}

// CHECK: vtable call 1: result = 47
// CHECK: vtable call 2: result = 300
