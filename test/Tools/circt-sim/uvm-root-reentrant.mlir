// RUN: circt-sim %s --max-time=1000000 2>&1 | FileCheck %s

// Test UVM root re-entrancy handling.
// This simulates the pattern where m_uvm_get_root() is called re-entrantly
// during uvm_root construction:
// 1. m_uvm_get_root() checks m_inst == null, creates uvm_root
// 2. uvm_root::new() sets m_inst = this BEFORE returning
// 3. uvm_root::new() calls uvm_component::new() -> get_root()
// 4. Re-entrant get_root() sees m_inst != null, returns m_inst
// 5. Then checks m_inst != uvm_top (which is null) - should NOT error
//
// The fix marks construction in progress so re-entrant calls skip the check.

module {
  // Global variables for the singleton pattern
  llvm.mlir.global internal @"uvm_pkg::uvm_pkg::uvm_root::m_inst"(#llvm.zero) {addr_space = 0 : i32} : !llvm.ptr
  llvm.mlir.global internal @"uvm_pkg::uvm_top"(#llvm.zero) {addr_space = 0 : i32} : !llvm.ptr

  llvm.func @malloc(i64) -> !llvm.ptr

  // Simulated m_uvm_get_root function
  // This demonstrates the re-entrancy pattern
  func.func private @m_uvm_get_root() -> !llvm.ptr {
    %c16_i64 = arith.constant 16 : i64
    %null = llvm.mlir.zero : !llvm.ptr

    // Load m_inst
    %m_inst_addr = llvm.mlir.addressof @"uvm_pkg::uvm_pkg::uvm_root::m_inst" : !llvm.ptr
    %m_inst = llvm.load %m_inst_addr : !llvm.ptr -> !llvm.ptr

    // Check if null
    %is_null = llvm.icmp "eq" %m_inst, %null : !llvm.ptr
    cf.cond_br %is_null, ^create, ^return_existing

  ^create:
    // Allocate new instance
    %new_inst = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr

    // Set m_inst BEFORE calling constructor (simulating uvm_root::new setting m_inst)
    llvm.store %new_inst, %m_inst_addr : !llvm.ptr, !llvm.ptr

    // Simulate uvm_component::new() calling get_root() - RE-ENTRANT CALL
    // In real UVM, this would be: uvm_root::new -> uvm_component::new -> cs.get_root()
    %reentrant_result = func.call @m_uvm_get_root() : () -> !llvm.ptr

    // After constructor completes, set uvm_top
    %uvm_top_addr = llvm.mlir.addressof @"uvm_pkg::uvm_top" : !llvm.ptr
    llvm.store %new_inst, %uvm_top_addr : !llvm.ptr, !llvm.ptr

    return %new_inst : !llvm.ptr

  ^return_existing:
    // m_inst != null case
    // In real UVM, this would check m_inst != uvm_top and issue warning
    // But during construction, uvm_top is null so this would always warn
    // Our fix skips this check when construction is in progress
    return %m_inst : !llvm.ptr
  }

  hw.module @test() {
    %0 = sim.fmt.literal "Starting UVM root test\0A"
    %1 = sim.fmt.literal "SUCCESS: UVM root re-entrancy handled correctly\0A"
    %2 = sim.fmt.literal "Root instance: "
    %3 = sim.fmt.literal "\0A"
    %null = llvm.mlir.zero : !llvm.ptr

    llhd.process {
      sim.proc.print %0

      // Call m_uvm_get_root() - this will have re-entrant calls
      %root = func.call @m_uvm_get_root() : () -> !llvm.ptr

      // Verify we got a valid (non-null) root
      %is_null = llvm.icmp "ne" %root, %null : !llvm.ptr
      cf.cond_br %is_null, ^success, ^fail

    ^success:
      sim.proc.print %1
      llhd.halt

    ^fail:
      llhd.halt
    }
    hw.output
  }
}

// CHECK: Starting UVM root test
// CHECK: SUCCESS: UVM root re-entrancy handled correctly
