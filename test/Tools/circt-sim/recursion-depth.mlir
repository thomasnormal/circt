// RUN: circt-sim %s --top test | FileCheck %s
// Test that the recursion depth guard (counter with threshold 20) allows
// legitimate re-entrant calls through a singleton pointer.
//
// Scenario: simulates the UVM factory pattern where nested factory calls all
// pass the same factory pointer as arg0. With the old visited-SET guard, the
// second call to @factory_create would be blocked because (factory_create, ptr)
// was already in the visited set. The depth-counter guard allows it because
// each re-entry only increments the depth to 2, well under the threshold of 20.
//
//   process -> factory_create(factory)     [depth 1 for factory_create]
//                -> factory_lookup(factory) [depth 1 for factory_lookup]
//                    -> factory_create(factory) [depth 2 for factory_create -- OK]
//                        returns 10
//                    returns 10 + 7 = 17
//                returns 17 + 25 = 42

// CHECK: nested_result = 42
// CHECK: [circt-sim] Simulation completed

module {
  // Simulate a singleton factory object with a single i32 field (config value).
  // struct factory_t { i32 config }
  llvm.mlir.global internal @factory_instance(dense<0> : tensor<1xi32>) : !llvm.array<1 x i32>

  // Inner create: just reads the factory config field and returns it.
  // This is the leaf of the call chain.
  func.func @factory_create_inner(%factory: !llvm.ptr) -> i32 {
    %cfg_ptr = llvm.getelementptr %factory[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
    %cfg = llvm.load %cfg_ptr : !llvm.ptr -> i32
    return %cfg : i32
  }

  // Lookup: calls factory_create (re-entrant with same arg0) and adds 7.
  func.func @factory_lookup(%factory: !llvm.ptr) -> i32 {
    %c7 = arith.constant 7 : i32
    // Re-entrant call to factory_create with the same factory pointer.
    // Under the old SET guard this would have been blocked.
    %inner = func.call @factory_create(%factory) : (!llvm.ptr) -> i32
    %result = arith.addi %inner, %c7 : i32
    return %result : i32
  }

  // Top-level create: calls factory_lookup (which calls factory_create again).
  // arg0 is the factory pointer throughout the chain.
  func.func @factory_create(%factory: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c25 = arith.constant 25 : i32

    // Check a flag to decide whether to recurse or return the leaf value.
    // Use a global counter to ensure only the first call recurses.
    %ctr_ptr = llvm.mlir.addressof @call_counter : !llvm.ptr
    %ctr = llvm.load %ctr_ptr : !llvm.ptr -> i32
    %c1 = arith.constant 1 : i32
    %new_ctr = arith.addi %ctr, %c1 : i32
    llvm.store %new_ctr, %ctr_ptr : i32, !llvm.ptr

    %is_first = arith.cmpi eq, %ctr, %c0 : i32
    cf.cond_br %is_first, ^do_lookup, ^leaf

  ^do_lookup:
    %lookup_result = func.call @factory_lookup(%factory) : (!llvm.ptr) -> i32
    %total = arith.addi %lookup_result, %c25 : i32
    return %total : i32

  ^leaf:
    // Second call: just read config and return it directly.
    %cfg_ptr = llvm.getelementptr %factory[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
    %cfg = llvm.load %cfg_ptr : !llvm.ptr -> i32
    return %cfg : i32
  }

  llvm.mlir.global internal @call_counter(0 : i32) : i32

  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Set factory config to 10
      %factory = llvm.mlir.addressof @factory_instance : !llvm.ptr
      %cfg_ptr = llvm.getelementptr %factory[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
      %c10 = arith.constant 10 : i32
      llvm.store %c10, %cfg_ptr : i32, !llvm.ptr

      // Call factory_create(factory) -> 42
      %result = func.call @factory_create(%factory) : (!llvm.ptr) -> i32

      %lit = sim.fmt.literal "nested_result = "
      %val = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %val, %nl)
      sim.proc.print %fmt

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
