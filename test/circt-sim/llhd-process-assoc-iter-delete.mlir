// RUN: circt-sim %s --top=test_assoc_iter_delete --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test that associative array iteration survives key deletion.
// This pattern occurs in UVM's get_adjacent_successor_nodes() which
// iterates over a phase successor map, deleting non-terminal nodes
// and inserting their successors during the same iteration loop.
//
// Before the fix, __moore_assoc_next(key) returned false when `key`
// had been deleted (find() failed). The fix uses upper_bound() as
// a fallback to find the next surviving key.

// CHECK: visited=3
// CHECK: [circt-sim] Simulation completed

// Runtime function declarations
llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr
llvm.func @__moore_assoc_get_ref(!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
llvm.func @__moore_assoc_delete_key(!llvm.ptr, !llvm.ptr)
llvm.func @__moore_assoc_first(!llvm.ptr, !llvm.ptr) -> i1
llvm.func @__moore_assoc_next(!llvm.ptr, !llvm.ptr) -> i1

// Test function: create array {10, 20, 30}, iterate while deleting
// the current key at each step.
llvm.func @test_iteration() -> i32 {
  // Create int-keyed array (keySize=8, valueSize=8)
  %ks = arith.constant 8 : i32
  %vs = arith.constant 8 : i32
  %arr = llvm.call @__moore_assoc_create(%ks, %vs) : (i32, i32) -> !llvm.ptr

  // Allocate key buffer on stack
  %one = arith.constant 1 : i64
  %key_buf = llvm.alloca %one x i64 : (i64) -> !llvm.ptr

  // Insert key 10
  %c10 = arith.constant 10 : i64
  llvm.store %c10, %key_buf : i64, !llvm.ptr
  %ref10 = llvm.call @__moore_assoc_get_ref(%arr, %key_buf, %vs) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
  %v100 = arith.constant 100 : i64
  llvm.store %v100, %ref10 : i64, !llvm.ptr

  // Insert key 20
  %c20 = arith.constant 20 : i64
  llvm.store %c20, %key_buf : i64, !llvm.ptr
  %ref20 = llvm.call @__moore_assoc_get_ref(%arr, %key_buf, %vs) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
  %v200 = arith.constant 200 : i64
  llvm.store %v200, %ref20 : i64, !llvm.ptr

  // Insert key 30
  %c30 = arith.constant 30 : i64
  llvm.store %c30, %key_buf : i64, !llvm.ptr
  %ref30 = llvm.call @__moore_assoc_get_ref(%arr, %key_buf, %vs) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
  %v300 = arith.constant 300 : i64
  llvm.store %v300, %ref30 : i64, !llvm.ptr

  // Begin iteration: first() -> key=10
  %has_first = llvm.call @__moore_assoc_first(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> i1
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  cf.cond_br %has_first, ^got_first, ^done(%c0_i32 : i32)

^got_first:
  // Visited key 10 → count=1. Delete it, then call next.
  llvm.call @__moore_assoc_delete_key(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> ()
  %n1 = llvm.call @__moore_assoc_next(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> i1
  cf.cond_br %n1, ^got_second, ^done(%c1_i32 : i32)

^got_second:
  // Visited key 20 → count=2. Delete it, then call next.
  %c2_i32 = arith.constant 2 : i32
  llvm.call @__moore_assoc_delete_key(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> ()
  %n2 = llvm.call @__moore_assoc_next(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> i1
  cf.cond_br %n2, ^got_third, ^done(%c2_i32 : i32)

^got_third:
  // Visited key 30 → count=3. Delete it, then call next.
  %c3_i32 = arith.constant 3 : i32
  llvm.call @__moore_assoc_delete_key(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> ()
  %n3 = llvm.call @__moore_assoc_next(%arr, %key_buf) : (!llvm.ptr, !llvm.ptr) -> i1
  cf.cond_br %n3, ^done(%c3_i32 : i32), ^done(%c3_i32 : i32)

^done(%count: i32):
  llvm.return %count : i32
}

hw.module @test_assoc_iter_delete() {
  %c0_i8 = hw.constant 0 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %sig = llhd.sig %c0_i8 : i8

  %fmt_pre = sim.fmt.literal "visited="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %count = llvm.call @test_iteration() : () -> i32
    %fmt_val = sim.fmt.dec %count : i32
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
