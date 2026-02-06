// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// Test that recursive DFS cycle detection prevents infinite loops on cyclic
// graphs. The traverse function walks a linked list via func.call with the
// node pointer as arg0. When a cycle is detected (same function called with
// same arg0 already in the recursion chain), it returns zero instead of
// recursing infinitely.
//
// Graph: nodeA (val=1) -> nodeB (val=2) -> nodeA (cycle!)
//
// Without cycle detection: infinite recursion, hits max call depth.
// With cycle detection:
//   traverse(A) = 1 + traverse(B) = 1 + (2 + traverse(A)[cycle=0]) = 3

// CHECK: RESULT=3
// CHECK: [circt-sim] Simulation completed

module {
  // Traverse a linked list, summing values. Node struct: { i32 value, ptr next }
  func.func @traverse(%this: !llvm.ptr) -> i32 {
    // Load value field
    %val_ptr = llvm.getelementptr %this[0, 0]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
    %val = llvm.load %val_ptr : !llvm.ptr -> i32

    // Load next pointer
    %next_ptr = llvm.getelementptr %this[0, 1]
        : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
    %next = llvm.load %next_ptr : !llvm.ptr -> !llvm.ptr

    // If next != null, recurse
    %null = llvm.mlir.zero : !llvm.ptr
    %has_next = llvm.icmp "ne" %next, %null : !llvm.ptr
    cf.cond_br %has_next, ^recurse, ^done(%val : i32)

  ^recurse:
    %sub = func.call @traverse(%next) : (!llvm.ptr) -> i32
    %total = arith.addi %val, %sub : i32
    cf.br ^done(%total : i32)

  ^done(%result: i32):
    return %result : i32
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %one = arith.constant 1 : i64
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32

      // Allocate two nodes
      %nodeA = llvm.alloca %one x !llvm.struct<(i32, ptr)> : (i64) -> !llvm.ptr
      %nodeB = llvm.alloca %one x !llvm.struct<(i32, ptr)> : (i64) -> !llvm.ptr

      // nodeA.value = 1, nodeA.next = &nodeB
      %vA = llvm.getelementptr %nodeA[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %c1, %vA : i32, !llvm.ptr
      %nA = llvm.getelementptr %nodeA[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %nodeB, %nA : !llvm.ptr, !llvm.ptr

      // nodeB.value = 2, nodeB.next = &nodeA (creates cycle!)
      %vB = llvm.getelementptr %nodeB[0, 0]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %c2, %vB : i32, !llvm.ptr
      %nB = llvm.getelementptr %nodeB[0, 1]
          : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %nodeA, %nB : !llvm.ptr, !llvm.ptr

      // traverse(A) should return 1 + 2 + 0 = 3
      %result = func.call @traverse(%nodeA) : (!llvm.ptr) -> i32

      %lit = sim.fmt.literal "RESULT="
      %v = sim.fmt.dec %result signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %v, %nl)
      sim.proc.print %fmt

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}
