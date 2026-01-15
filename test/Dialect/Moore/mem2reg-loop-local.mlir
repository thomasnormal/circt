// RUN: circt-opt --mem2reg --verify-diagnostics %s | FileCheck %s

// Test that variables declared inside loops are NOT promoted by mem2reg.
// This prevents a dominance error where Mem2Reg would create block arguments
// at loop headers, but the variable definition inside the loop doesn't
// dominate the entry edge from outside the loop.

// CHECK-LABEL: func.func @LoopLocalVariable
// Variable declared inside the loop body should NOT be promoted
func.func @LoopLocalVariable(%arg0: i1) -> !moore.i32 {
  // CHECK: cf.br ^[[HEADER:bb[0-9]+]]
  cf.br ^bb1
^bb1:
  // CHECK: ^[[HEADER]]:
  // CHECK: %[[VAR:.*]] = moore.variable : <i32>
  // The variable must remain - it should NOT be promoted
  %0 = moore.constant 42 : i32
  %var = moore.variable : <i32>
  moore.blocking_assign %var, %0 : i32
  // CHECK: cf.cond_br %arg0, ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  // CHECK: ^[[BODY]]:
  // CHECK: moore.read %[[VAR]]
  %1 = moore.read %var : <i32>
  %2 = moore.constant 1 : i32
  %3 = moore.add %1, %2 : i32
  moore.blocking_assign %var, %3 : i32
  // Back-edge to loop header
  cf.br ^bb1
^bb3:
  // CHECK: ^[[EXIT]]:
  %result = moore.read %var : <i32>
  return %result : !moore.i32
}

// CHECK-LABEL: func.func @VariableOutsideLoop
// Variable declared OUTSIDE the loop CAN be promoted
func.func @VariableOutsideLoop(%arg0: i1, %arg1: !moore.i32) -> !moore.i32 {
  // CHECK-NOT: moore.variable
  // CHECK: [[DEFAULT:%.+]] = moore.constant hXXXXXXXX : l32
  %var = moore.variable : <l32>
  %init = moore.constant 0 : l32
  moore.blocking_assign %var, %init : l32
  cf.br ^bb1
^bb1:
  // Loop header - variable defined outside, so it can be promoted
  // CHECK: ^[[HEADER:bb[0-9]+]]([[ARG:%.+]]: !moore.l32):
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  // Loop body
  %val = moore.read %var : <l32>
  %one = moore.constant 1 : l32
  %newval = moore.add %val, %one : l32
  moore.blocking_assign %var, %newval : l32
  // Back-edge
  cf.br ^bb1
^bb3:
  // Exit
  %result = moore.read %var : <l32>
  // CHECK: return
  return %result : !moore.l32
}

// CHECK-LABEL: func.func @WhileLoopLocalVariable
// Test with a while-loop structure (condition check at header)
func.func @WhileLoopLocalVariable(%limit: !moore.i32) -> !moore.i32 {
  %zero = moore.constant 0 : i32
  cf.br ^bb1
^bb1:
  // Loop header with condition
  // CHECK: ^[[HEADER:bb[0-9]+]]:
  // CHECK: %[[VAR:.*]] = moore.variable : <i32>
  // Variable inside loop should NOT be promoted
  %counter = moore.variable : <i32>
  moore.blocking_assign %counter, %zero : i32
  %val = moore.read %counter : <i32>
  %cond = moore.ult %val, %limit : i32 -> i1
  %cond_bool = moore.to_builtin_bool %cond : i1
  cf.cond_br %cond_bool, ^bb2, ^bb3
^bb2:
  // Loop body - increment counter
  // CHECK: ^[[BODY:bb[0-9]+]]:
  %curr = moore.read %counter : <i32>
  %one = moore.constant 1 : i32
  %next = moore.add %curr, %one : i32
  moore.blocking_assign %counter, %next : i32
  // Back-edge to loop header
  cf.br ^bb1
^bb3:
  // Exit
  %result = moore.read %counter : <i32>
  return %result : !moore.i32
}

// CHECK-LABEL: func.func @NestedLoopLocalVariable
// Test nested loops - inner loop variable should not be promoted
func.func @NestedLoopLocalVariable(%arg0: i1, %arg1: i1) -> !moore.i32 {
  cf.br ^outer_header
^outer_header:
  // Outer loop header
  // CHECK: ^[[OUTER:bb[0-9]+]]:
  cf.cond_br %arg0, ^inner_header, ^exit
^inner_header:
  // Inner loop header with local variable
  // CHECK: ^[[INNER:bb[0-9]+]]:
  // CHECK: %[[VAR:.*]] = moore.variable : <i32>
  %inner_var = moore.variable : <i32>
  %init = moore.constant 0 : i32
  moore.blocking_assign %inner_var, %init : i32
  cf.cond_br %arg1, ^inner_body, ^outer_latch
^inner_body:
  // CHECK: ^[[INNER_BODY:bb[0-9]+]]:
  %val = moore.read %inner_var : <i32>
  %one = moore.constant 1 : i32
  %newval = moore.add %val, %one : i32
  moore.blocking_assign %inner_var, %newval : i32
  // Back-edge to inner loop header
  cf.br ^inner_header
^outer_latch:
  // Back-edge to outer loop header
  cf.br ^outer_header
^exit:
  %result = moore.constant 99 : i32
  return %result : !moore.i32
}

// CHECK-LABEL: func.func @SingleBlockLoopWithUsersInOtherBlock
// Variable in a single-block loop with a user (read) in the exit block
// This should NOT be promoted because the variable is in a loop and has
// users in another block.
func.func @SingleBlockLoopWithUsersInOtherBlock(%arg0: i1) -> !moore.i32 {
  cf.br ^loop
^loop:
  // CHECK: ^[[LOOP:bb[0-9]+]]:
  // CHECK: %[[VAR:.*]] = moore.variable : <i32>
  // Variable should NOT be promoted because it's in a loop with users in exit block
  %var = moore.variable : <i32>
  %val = moore.constant 42 : i32
  moore.blocking_assign %var, %val : i32
  // Self-loop back-edge
  cf.cond_br %arg0, ^loop, ^exit
^exit:
  // This read is in a different block from the variable definition
  // CHECK: ^[[EXIT:bb[0-9]+]]:
  // CHECK: moore.read %[[VAR]]
  %read = moore.read %var : <i32>
  return %read : !moore.i32
}

// CHECK-LABEL: func.func @LoopLocalVariableAllUsersInSameBlock
// Variable in a loop where ALL users are in the same block as the definition.
// This CAN be promoted because mem2reg doesn't need to create merge points
// when all users are local to the definition block.
func.func @LoopLocalVariableAllUsersInSameBlock(%arg0: i1) -> !moore.i32 {
  cf.br ^loop
^loop:
  // CHECK: ^[[LOOP:bb[0-9]+]]:
  // The variable CAN be promoted since all users (assign, read) are in the same block
  // CHECK-NOT: moore.variable
  %var = moore.variable : <i32>
  %val = moore.constant 42 : i32
  moore.blocking_assign %var, %val : i32
  %read = moore.read %var : <i32>
  // Self-loop back-edge
  cf.cond_br %arg0, ^loop, ^exit
^exit:
  // %read was computed in ^loop, no moore.read of %var here
  return %read : !moore.i32
}
