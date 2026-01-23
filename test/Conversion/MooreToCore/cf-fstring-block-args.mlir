// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that cf.cond_br with sim::FormatStringType block arguments works correctly.
// This tests the identity type conversion for sim types in control flow operations.

// The issue this test exercises: When MLIR canonicalization merges blocks during
// Moore-to-Core conversion, it can create cf.cond_br ops with !sim.fstring block
// arguments. The CF structural type conversion patterns need to recognize these
// as legal because sim::FormatStringType is already a legal type.

// CHECK-LABEL: func.func @fstring_cf_branch
// CHECK-SAME: (%[[COND:.*]]: i1, %[[S1:.*]]: !sim.fstring, %[[S2:.*]]: !sim.fstring)
// CHECK: cf.cond_br %[[COND]], ^bb1(%[[S1]] : !sim.fstring), ^bb1(%[[S2]] : !sim.fstring)
// CHECK: ^bb1(%[[ARG:.*]]: !sim.fstring):
// CHECK: sim.proc.print %[[ARG]]
// CHECK: return
func.func @fstring_cf_branch(%cond: i1, %s1: !sim.fstring, %s2: !sim.fstring) {
  cf.cond_br %cond, ^bb1(%s1 : !sim.fstring), ^bb1(%s2 : !sim.fstring)
^bb1(%arg: !sim.fstring):
  sim.proc.print %arg
  return
}

// Test with cf.br - use a loop structure to preserve block arguments
// CHECK-LABEL: func.func @fstring_cf_br
// CHECK-SAME: (%[[COND:.*]]: i1, %[[S1:.*]]: !sim.fstring, %[[S2:.*]]: !sim.fstring)
// CHECK: cf.br ^bb1(%[[S1]] : !sim.fstring)
// CHECK: ^bb1(%[[ARG:.*]]: !sim.fstring):
// CHECK: sim.proc.print %[[ARG]]
// CHECK: cf.cond_br %[[COND]], ^bb1(%[[S2]] : !sim.fstring), ^bb2
// CHECK: ^bb2:
// CHECK: return
func.func @fstring_cf_br(%cond: i1, %s1: !sim.fstring, %s2: !sim.fstring) {
  cf.br ^bb1(%s1 : !sim.fstring)
^bb1(%arg: !sim.fstring):
  sim.proc.print %arg
  cf.cond_br %cond, ^bb1(%s2 : !sim.fstring), ^bb2
^bb2:
  return
}
