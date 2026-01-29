// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %clk: i1, in %a: i1, in %b: i1, out out: i1) {
  %delayed = ltl.delay %b, 1, 0 : i1
  %prop = ltl.implication %a, %delayed : i1, !ltl.sequence
  verif.clocked_assert %prop, posedge %clk : !ltl.property
  hw.output %a : i1
}

hw.module @modB(in %clk: i1, in %a: i1, in %b: i1, out out: i1) {
  %delayed = ltl.delay %b, 1, 0 : i1
  %prop = ltl.implication %a, %delayed : i1, !ltl.sequence
  verif.clocked_assert %prop, posedge %clk : !ltl.property
  hw.output %a : i1
}

// CHECK: smt.solver
// CHECK: smt.assert
