// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %clk: i1, in %a: i1, in %b: i1, out out: i1) {
  %prop = sva.prop.implication %a, %b : i1, i1
  sva.clocked_assert %prop, posedge %clk : !sva.property
  hw.output %a : i1
}

hw.module @modB(in %clk: i1, in %a: i1, in %b: i1, out out: i1) {
  %prop = sva.prop.implication %a, %b : i1, i1
  sva.clocked_assert %prop, posedge %clk : !sva.property
  hw.output %a : i1
}

// CHECK: smt.solver
// CHECK: smt.assert
