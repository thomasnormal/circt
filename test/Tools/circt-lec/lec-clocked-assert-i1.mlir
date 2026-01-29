// RUN: circt-lec --emit-mlir -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %clk: i1, in %a: i1, out out: i1) {
  verif.clocked_assert %a, posedge %clk : i1
  hw.output %a : i1
}

hw.module @modB(in %clk: i1, in %a: i1, out out: i1) {
  verif.clocked_assert %a, posedge %clk : i1
  hw.output %a : i1
}

// CHECK: smt.solver
// CHECK: smt.assert
