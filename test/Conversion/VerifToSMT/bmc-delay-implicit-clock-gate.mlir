// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=2" --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @top
// CHECK: scf.for
// CHECK:   func.call @bmc_loop
// CHECK:   %[[NOT:.*]] = smt.bv.not
// CHECK:   %[[EDGE_AND:.*]] = smt.bv.and
// CHECK:   %[[POSEDGE:.*]] = smt.eq %[[EDGE_AND]]
// CHECK:   func.call @bmc_circuit
// CHECK:   smt.ite %[[POSEDGE]], {{%.+}}, {{%.+}} : !smt.bv<1>

hw.module @top(in %clk: i1, in %sig: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %clkc = seq.to_clock %clk
  %seq = ltl.delay %sig, 1, 0 : i1
  %prop = ltl.implication %sig, %seq : i1, !ltl.sequence
  verif.assert %prop : !ltl.property
  hw.output
}
