// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s
// CHECK-LABEL: func.func @bmc_final_check_edge
// CHECK: scf.for
// CHECK:   func.call @bmc_loop
// CHECK:   smt.bv.not
// CHECK:   smt.bv.and
// CHECK:   smt.eq
// CHECK:   func.call @bmc_circuit
// CHECK:   smt.ite
func.func @bmc_final_check_edge() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk", "sig"]
  }
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from = seq.from_clock %clk
    %true = hw.constant true
    %nclk = comb.xor %from, %true : i1
    %new = seq.to_clock %nclk
    verif.yield %new : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1):
    %prop = ltl.delay %sig, 0, 0 : i1
    verif.assert %prop {bmc.final, bmc.clock = "clk",
                        bmc.clock_edge = #ltl<clock_edge negedge>} : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
