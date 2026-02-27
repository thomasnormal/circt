// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Regression: when a clocked check shares an ltl.delay with a plain check that
// triggers NFA-based sequence lowering, BMC must still normalize clocked checks
// first so delay/past rewrites see one uniform check representation.
// CHECK: smt.solver
// CHECK-NOT: verif.clocked_assert
func.func @bmc_clocked_shared_delay_with_plain_nfa() -> i1 {
  %res = verif.bmc bound 3 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk_i1", "sig", "en"]
  }
  init {
    %f = hw.constant false
    %clk = seq.to_clock %f
    %sig = hw.constant false
    %en = hw.constant true
    verif.yield %clk, %sig, %en : !seq.clock, i1, i1
  }
  loop {
  ^bb0(%clk: !seq.clock, %sig: i1, %en: i1):
    verif.yield %clk, %sig, %en : !seq.clock, i1, i1
  }
  circuit {
  ^bb0(%clk: !seq.clock, %sig: i1, %en: i1):
    %clk_i1 = seq.from_clock %clk
    %sig_seq = ltl.delay %sig, 0, 0 : i1
    %del = ltl.delay %sig, 1, 0 : i1
    %seq = ltl.concat %del, %sig_seq : !ltl.sequence, !ltl.sequence
    %plain = ltl.implication %seq, %sig : !ltl.sequence, i1
    verif.assert %plain : !ltl.property
    %clocked = ltl.implication %del, %sig : !ltl.sequence, i1
    verif.clocked_assert %clocked if %en, posedge %clk_i1 : !ltl.property
    verif.yield %clk, %sig, %en : !seq.clock, i1, i1
  }
  func.return %res : i1
}
