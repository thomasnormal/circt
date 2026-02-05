// RUN: circt-opt %s --convert-verif-to-smt --verify-diagnostics -allow-unregistered-dialect

func.func @bmc_sequence_clock_conflict() -> i1 {
  // expected-error@+1 {{failed to legalize operation 'verif.bmc' that was explicitly marked illegal}}
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["clk0", "clk1", "sig"]
  }
  init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  }
  loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    %from0 = seq.from_clock %clk0
    %from1 = seq.from_clock %clk1
    %true = hw.constant true
    %nclk0 = comb.xor %from0, %true : i1
    %nclk1 = comb.xor %from1, %true : i1
    %new0 = seq.to_clock %nclk0
    %new1 = seq.to_clock %nclk1
    verif.yield %new0, %new1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %sig: i1):
    %rep = ltl.repeat %sig, 2, 0 : i1
    %from1 = seq.from_clock %clk1
    %clocked = ltl.clock %rep, posedge %from1 : !ltl.sequence
    %prop = ltl.implication %sig, %clocked : i1, !ltl.sequence
    // expected-error@+1 {{clocked property uses conflicting clock information; ensure each property uses a single clock/edge}}
    verif.assert %prop {bmc.clock = "clk0"} : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}
