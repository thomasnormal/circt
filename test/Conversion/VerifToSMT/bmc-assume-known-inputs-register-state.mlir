// RUN: circt-opt %s --convert-verif-to-smt="assume-known-inputs=true" --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Register state should not be constrained by --assume-known-inputs.
// The option applies to non-state inputs only.
// CHECK-LABEL: func.func @test_bmc_known_inputs_skip_register_state
// CHECK: %[[INITCLOCK:.*]] = func.call @bmc_init() : () -> !smt.bv<1>
// CHECK-NEXT: %[[FOR:.*]]:3 = scf.for
// CHECK: %[[LOOPCALL:.*]] = func.call @bmc_loop(%arg1) : (!smt.bv<1>) -> !smt.bv<1>
func.func @test_bmc_known_inputs_skip_register_state() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 1 initial_values [1 : i2] attributes {
    bmc_input_names = ["clk", "reg_state"],
    bmc_reg_clocks = ["clk"]
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
  ^bb0(%clk: !seq.clock, %reg: !hw.struct<value: i1, unknown: i1>):
    %f = hw.constant false
    verif.assert %f : i1
    verif.yield %reg : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}
