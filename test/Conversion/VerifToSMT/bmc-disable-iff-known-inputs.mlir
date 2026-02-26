// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// A 4-state input used as a disable-iff guard should be constrained to known
// bits even when --assume-known-inputs=false, to avoid spurious X-only fails
// on reset-style guards.
// CHECK-LABEL: func.func @disable_iff_known_input
// CHECK: [[RST:%.+]] = smt.declare_fun : !smt.bv<2>
// CHECK: [[UNK:%.+]] = smt.bv.extract [[RST]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: [[KNOWN:%.+]] = smt.eq [[UNK]], %c0_bv1 : !smt.bv<1>
// CHECK: smt.assert [[KNOWN]]
func.func @disable_iff_known_input() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %rst: !hw.struct<value: i1, unknown: i1>):
    %rst_value = hw.struct_extract %rst["value"] : !hw.struct<value: i1, unknown: i1>
    %rst_unknown = hw.struct_extract %rst["unknown"] : !hw.struct<value: i1, unknown: i1>
    %true = hw.constant true
    %rst_not_unknown = comb.xor %rst_unknown, %true : i1
    %rst_bool = comb.and %rst_value, %rst_not_unknown : i1

    // Keep a trivially true clocked property so the guard participates in the
    // same shape emitted by import-verilog disable iff lowering.
    %one = hw.constant true
    %prop_seq = ltl.implication %one, %one : i1, i1
    %clk_i1 = seq.from_clock %clk
    %clocked = ltl.clock %prop_seq, posedge %clk_i1 : !ltl.property
    %guarded = ltl.or %rst_bool, %clocked {sva.disable_iff} : i1, !ltl.property

    verif.assert %guarded : !ltl.property
    verif.yield %rst : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}
