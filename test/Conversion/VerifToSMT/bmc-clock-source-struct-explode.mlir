// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// CHECK: smt.solver
func.func @clock_source_struct_explode() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_clock_sources = [{arg_index = 1 : i32, clock_pos = 0 : i32, invert = false}]
  } init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  } loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  } circuit {
  ^bb0(%clk: !seq.clock, %sig: !hw.struct<value: i1, unknown: i1>):
    %value, %unknown = hw.struct_explode %sig : !hw.struct<value: i1, unknown: i1>
    %packed = comb.concat %value, %unknown : i1, i1
    %clock = comb.extract %packed from 1 : (i2) -> i1
    %clocked = ltl.clock %clock, posedge %clock : i1
    verif.assert %clocked : !ltl.sequence
    verif.yield %sig : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}
