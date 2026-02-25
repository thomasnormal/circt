// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s

// Ensure clock-source mapping can trace through llhd.sig/llhd.prb wrappers.
// CHECK: smt.solver
func.func @bmc_clock_source_llhd_probe() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_input_names = ["bmc_clk0", "bmc_clk1", "clk_sig"],
    bmc_clock_sources = [{arg_index = 2 : i32, clock_pos = 1 : i32, invert = false}],
    bmc_clock_keys = ["port:bmc_clk0", "port:bmc_clk1"]
  } init {
    %false = hw.constant false
    %clk0 = seq.to_clock %false
    %clk1 = seq.to_clock %false
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  } loop {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock):
    verif.yield %clk0, %clk1 : !seq.clock, !seq.clock
  } circuit {
  ^bb0(%clk0: !seq.clock, %clk1: !seq.clock, %clkSigIn: !hw.struct<value: i1, unknown: i1>):
    %t = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %clkSigIn : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig, %clkSigIn after %t : !hw.struct<value: i1, unknown: i1>
    %prb = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    %v = hw.struct_extract %prb["value"] : !hw.struct<value: i1, unknown: i1>
    %u = hw.struct_extract %prb["unknown"] : !hw.struct<value: i1, unknown: i1>
    %true = hw.constant true
    %notU = comb.xor %u, %true : i1
    %known = comb.and %v, %notU : i1
    %clocked = ltl.clock %known, posedge %known : i1
    verif.assert %clocked : !ltl.sequence
    verif.yield %clkSigIn : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}
