// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=2" | FileCheck %s

// Ensure LowerToBMC handles wide four-state lanes with width-matched knownness
// masks. This used to build `comb.xor` with `(iN, i1)` operands.
module {
  hw.module @top(in %clk : !hw.struct<value: i2, unknown: i2>, in %sig : i1) attributes {
    num_regs = 0 : i32,
    initial_values = [],
    bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}],
    bmc_reg_clocks = [""]
  } {
    verif.assert %sig : i1
    hw.output
  }
}

// CHECK-LABEL: verif.bmc
// CHECK: ^bb0(%[[CLK:.+]]: !seq.clock, %[[CLK4S:.+]]: !hw.struct<value: i2, unknown: i2>, %[[SIG:.+]]: i1)
// CHECK: %[[VAL:.+]] = hw.struct_extract %[[CLK4S]]["value"] : !hw.struct<value: i2, unknown: i2>
// CHECK: %[[UNK:.+]] = hw.struct_extract %[[CLK4S]]["unknown"] : !hw.struct<value: i2, unknown: i2>
// CHECK: %[[ONES:.+]] = hw.constant -1 : i2
// CHECK: %[[NOT_UNK:.+]] = comb.xor %[[UNK]], %[[ONES]] : i2
