// RUN: circt-opt %s --lower-to-bmc="top-module=top bound=2" | FileCheck %s

// Dead seq.to_clock artifacts can remain after lowering/externalization.
// Ensure clock discovery ignores dead to_clock ops and maps BMC clock inputs
// from live/register-clock metadata sources instead.
module {
  hw.module @top(in %clk : !hw.struct<value: i1, unknown: i1>, in %reg_state : i1, out reg_next : i1) attributes {
    num_regs = 1 : i32,
    initial_values = [false],
    bmc_reg_clock_sources = [{arg_index = 0 : i32, invert = false}],
    bmc_reg_clocks = [""]
  } {
    %value, %unknown = hw.struct_explode %clk : !hw.struct<value: i1, unknown: i1>
    %true = hw.constant true
    %not_unknown = comb.xor %unknown, %true : i1
    %known_high = comb.and %value, %not_unknown : i1

    // Intentionally dead: never consumed.
    %or = comb.or %unknown, %known_high : i1
    %not_or = comb.xor %or, %true : i1
    %dead = comb.and %known_high, %not_or : i1
    %dead_clk = seq.to_clock %dead

    %next = comb.and %reg_state, %known_high : i1
    verif.assert %next : i1
    hw.output %next : i1
  }
}

// CHECK-LABEL: verif.bmc
// CHECK: ^bb0(%arg0: !seq.clock, %arg1: !hw.struct<value: i1, unknown: i1>, %arg2: i1)
// CHECK: %[[VALUE:.+]] = hw.struct_extract %arg1["value"] : !hw.struct<value: i1, unknown: i1>
// CHECK: %[[UNKNOWN:.+]] = hw.struct_extract %arg1["unknown"] : !hw.struct<value: i1, unknown: i1>
// CHECK: %[[TRUE:.+]] = hw.constant true
// CHECK: %[[NOT_UNKNOWN:.+]] = comb.xor %[[UNKNOWN]], %[[TRUE]] : i1
// CHECK: %[[KNOWN_HIGH:.+]] = comb.and %[[VALUE]], %[[NOT_UNKNOWN]] : i1
// CHECK: %[[FROM_CLK:.+]] = seq.from_clock %arg0
// CHECK: %[[EQ:.+]] = comb.icmp eq %[[FROM_CLK]], %[[KNOWN_HIGH]] : i1
// CHECK: verif.assume %[[EQ]] : i1
