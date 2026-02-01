// RUN: circt-verilog %s --ir-moore | FileCheck %s --check-prefixes=MOORE
// RUN: circt-verilog %s 2>&1 | FileCheck %s --check-prefixes=CORE

// Test supply0 and supply1 net types (IEEE 1800-2017 Section 6.7.1)

// MOORE-LABEL: moore.module @test_supply_nets
// CORE-LABEL: hw.module @test_supply_nets
module test_supply_nets(output wire o);
  // MOORE: moore.net supply0 : <l1>
  supply0 gnd;

  // MOORE: moore.net supply1 : <l1>
  supply1 vcc;

  // Use the supply nets so they are not optimized away
  // Supply0 (gnd) XOR Supply1 (vcc) results in 1, which is [true, false] in 4-state
  // CORE: hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
  assign o = gnd ^ vcc;
endmodule
