// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_local_var(input logic clk, valid,
                     input logic [7:0] in,
                     output logic [7:0] out);
  // CHECK-LABEL: moore.module @sva_local_var

  sequence seq;
    int x;
    @(posedge clk) (valid, x = in) ##4 (out == x + 4);
  endsequence
  // CHECK-DAG: moore.past {{%[a-z0-9]+}} delay 4
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 3, 0 : i1
  assert property (seq);

  property prop;
    int y;
    @(posedge clk) (valid, y = in) |-> ##4 (out == y + 4);
  endproperty
  // CHECK-DAG: moore.past {{%[a-z0-9]+}} delay 4
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 4, 0 : i1
  assert property (prop);

  property prop_compound_match_item;
    int z;
    @(posedge clk) (valid, z = in) ##1 (valid, z += 1) ##1 (out == z[7:0]);
  endproperty
  // CHECK-DAG: moore.add {{%[a-z0-9]+}}, {{%[a-z0-9]+}}
  assert property (prop_compound_match_item);

  property prop_shift_match_item;
    int s;
    @(posedge clk) (valid, s = in) ##1 (valid, s <<= 1) ##1 (out == s[7:0]);
  endproperty
  // CHECK-DAG: moore.shl {{%[a-z0-9]+}}, {{%[a-z0-9]+}}
  assert property (prop_shift_match_item);
endmodule
