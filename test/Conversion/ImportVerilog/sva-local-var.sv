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
endmodule
