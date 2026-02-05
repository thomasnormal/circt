// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module sva_sequence_subroutine(input logic clk, valid,
                               input logic [7:0] in,
                               output logic [7:0] out);
  // CHECK-LABEL: moore.module @sva_sequence_subroutine

  sequence seq;
    int x;
    @(posedge clk) (valid, x = in) ##4 (out == x + 4, $display("seq"));
  endsequence
  // CHECK-DAG: moore.past {{%[a-z0-9]+}} delay 4
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 3, 0 : i1
  assert property (seq);
endmodule
