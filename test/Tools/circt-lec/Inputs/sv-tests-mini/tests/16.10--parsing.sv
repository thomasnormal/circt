/*
:name: parsing_only
:description: parsing-only SVA test (no LEC)
:type: simulation elaboration parsing
:tags: 16.10
:unsynthesizable: 1
*/

module top;
  logic clk;

  sequence seq;
    int x;
    @(posedge clk) (1'b1, x = 0) ##1 1'b1;
  endsequence

  assert property (seq);
endmodule
