// RUN: circt-verilog --ignore-timing-controls %s
// REQUIRES: slang

module top(input logic clk);
  task automatic wait_for_clk();
    @(posedge clk);
    #1;
  endtask

  initial begin
    wait_for_clk();
  end
endmodule
