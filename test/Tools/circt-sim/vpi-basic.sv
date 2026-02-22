// Simple SystemVerilog module for VPI testing.
// Has a clock, reset, and a counter signal.

module vpi_test_top(input logic clk, input logic rst);
  logic [7:0] counter;

  always_ff @(posedge clk or posedge rst) begin
    if (rst)
      counter <= 8'h00;
    else
      counter <= counter + 8'h01;
  end

  initial begin
    #100;
    $display("counter=%0d", counter);
    $finish;
  end
endmodule
