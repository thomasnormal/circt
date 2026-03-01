// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// Regression for issue #44: $rose/$fell inside always_ff should use the
// always_ff event clock.

module tb;
  logic clk = 0;
  logic sig = 0;
  int rises = 0;
  int falls = 0;

  always #5 clk = ~clk;

  always_ff @(posedge clk) begin
    if ($rose(sig))
      rises++;
    if ($fell(sig))
      falls++;
  end

  initial begin
    repeat (2) @(posedge clk);
    sig = 1;
    repeat (2) @(posedge clk);
    sig = 0;
    repeat (2) @(posedge clk);
    if (rises == 1 && falls == 1)
      $display("PASS rises=%0d falls=%0d", rises, falls);
    else
      $display("FAIL rises=%0d falls=%0d", rises, falls);
    $finish;
  end
endmodule
