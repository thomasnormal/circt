// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=1100000000 2>&1 | FileCheck %s
//
// Regression: failing sequence local-var assertions must not be masked as
// unresolved/pending at end-of-run.
//
// CHECK: SVA assertion failed
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

module clk_gen(
    input            valid,
    input            clk,
    output reg [7:0] out,
    input      [7:0] in
);
  reg [7:0] data_reg_0;
  reg [7:0] data_reg_1;
  reg [7:0] data_reg_2;

  initial begin
    data_reg_0 = 0;
    data_reg_1 = 0;
    data_reg_2 = 0;
    out = 0;
  end

  always @(posedge clk) begin
    if (valid) begin
      data_reg_0 <= in + 1;
      data_reg_1 <= data_reg_0 + 1;
      data_reg_2 <= data_reg_1 + 1;
      out <= data_reg_2 + 1;
    end
  end
endmodule

module top;
  int cycle;
  logic valid;
  logic clk;
  logic [7:0] out;
  logic [7:0] in;

  clk_gen dut(.valid(valid), .clk(clk), .out(out), .in(in));

  sequence seq;
    int x;
    @(posedge clk) (valid, x = in) ##4 (out == x + 3);
  endsequence

  assert property (seq) else
    $error("sequence check failed :assert: (False)");

  assign in = cycle[7:0];

  always @(posedge clk)
    cycle = cycle + 1;

  initial begin
    cycle = 0;
    clk = 0;
    valid = 1;
  end

  initial begin
    forever #50 clk = ~clk;
  end

  initial #1000 $finish;
endmodule
