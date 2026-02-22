// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir --top pipeline_tb --max-time=100000000 | FileCheck %s
// CHECK: PASS

module pipeline_stage(
  input wire clk,
  input wire rst,
  input wire [7:0] din,
  output reg [7:0] dout
);
  always @(posedge clk or posedge rst)
    if (rst) dout <= 0;
    else     dout <= din + 1;
endmodule

module pipeline_tb;
  reg clk = 0;
  reg rst = 1;
  wire [7:0] s0, s1, s2, s3, s4, s5, s6, s7;

  always #5 clk = ~clk;

  pipeline_stage p0(.clk(clk), .rst(rst), .din(8'd0), .dout(s0));
  pipeline_stage p1(.clk(clk), .rst(rst), .din(s0),   .dout(s1));
  pipeline_stage p2(.clk(clk), .rst(rst), .din(s1),   .dout(s2));
  pipeline_stage p3(.clk(clk), .rst(rst), .din(s2),   .dout(s3));
  pipeline_stage p4(.clk(clk), .rst(rst), .din(s3),   .dout(s4));
  pipeline_stage p5(.clk(clk), .rst(rst), .din(s4),   .dout(s5));
  pipeline_stage p6(.clk(clk), .rst(rst), .din(s5),   .dout(s6));
  pipeline_stage p7(.clk(clk), .rst(rst), .din(s6),   .dout(s7));

  initial begin
    #20 rst = 0;
    #99999980;
    $display("PASS s7=%0d", s7);
    $finish;
  end
endmodule
