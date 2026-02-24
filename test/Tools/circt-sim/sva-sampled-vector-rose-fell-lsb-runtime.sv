// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: vector rose/fell use bit0 semantics
// CHECK-NOT: SVA assertion failed

module top;
  logic clk = 0;
  logic [1:0] v = 2'b00;
  logic [2:0] step = 3'd0;

  always #5 clk = ~clk;

  always @(negedge clk) begin
    case (step)
      3'd0: begin
        v <= 2'b10; // bit0 stays 0
        step <= 3'd1;
      end
      3'd1: begin
        v <= 2'b11; // bit0 rises 0->1
        step <= 3'd2;
      end
      3'd2: begin
        v <= 2'b01; // bit0 stays 1
        step <= 3'd3;
      end
      3'd3: begin
        v <= 2'b00; // bit0 falls 1->0
        step <= 3'd4;
      end
      3'd4: step <= 3'd5;
      default: ;
    endcase
  end

  assert property (@(posedge clk) (step == 3'd1) |-> !$rose(v));
  assert property (@(posedge clk) (step == 3'd2) |-> $rose(v));
  assert property (@(posedge clk) (step == 3'd3) |-> !$fell(v));
  assert property (@(posedge clk) (step == 3'd4) |-> $fell(v));

  always @(posedge clk) begin
    if (step == 3'd5) begin
      $display("SVA_PASS: vector rose/fell use bit0 semantics");
      $finish;
    end
  end
endmodule
