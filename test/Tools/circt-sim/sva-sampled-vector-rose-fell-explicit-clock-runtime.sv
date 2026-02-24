// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: explicit-clock vector rose/fell use bit0 semantics

module top;
  logic clk = 0;
  logic [1:0] v = 2'b00;
  logic rose_v, fell_v;
  logic [2:0] step = 3'd0;

  assign rose_v = $rose(v, @(posedge clk));
  assign fell_v = $fell(v, @(posedge clk));

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
      3'd5: step <= 3'd6;
      default: ;
    endcase
  end

  // Explicit-clock sampled helpers become visible one cycle after sampling.
  assert property (@(posedge clk) (step == 3'd1) |=> !rose_v);
  assert property (@(posedge clk) (step == 3'd2) |=> rose_v);
  assert property (@(posedge clk) (step == 3'd3) |=> !fell_v);
  assert property (@(posedge clk) (step == 3'd4) |=> fell_v);

  always @(posedge clk) begin
    if (step == 3'd6) begin
      $display("SVA_PASS: explicit-clock vector rose/fell use bit0 semantics");
      $finish;
    end
  end
endmodule
