// RUN: circt-verilog %s --ir-hw --ir-llhd -o %t.mlir 2>&1 | FileCheck %s --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s --check-prefix=SIM

// Test that assigning '0 to an array of packed structs with mixed 2-state
// (bit) and 4-state (logic) fields compiles and simulates correctly.
// This exercises the bitcast fix for mixed-state packed struct arrays where
// the flat 4-state representation has different bitwidth than the structured
// array-of-structs target type.

// CHECK-NOT: error:

package pkg;
  typedef enum logic [2:0] {
    OpA = 3'd0,
    OpB = 3'd1
  } op_e;

  typedef struct packed {
    bit          pend;    // 2-state
    op_e         opcode;  // 4-state (logic-based enum)
    logic [1:0]  size;    // 4-state
    logic [3:0]  mask;    // 4-state
  } pend_req_t;
endpackage

module top (
  input logic clk_i,
  input logic rst_ni
);
  import pkg::*;

  pend_req_t [3:0] pend_req;
  int cycle;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pend_req <= '0;
      cycle <= 0;
    end else begin
      cycle <= cycle + 1;
      if (cycle == 0) begin
        // After reset, all fields should be zero
        // SIM: pend_req[0].pend=0
        $display("pend_req[0].pend=%0d", pend_req[0].pend);
        // SIM: pend_req[0].opcode=0
        $display("pend_req[0].opcode=%0d", pend_req[0].opcode);
        // SIM: PASS
        $display("PASS");
        $finish;
      end
    end
  end
endmodule
