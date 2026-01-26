// RUN: circt-verilog --ir-hw %s 2>&1 | FileCheck %s

// This test reproduces a Moore-to-Core lowering bug found in OpenTitan's
// prim_diff_decode.sv. The pattern involves a unique case statement with
// nested if-else chains that assign to the same variables.
//
// The bug manifests as a CFG mismatch where cf.cond_br passes more
// operands to a successor block than the block expects as arguments.
//
// Original error from prim_diff_decode.sv:
//   error: branch has 7 operands for successor #0, but target block has 4
// This reproducer triggers:
//   error: branch has 5 operands for successor #0, but target block has 3

// XFAIL: *
// CHECK: error: branch has {{[0-9]+}} operands for successor #0, but target block has {{[0-9]+}}

module nested_control_flow_bug (
  input  logic clk_i,
  input  logic rst_ni,
  input  logic cond1,
  input  logic cond2,
  input  logic cond3,
  output logic out1,
  output logic out2,
  output logic out3
);

  typedef enum logic [1:0] {StateA, StateB, StateC} state_e;
  state_e state_d, state_q;

  logic level_d, level_q;

  // This combinational block mirrors the structure in prim_diff_decode.sv
  // that causes the Moore-to-Core lowering bug
  always_comb begin
    // defaults
    state_d = state_q;
    level_d = level_q;
    out1    = 1'b0;
    out2    = 1'b0;
    out3    = 1'b0;

    unique case (state_q)
      StateA: begin
        if (cond1) begin
          level_d = cond2;
          if (cond2 && cond3) begin
            if (level_q) begin
              out1 = 1'b1;
            end else begin
              out2 = 1'b1;
            end
          end
        end else begin
          if (cond3) begin
            state_d = StateC;
            out3    = 1'b1;
          end else begin
            state_d = StateB;
          end
        end
      end

      StateB: begin
        if (cond1) begin
          state_d = StateA;
          level_d = cond2;
          if (level_q) out1 = 1'b1;
          else         out2 = 1'b1;
        end else begin
          if (cond2) begin
            // still in StateB
          end else begin
            state_d = StateC;
            out3    = 1'b1;
          end
        end
      end

      StateC: begin
        out3 = 1'b1;
        if (cond1) begin
          state_d = StateA;
          out3    = 1'b0;
          level_d = cond2;
          if (level_q) out1 = 1'b1;
          else         out2 = 1'b1;
        end
      end

      default: ;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= StateA;
      level_q <= 1'b0;
    end else begin
      state_q <= state_d;
      level_q <= level_d;
    end
  end

endmodule
