// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

// Check that hierarchical enum-member references (u.Idle) lower as constants
// and do not create extra threaded module outputs.

// CHECK-LABEL: moore.module private @child(
// CHECK-SAME: out o : !moore.l1, out state_q : !moore.ref<l2>)
// CHECK-NOT: out Idle
module child(input logic clk, input logic rst_n, output logic o);
  typedef enum logic [1:0] { Idle, Busy } state_e;
  state_e state_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state_q <= Idle;
    else
      state_q <= Busy;
  end

  assign o = (state_q == Idle);
endmodule

// CHECK-LABEL: moore.module @top(
// CHECK: %u.o, %u.state_q = moore.instance "u" @child
// CHECK-SAME: -> (o: !moore.l1, state_q: !moore.ref<l2>)
module top(input logic clk, input logic rst_n);
  logic o;
  child u(.clk(clk), .rst_n(rst_n), .o(o));
  wire is_idle = (u.state_q == u.Idle);
endmodule
