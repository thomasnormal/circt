// EXPECT_CIRCT_DIAG: driven by always_ff procedure
typedef struct packed {
  logic a;
  logic b;
} illegal_s_t;

module illegal_struct_member_multi_ff_driver (
  input logic clk,
  input logic x,
  output illegal_s_t y
);
  always_ff @(posedge clk)
    y.a <= x;
  always_ff @(posedge clk)
    y.a <= ~x;
endmodule
