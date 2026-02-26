// EXPECT_CIRCT_DIAG: driven by always_ff procedure
module t(input logic clk, input logic a, input logic b);
  typedef struct packed {logic x; logic y;} S;
  S s;
  always_ff @(posedge clk) s <= '{x:a, y:b};
  always_ff @(posedge clk) s.x <= b;
endmodule
