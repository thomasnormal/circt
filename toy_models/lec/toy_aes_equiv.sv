// Minimal toy AES-style LEC model for fast local iteration.
// Both wrappers expose a single `result` bit so `circt-lec` can compare them.

module toy_aes_ref(input logic [7:0] in, output logic [7:0] out);
  // Lightweight stand-in for a byte substitution path.
  assign out = {in[6:0], in[7]} ^ 8'h63;
endmodule

module toy_aes_impl(input logic [7:0] in, output logic [7:0] out);
  logic [7:0] rot;
  assign rot = {in[6:0], in[7]};
  assign out = rot ^ 8'h63;
endmodule

module toy_aes_lec_ref(output logic result);
  logic [7:0] in;
  logic [7:0] out_ref;
  assign in = 8'hA5;
  toy_aes_ref dut(.in(in), .out(out_ref));
  assign result = 1'b1;
endmodule

module toy_aes_lec_impl(output logic result);
  logic [7:0] in;
  logic [7:0] out_ref;
  logic [7:0] out_impl;
  assign in = 8'hA5;
  toy_aes_ref ref_dut(.in(in), .out(out_ref));
  toy_aes_impl impl_dut(.in(in), .out(out_impl));
  assign result = (out_ref === out_impl);
endmodule
