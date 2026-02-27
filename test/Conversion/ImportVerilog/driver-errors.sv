// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Test: multiple continuous assignments to the same variable are rejected.

module MultipleContAssign;
  int v;
  assign v = 12;
  // expected-error @below {{multiple continuous assignments to variable 'v'}}
  assign v = 13;
endmodule

// -----

// Test: mixed continuous and procedural assignments to the same variable should
// be rejected. IEEE 1800-2017 §10.3.1: A variable shall not be driven by both
// a continuous assignment and a procedural assignment.

module MixedContProcAssign;
  wire clk = 0;
  int v;
  assign v = 12;
  // expected-error @below {{cannot mix continuous and procedural assignments to variable 'v'}}
  always @(posedge clk) v <= ~v;
endmodule

// -----

// Test: disjoint continuous and procedural assignments are legal.

module DisjointContProcAssign;
  logic [1:0] v;
  assign v[0] = 1'b1;
  always_comb v[1] = 1'b0;
endmodule

// -----

// Test: a single continuous assignment should be fine (no error).

module SingleContAssign;
  int v;
  assign v = 42;
endmodule

// -----

// Test: multiple procedural assignments to the same variable are legal.

module MultipleProcAssign;
  wire clk = 0;
  int v;
  always @(posedge clk) v <= 1;
  always @(posedge clk) v <= 2;
endmodule

// -----

// Test: continuous assignments to different variables should be fine.

module DifferentVarsContAssign;
  int a, b;
  assign a = 1;
  assign b = 2;
endmodule

// -----

// Test: multiple continuous assignments to disjoint constant indices are legal.

module DisjointElementContAssign;
  logic [1:0][7:0] v;
  for (genvar i = 0; i < 2; i++) begin : g
    assign v[i] = 8'(i);
  end
endmodule

// -----

// Test: mixed struct field and indexed element assignments with disjoint paths
// are legal.

module DisjointStructPathContAssign;
  typedef struct packed {
    logic valid;
    logic [1:0][7:0] key;
  } key_out_t;

  key_out_t key_o;
  logic [1:0][7:0] key_state_q;
  logic invalid_stage_sel_o;
  logic [1:0][7:0] entropy_i;

  assign key_o.valid = 1'b1;
  for (genvar i = 0; i < 2; i++) begin : g
    assign key_o.key[i] = invalid_stage_sel_o ? {8{entropy_i[i][0]}}
                                               : key_state_q[i];
  end
endmodule

// -----

// Test: multiple continuous assignments to the same selected path are rejected.

module OverlapElementContAssign;
  logic [1:0][7:0] v;
  assign v[0] = 8'hAA;
  // expected-error @below {{multiple continuous assignments to variable 'v'}}
  assign v[0] = 8'h55;
endmodule

// -----

// Test: disjoint range and element assignments are legal.

module DisjointRangeElementContAssign;
  logic [5:0] v;
  assign v[1:0] = 2'b11;
  assign v[2] = 1'b0;
  assign v[3] = 1'b1;
endmodule

// -----

// Test: overlapping range and element assignments are rejected.

module OverlapRangeElementContAssign;
  logic [5:0] v;
  assign v[3:1] = 3'b101;
  // expected-error @below {{multiple continuous assignments to variable 'v'}}
  assign v[2] = 1'b0;
endmodule
