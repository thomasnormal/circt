// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Test: multiple continuous assignments to the same variable should be rejected.
// IEEE 1800-2017 §10.3.1: It shall be an error to have multiple continuous
// assignments to the same variable.

module MultipleContAssign;
  int v;
  assign v = 12;
  // expected-error @below {{cannot have multiple continuous assignments to variable 'v'}}
  assign v = 13;
endmodule

// -----

// Test: mixed continuous and procedural assignments to the same variable should
// be rejected. IEEE 1800-2017 §10.3.1: A variable shall not be driven by both
// a continuous assignment and a procedural assignment.

module MixedContProcAssign;
  wire clk = 0;
  // expected-error @below {{cannot mix continuous and procedural assignments to variable 'v'}}
  int v;
  assign v = 12;
  always @(posedge clk) v <= ~v;
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
