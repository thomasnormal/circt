// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Test: variables assigned in always_comb cannot be assigned by another
// always_comb process.
module MultiAlwaysCombDriver(
  input logic a,
  input logic b,
  output logic y
);
  always_comb y = a;
  // expected-error @below {{variable 'y' driven by always_comb procedure}}
  always_comb y = b;
endmodule

// -----

// Test: variables assigned in always_ff cannot be assigned by another
// always_ff process.
module MultiAlwaysFFDriver(
  input logic clk,
  input logic a,
  input logic b,
  output logic y
);
  always_ff @(posedge clk) y <= a;
  // expected-error @below {{variable 'y' driven by always_ff procedure}}
  always_ff @(posedge clk) y <= b;
endmodule

// -----

// Test: variables assigned in always_comb cannot be assigned by other
// procedural blocks.
module AlwaysCombAndInitialDriver(
  input logic a,
  output logic y
);
  always_comb y = a;
  // expected-error @below {{variable 'y' driven by always_comb procedure}}
  initial y = 1'b0;
endmodule

// -----

// Test: mixed always_ff and always_comb assignment reports always_comb
// conflict on the second writer.
module AlwaysFFAndAlwaysCombDriver(
  input logic clk,
  input logic a,
  input logic b,
  output logic y
);
  always_ff @(posedge clk) y <= a;
  // expected-error @below {{variable 'y' driven by always_comb procedure}}
  always_comb y = b;
endmodule
