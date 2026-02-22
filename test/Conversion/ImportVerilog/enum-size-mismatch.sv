// RUN: circt-translate --import-verilog --verify-diagnostics --split-input-file %s
// REQUIRES: slang

// Test: enum value with mismatched sized literal width should be an error.
// IEEE 1800-2017 §6.19: If the integer value expression is a sized literal
// constant, it shall be an error if the size is different from the enum base
// type, even if the value is within the representable range.
//
// Regression test: a previous change downgraded this from error to warning
// for VCS/Xcelium compatibility. This test ensures it stays an error.

module EnumSizeMismatch;
  enum logic [2:0] {
    // expected-error @below {{expression width of 4 does not exactly match declared enum type width of 3}}
    Global = 4'h2,
    // expected-error @below {{expression width of 4 does not exactly match declared enum type width of 3}}
    Local = 4'h3
  } myenum;
endmodule

// -----

// Test: enum value with matching width should be fine.

module EnumSizeMatch;
  enum logic [2:0] {
    Global = 3'h2,
    Local = 3'h3
  } myenum;
endmodule

// -----

// Test: unsized enum values should be fine (no mismatch).

module EnumUnsized;
  enum logic [2:0] {
    A = 2,
    B = 3
  } myenum;
endmodule
