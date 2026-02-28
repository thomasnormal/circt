// RUN: circt-opt --sim-lower-dpi-func %s -verify-diagnostics --split-input-file

module {
  func.func private @cfoo(i64)
  // expected-error @below {{DPI callee @cfoo has type '(i64) -> ()' but expected '(i32) -> ()' for sim.func.dpi @dpi_alias}}
  sim.func.dpi @dpi_alias(in %arg0 : i32) attributes {verilogName = "cfoo"}
}
