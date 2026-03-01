// RUN: circt-opt --externalize-registers --split-input-file --verify-diagnostics %s

// Inventory regression for unsupported clock-shape diagnostics in
// externalize-registers. Non-traceable/non-keyable clock expressions should
// fail with a precise unsupported-clock-root message.

hw.module @unsupported_clock_div(in %clk: !seq.clock, in %in: i1, out out: i1) {
  %div = seq.clock_div %clk by 1
  // expected-error @below {{only clocks derived from block arguments, constants, process results, or keyable i1 expressions are supported}}
  %reg = seq.compreg %in, %div : i1
  hw.output %reg : i1
}
