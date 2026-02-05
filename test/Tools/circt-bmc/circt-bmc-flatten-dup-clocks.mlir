// RUN: circt-bmc --emit-mlir -b 2 --module top %s | FileCheck %s

// This constructs identical derived clocks in separate instances. After
// flattening, CSE should dedupe the seq.to_clock ops so BMC sees only one clock.
module {
  hw.module @child(in %clk_i: i1, in %in: i1, out out: i1) attributes {sym_visibility = "private"} {
    %clk = seq.to_clock %clk_i
    %reg = seq.compreg %in, %clk : i1
    hw.output %reg : i1
  }

  hw.module @top(in %clk_i: i1, in %in: i1) {
    %out0 = hw.instance "u0" @child(clk_i: %clk_i: i1, in: %in: i1) -> (out: i1)
    %out1 = hw.instance "u1" @child(clk_i: %clk_i: i1, in: %in: i1) -> (out: i1)
    %or = comb.or %out0, %out1 : i1
    verif.assert %or : i1
    hw.output
  }
}

// CHECK: func.func @top
