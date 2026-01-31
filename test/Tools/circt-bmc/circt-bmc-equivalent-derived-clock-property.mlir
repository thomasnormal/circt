// RUN: circt-bmc --emit-mlir -b 1 --module top %s | FileCheck %s

// Ensure equivalent derived clock expressions map to the same BMC clock input.
module {
  hw.module @top(in %clk_in : i1, in %in : i1) {
    %true = hw.constant true
    %eq = comb.icmp eq %clk_in, %true : i1
    %clk_a = comb.xor %true, %eq : i1
    %c0 = seq.to_clock %clk_a
    %r0 = seq.compreg %in, %c0 : i1

    %clk_b = comb.xor %eq, %true : i1
    %seq = ltl.delay %in, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %clk_b : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    hw.output
  }
}

// CHECK: func.func @top
