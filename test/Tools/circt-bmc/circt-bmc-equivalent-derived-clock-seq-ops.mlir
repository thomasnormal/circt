// RUN: circt-bmc --emit-mlir -b 1 --module clock_gate_const %s | FileCheck %s --check-prefix=GATE
// RUN: circt-bmc --emit-mlir -b 1 --module clock_mux_const %s | FileCheck %s --check-prefix=MUX

// Ensure derived clocks through seq.clock_gate/seq.clock_mux with constant
// enables map to the same BMC clock input.
module {
  hw.module @clock_gate_const(in %clk_in: i1, in %in: i1) {
    %true = hw.constant true
    %clk = seq.to_clock %clk_in
    %gate = seq.clock_gate %clk, %true
    %seq = ltl.delay %in, 0, 0 : i1
    %gate_i1 = seq.from_clock %gate
    %clocked = ltl.clock %seq, posedge %gate_i1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    hw.output
  }

  hw.module @clock_mux_const(in %clk0: i1, in %clk1: i1, in %in: i1) {
    %true = hw.constant true
    %clk0_c = seq.to_clock %clk0
    %clk1_c = seq.to_clock %clk1
    %mux = seq.clock_mux %true, %clk0_c, %clk1_c
    %seq = ltl.delay %in, 0, 0 : i1
    %mux_i1 = seq.from_clock %mux
    %clocked = ltl.clock %seq, posedge %mux_i1 : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    hw.output
  }
}

// GATE: func.func @clock_gate_const
// MUX: func.func @clock_mux_const
