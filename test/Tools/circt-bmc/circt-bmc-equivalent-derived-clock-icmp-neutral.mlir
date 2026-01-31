// RUN: circt-bmc --emit-mlir -b 1 --module icmp_eq %s | FileCheck %s --check-prefix=ICMP
// RUN: circt-bmc --emit-mlir -b 1 --module neutral_ops %s | FileCheck %s --check-prefix=NEUTRAL
// RUN: circt-bmc --emit-mlir -b 1 --module mux_const %s | FileCheck %s --check-prefix=MUX

// Ensure derived clock expressions that simplify to the same signal map to a
// single BMC clock input.
module {
  hw.module @icmp_eq(in %clk_in: i1, in %in: i1) {
    %true = hw.constant true
    %eq = comb.icmp eq %clk_in, %true : i1
    %c0 = seq.to_clock %clk_in
    %r0 = seq.compreg %in, %c0 : i1

    %seq = ltl.delay %in, 0, 0 : i1
    %clocked = ltl.clock %seq, posedge %eq : !ltl.sequence
    verif.assert %clocked : !ltl.sequence
    hw.output
  }

  hw.module @neutral_ops(in %clk_in: i1, in %in: i1) {
    %true = hw.constant true
    %false = hw.constant false
    %and = comb.and bin %clk_in, %true : i1
    %or = comb.or bin %clk_in, %false : i1
    %c0 = seq.to_clock %clk_in
    %r0 = seq.compreg %in, %c0 : i1

    %seq = ltl.delay %in, 0, 0 : i1
    %clocked0 = ltl.clock %seq, posedge %and : !ltl.sequence
    %clocked1 = ltl.clock %seq, posedge %or : !ltl.sequence
    verif.assert %clocked0 : !ltl.sequence
    verif.assert %clocked1 : !ltl.sequence
    hw.output
  }

  hw.module @mux_const(in %clk_in: i1, in %alt: i1, in %in: i1) {
    %true = hw.constant true
    %mux = comb.mux %true, %clk_in, %alt : i1
    %c0 = seq.to_clock %clk_in
    %r0 = seq.compreg %in, %c0 : i1

    %c1 = seq.to_clock %mux
    %r1 = seq.compreg %in, %c1 : i1
    verif.assert %r0 : i1
    verif.assert %r1 : i1
    hw.output
  }
}

// ICMP: func.func @icmp_eq
// NEUTRAL: func.func @neutral_ops
// MUX: func.func @mux_const
