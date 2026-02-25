// RUN: circt-bmc -b 1 --allow-multi-clock --module top --emit-mlir %s | FileCheck %s

// Ensure child clock aliases are remapped to caller BMC clocks before SMT
// conversion. Previously this failed with:
//   clocked property uses a clock that is not a BMC clock input
//   (bmc.clock=clk_src_i)
// CHECK-LABEL: func.func @top()
// CHECK: smt.solver

module {
  hw.module private @child(in %clk_src_i : !seq.clock, in %d : i1, out q : i1) {
    %q = seq.compreg %d, %clk_src_i : i1
    %clk_src_i_i1 = seq.from_clock %clk_src_i
    verif.clocked_assert %q, posedge %clk_src_i_i1 : i1
    hw.output %q : i1
  }

  hw.module @top(in %clk_i : !seq.clock, in %clk_aon_i : !seq.clock,
                 in %d : i1) {
    %inst = hw.instance "u" @child(clk_src_i: %clk_i: !seq.clock,
                                   d: %d: i1) -> (q: i1)
    %other = seq.compreg %d, %clk_aon_i : i1
    hw.output
  }
}
