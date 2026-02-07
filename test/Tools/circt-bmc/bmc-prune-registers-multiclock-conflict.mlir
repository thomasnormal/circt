// RUN: not circt-bmc --emit-mlir -b 1 --module top --prune-bmc-registers=true --allow-multi-clock %s 2>&1 | FileCheck %s

// CHECK: --prune-bmc-registers is currently incompatible with --allow-multi-clock

hw.module @top(in %clk_a : !seq.clock, in %clk_b : !seq.clock, in %in : i1) {
  %r0 = seq.compreg %in, %clk_a : i1
  %r1 = seq.compreg %in, %clk_b : i1
  verif.assert %r0 : i1
  verif.assert %r1 : i1
  hw.output
}
