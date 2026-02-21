// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=CHECK-IMPORT
// RUN: circt-verilog --ir-moore %s | FileCheck %s --check-prefix=CHECK-MOORE
// REQUIRES: slang

module sva_strong_weak(input logic clk, a, b);
  // CHECK-IMPORT-LABEL: moore.module @sva_strong_weak
  // CHECK-IMPORT: [[SSEQ:%.*]] = ltl.concat
  // CHECK-IMPORT: [[SEV:%.*]] = ltl.eventually [[SSEQ]]
  // CHECK-IMPORT: [[SSTRONG:%.*]] = ltl.and [[SSEQ]], [[SEV]]
  // CHECK-IMPORT: verif.assert
  // CHECK-IMPORT: verif.assert

  // CHECK-MOORE-LABEL: moore.module @sva_strong_weak
  // CHECK-MOORE: [[MSEQ:%.*]] = ltl.concat
  // CHECK-MOORE: [[MEV:%.*]] = ltl.eventually [[MSEQ]]
  // CHECK-MOORE: [[MSTRONG:%.*]] = ltl.and [[MSEQ]], [[MEV]]
  // CHECK-MOORE: verif.clocked_assert [[MSTRONG]], posedge
  // CHECK-MOORE: verif.clocked_assert [[MSEQ]], posedge
  assert property (@(posedge clk) strong(a ##1 b));

  assert property (@(posedge clk) weak(a ##1 b));
endmodule
