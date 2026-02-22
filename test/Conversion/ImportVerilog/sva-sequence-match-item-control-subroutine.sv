// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemControlSubroutine(input logic clk, a);
  sequence s_stop;
    (1, $stop(1)) ##1 a;
  endsequence
  sequence s_finish;
    (1, $finish(0)) ##1 a;
  endsequence
  sequence s_dumpvars;
    (1, $dumpvars) ##1 a;
  endsequence

  // Control-side system subroutines in match-items should preserve effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemControlSubroutine
  // CHECK: moore.builtin.stop
  // CHECK: verif.assert
  assert property (@(posedge clk) s_stop);

  // CHECK: moore.builtin.finish 0
  // CHECK: verif.assert
  assert property (@(posedge clk) s_finish);

  // CHECK: moore.builtin.display
  // CHECK: {circt.dumpvars}
  // CHECK: verif.assert
  assert property (@(posedge clk) s_dumpvars);
endmodule
