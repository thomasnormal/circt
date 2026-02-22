// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemDebugCheckpointSubroutine(input logic clk, a);
  sequence s_debug;
    (1, $showscopes(), $input("CMD"), $key(), $nokey(), $log(), $nolog()) ##1 a;
  endsequence

  sequence s_checkpoint;
    (1, $save("CKPT"), $restart("CKPT"), $incsave("CKPT"), $reset()) ##1 a;
  endsequence

  // Debug/checkpoint match-item tasks should be recognized and not ignored.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemDebugCheckpointSubroutine
  // CHECK: verif.assert
  assert property (@(posedge clk) s_debug);
  assert property (@(posedge clk) s_checkpoint);
endmodule

// DIAG: warning: $save is not supported in circt-sim (checkpoint/restart not implemented)
// DIAG: warning: $restart is not supported in circt-sim (checkpoint/restart not implemented)
// DIAG: warning: $incsave is not supported in circt-sim (checkpoint/restart not implemented)
// DIAG: warning: $reset is not supported in circt-sim (checkpoint/restart not implemented)
// DIAG-NOT: ignoring system subroutine `$showscopes`
// DIAG-NOT: ignoring system subroutine `$input`
// DIAG-NOT: ignoring system subroutine `$key`
// DIAG-NOT: ignoring system subroutine `$nokey`
// DIAG-NOT: ignoring system subroutine `$log`
// DIAG-NOT: ignoring system subroutine `$nolog`
// DIAG-NOT: ignoring system subroutine `$save`
// DIAG-NOT: ignoring system subroutine `$restart`
// DIAG-NOT: ignoring system subroutine `$incsave`
// DIAG-NOT: ignoring system subroutine `$reset`
