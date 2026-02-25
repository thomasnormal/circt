// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SVAActionBlockIOLabels(input logic clk, a, b, c, d);
  integer fd;
  logic [7:0] dyn;

  assert property (@(posedge clk) a |-> b) else $strobe("strobe_fail");
  assert property (@(posedge clk) b |-> c) else $monitor("monitor_fail");
  assert property (@(posedge clk) c |-> d) else $strobeb("strobeb_fail");
  assert property (@(posedge clk) d |-> a) else $fdisplay(fd, "fdisplay_fail");
  assert property (@(posedge clk) a |=> c) else $fstrobe(fd, dyn);

  // CHECK-LABEL: moore.module @SVAActionBlockIOLabels
  // CHECK: verif.clocked_assert {{.*}} label "strobe_fail"
  // CHECK: verif.clocked_assert {{.*}} label "monitor_fail"
  // CHECK: verif.clocked_assert {{.*}} label "strobeb_fail"
  // CHECK: verif.clocked_assert {{.*}} label "fdisplay_fail"
  // CHECK: verif.clocked_assert {{.*}} label "$fstrobe"
endmodule
