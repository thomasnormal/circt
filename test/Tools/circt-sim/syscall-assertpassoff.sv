// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top > %t.out 2>&1
// RUN: /usr/bin/grep -q "PASS_MSG_AFTER_PASSON" %t.out
// RUN: /usr/bin/grep -q "PASS_MSG_AFTER_CTRL6" %t.out
// RUN: /usr/bin/grep -q "assertpassoff_test_done" %t.out
// RUN: not /usr/bin/grep -q "PASS_MSG_SHOULD_NOT_PRINT" %t.out
// RUN: not /usr/bin/grep -q "PASS_MSG_SHOULD_NOT_PRINT_CTRL7" %t.out
// Test pass-action control for immediate assertions.
// IEEE 1800-2017 Section 20.12:
// - $assertpassoff suppresses assertion pass statements.
// - $assertpasson re-enables assertion pass statements.
// - $assertcontrol(7) maps to passoff, $assertcontrol(6) maps to passon.
module top;
  reg a = 1;

  initial begin
    $assertpassoff;
    assert (a == 1) $display("PASS_MSG_SHOULD_NOT_PRINT");

    $assertpasson;
    assert (a == 1) $display("PASS_MSG_AFTER_PASSON");

    $assertcontrol(7);
    assert (a == 1) $display("PASS_MSG_SHOULD_NOT_PRINT_CTRL7");

    $assertcontrol(6);
    assert (a == 1) $display("PASS_MSG_AFTER_CTRL6");

    $display("assertpassoff_test_done");
    $finish;
  end
endmodule
