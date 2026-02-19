// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $assertcontrol with various control types
// control_type: 3=AssertOff, 4=AssertOn, 5=AssertKill
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("ASSERTCONTROL_FAIL");
  end

  initial begin
    // Turn off assertions using $assertcontrol (type 3 = off)
    $assertcontrol(3);
    // CHECK: control_off
    $display("control_off");
    #1 clk = 1; #1 clk = 0;
    // CHECK-NOT: ASSERTCONTROL_FAIL
    // CHECK: no_fail_after_off
    $display("no_fail_after_off");

    // Turn on assertions using $assertcontrol (type 4 = on)
    $assertcontrol(4);
    #1 clk = 1; #1 clk = 0;
    // CHECK: ASSERTCONTROL_FAIL
    // CHECK: after_on
    $display("after_on");

    // Type 5 (kill) should suppress subsequent immediate assertion checks.
    a = 0;
    $assertcontrol(5);
    // CHECK: control_kill
    $display("control_kill");
    #1 clk = 1; #1 clk = 0;
    // CHECK-NOT: ASSERTCONTROL_FAIL
    // CHECK: no_fail_after_kill
    $display("no_fail_after_kill");

    $finish;
  end
endmodule
