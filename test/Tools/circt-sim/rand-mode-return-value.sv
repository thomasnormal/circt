// Test that randomize() returns 0 when all rand variables are disabled via
// rand_mode(0), and returns 1 when at least one is enabled.
// IEEE 1800-2017 Section 18.8.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class a;
    rand int x;
    constraint c { x > 0; x < 100; }
endclass

module top;
  initial begin
    a obj = new;
    int ret;

    // Default: rand_mode on, randomize should succeed
    ret = obj.randomize();
    if (ret == 1)
      $display("TEST1_PASS ret=%0d", ret);
    else
      $display("TEST1_FAIL ret=%0d", ret);

    // Disable rand_mode for x (the only rand field)
    obj.x.rand_mode(0);

    // All rand fields disabled: randomize should return 0
    ret = obj.randomize();
    if (ret == 0)
      $display("TEST2_PASS ret=%0d", ret);
    else
      $display("TEST2_FAIL ret=%0d", ret);

    // Re-enable rand_mode
    obj.x.rand_mode(1);

    // randomize should succeed again
    ret = obj.randomize();
    if (ret == 1)
      $display("TEST3_PASS ret=%0d", ret);
    else
      $display("TEST3_FAIL ret=%0d", ret);

    $finish;
  end
endmodule

// CHECK: TEST1_PASS
// CHECK: TEST2_PASS
// CHECK: TEST3_PASS
// CHECK: Simulation completed
