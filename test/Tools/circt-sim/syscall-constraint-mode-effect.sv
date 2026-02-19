// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that constraint_mode(0) actually disables a constraint during randomize,
// and constraint_mode(1) re-enables it.
// Bug: constraint_mode tracks state but doesn't affect randomize() behavior.
class bounded;
  rand int x;
  constraint c_range { x >= 100; x <= 100; }

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    bounded obj = new();
    int ok;
    int saw_not_100 = 0;
    int i;

    // With constraint enabled, x should always be 100
    ok = obj.randomize();
    // CHECK: with_constraint=100
    $display("with_constraint=%0d", obj.x);

    // Disable the constraint
    obj.c_range.constraint_mode(0);

    // Randomize multiple times — without constraint, x should NOT always be 100
    for (i = 0; i < 10; i++) begin
      ok = obj.randomize();
      if (obj.x != 100) saw_not_100 = 1;
    end

    // CHECK: constraint_disabled_effect=1
    $display("constraint_disabled_effect=%0d", saw_not_100);

    // Re-enable the constraint and verify it's enforced again
    obj.c_range.constraint_mode(1);
    ok = obj.randomize();
    // CHECK: re_enabled=100
    $display("re_enabled=%0d", obj.x);

    $finish;
  end
endmodule
