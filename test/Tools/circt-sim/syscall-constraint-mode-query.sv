// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test constraint_mode() query returns correct value AND that disabling
// a constraint actually affects randomize() behavior.
// Bug: constraint_mode tracks on/off state but randomize() ignores it.
class bounded;
  rand int x;
  constraint c_exact { x == 42; }

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    bounded obj = new();
    int i;
    int saw_not_42 = 0;

    // Query should return 1 (enabled) by default
    // CHECK: query_default=1
    $display("query_default=%0d", obj.c_exact.constraint_mode());

    // With constraint enabled, x must always be 42
    void'(obj.randomize());
    // CHECK: constrained=42
    $display("constrained=%0d", obj.x);

    // Disable the constraint
    obj.c_exact.constraint_mode(0);

    // Query should now return 0
    // CHECK: query_disabled=0
    $display("query_disabled=%0d", obj.c_exact.constraint_mode());

    // Randomize many times — without constraint, x should vary
    for (i = 0; i < 30; i++) begin
      void'(obj.randomize());
      if (obj.x != 42) saw_not_42 = 1;
    end

    // With constraint disabled, SOME value should differ from 42
    // CHECK: saw_different=1
    $display("saw_different=%0d", saw_not_42);

    $finish;
  end
endmodule
