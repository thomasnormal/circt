// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test deterministic randstate replay with seeded process:
// - srandom(seed) followed by randomize produces deterministic values
// - get_randstate/set_randstate can replay the exact same sequence
module top;
  class item;
    rand int x;
    constraint c { x >= 1; x <= 100; }

    function new();
      x = 0;
    endfunction
  endclass

  initial begin
    item obj = new();
    process p = process::self();
    string state;
    int a1, a2, a3;
    int b1, b2, b3;

    // Seed the process for deterministic behavior
    p.srandom(12345);

    // Save state
    state = p.get_randstate();

    // Generate 3 values
    void'(obj.randomize());
    a1 = obj.x;
    void'(obj.randomize());
    a2 = obj.x;
    void'(obj.randomize());
    a3 = obj.x;

    // Restore state and regenerate — should produce identical sequence
    p.set_randstate(state);
    void'(obj.randomize());
    b1 = obj.x;
    void'(obj.randomize());
    b2 = obj.x;
    void'(obj.randomize());
    b3 = obj.x;

    // CHECK: match_0=1
    $display("match_0=%0d", a1 == b1);
    // CHECK: match_1=1
    $display("match_1=%0d", a2 == b2);
    // CHECK: match_2=1
    $display("match_2=%0d", a3 == b3);

    // First value should be in constraint range [1,100]
    // CHECK: v0_in_range=1
    $display("v0_in_range=%0d", a1 >= 1);

    $finish;
  end
endmodule
