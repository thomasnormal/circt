// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test get_randstate/set_randstate produce deterministic replay.
// Bug: set_randstate is a no-op, so restoring RNG state has no effect.
// IEEE 1800-2017 Section 18.13.3: process::get_randstate() returns the
// current random state, and set_randstate() restores it.
//
// This test saves state, generates a value, advances RNG significantly,
// restores state, generates again, and checks the FIRST value repeats.
class item;
  rand int x;
  constraint c { x >= 1; x <= 10000; }

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    item obj = new();
    process p;
    string state;
    int first_val, second_val, third_val;
    int i;

    p = process::self();

    // Save state
    state = p.get_randstate();

    // Generate first sequence
    void'(obj.randomize());
    first_val = obj.x;

    // Advance RNG a lot
    for (i = 0; i < 100; i++)
      void'(obj.randomize());
    third_val = obj.x;

    // Restore to saved state
    p.set_randstate(state);

    // Generate again — should match first_val
    void'(obj.randomize());
    second_val = obj.x;

    // CHECK: first_eq_second=1
    $display("first_eq_second=%0d", first_val == second_val);

    // The third value (after 100 randomizations) should differ from first
    // (sanity check that RNG is actually advancing)
    // CHECK: first_ne_third=1
    $display("first_ne_third=%0d", first_val != third_val);

    $finish;
  end
endmodule
