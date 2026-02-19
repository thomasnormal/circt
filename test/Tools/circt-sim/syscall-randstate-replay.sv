// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test get_randstate/set_randstate can replay randomization sequences.
// Bug: get_randstate/set_randstate are stubs — get returns empty string,
// set is a no-op, so replaying a random sequence is impossible.
class item;
  rand int x;
  constraint c { x >= 1; x <= 1000000; }

  function new();
    x = 0;
  endfunction
endclass

module top;
  initial begin
    item obj = new();
    process p;
    string saved_state;
    int val1, val2;

    p = process::self();

    // Save the random state
    saved_state = p.get_randstate();

    // CHECK: state_nonempty=1
    $display("state_nonempty=%0d", saved_state.len() > 0);

    // Generate first random value
    void'(obj.randomize());
    val1 = obj.x;

    // Restore the saved state
    p.set_randstate(saved_state);

    // Generate again — should reproduce the same value
    void'(obj.randomize());
    val2 = obj.x;

    // CHECK: replay_match=1
    $display("replay_match=%0d", val1 == val2);

    $finish;
  end
endmodule
