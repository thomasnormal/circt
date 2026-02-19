// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that rand_mode(0) preserves a field's value across randomize() calls,
// AND that other rand fields still get randomized.
// Bug: rand_mode(0) disables the field but randomize() doesn't actually
// randomize the remaining enabled fields (they all stay at 0).
class config_item;
  rand int priority_val;
  rand int count;
  rand int flags;

  function new();
    priority_val = 0;
    count = 0;
    flags = 0;
  endfunction
endclass

module top;
  initial begin
    config_item c = new();
    int saw_count_change = 0;
    int saw_flags_change = 0;
    int priority_preserved = 1;
    int i;

    // Set priority to a known value, then disable it
    c.priority_val = 12345;
    c.priority_val.rand_mode(0);

    // Randomize many times
    for (i = 0; i < 20; i++) begin
      void'(c.randomize());
      if (c.count != 0) saw_count_change = 1;
      if (c.flags != 0) saw_flags_change = 1;
      if (c.priority_val != 12345) priority_preserved = 0;
    end

    // priority_val should stay at 12345
    // CHECK: priority_preserved=1
    $display("priority_preserved=%0d", priority_preserved);

    // count should have been randomized (changed from 0)
    // CHECK: count_randomized=1
    $display("count_randomized=%0d", saw_count_change);

    // flags should have been randomized (changed from 0)
    // CHECK: flags_randomized=1
    $display("flags_randomized=%0d", saw_flags_change);

    $finish;
  end
endmodule
