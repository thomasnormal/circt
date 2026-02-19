// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that randomize() actually produces random values for rand fields.
// Bug: Unconstrained randomize() leaves all rand fields at their initial value (0).
// IEEE 1800-2017 Section 18.7: randomize() should assign random values
// to all rand-qualified variables.
class item;
  rand int val;

  function new();
    val = 0;
  endfunction
endclass

module top;
  initial begin
    item obj = new();
    int saw_nonzero = 0;
    int i;

    // Randomize many times — at least ONE should produce a non-zero value
    for (i = 0; i < 20; i++) begin
      void'(obj.randomize());
      if (obj.val != 0) saw_nonzero = 1;
    end

    // CHECK: randomize_nonzero=1
    $display("randomize_nonzero=%0d", saw_nonzero);

    $finish;
  end
endmodule
