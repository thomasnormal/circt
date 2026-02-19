// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that unconstrained randomize() (no class or inline constraints)
// actually produces random values — not all zeros.
// Bug: randomize() without constraints is a no-op, leaving fields at init value.
// IEEE 1800-2017 Section 18.7: randomize() should set rand-qualified fields
// to random values even without constraints.
class simple;
  rand bit [15:0] a;
  rand bit [15:0] b;

  function new();
    a = 0;
    b = 0;
  endfunction
endclass

module top;
  initial begin
    simple obj = new();
    int a_changed = 0;
    int b_changed = 0;
    int both_different = 0;
    int i;

    for (i = 0; i < 20; i++) begin
      void'(obj.randomize());
      if (obj.a != 0) a_changed = 1;
      if (obj.b != 0) b_changed = 1;
      if (obj.a != obj.b) both_different = 1;
    end

    // At least one field should have become non-zero
    // CHECK: a_changed=1
    $display("a_changed=%0d", a_changed);

    // CHECK: b_changed=1
    $display("b_changed=%0d", b_changed);

    // a and b should differ at least once (independent random values)
    // CHECK: fields_differ=1
    $display("fields_differ=%0d", both_different);

    $finish;
  end
endmodule
