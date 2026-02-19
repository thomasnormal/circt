// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that randomize() actually randomizes unconstrained rand fields.
// Bug: randomize() without constraints leaves all fields at their initial
// values. With rand_mode(0) on x, randomize should still randomize y,
// but currently y stays at 0 because unconstrained randomize is a no-op.
class pkt;
  rand int x;
  rand int y;

  function new();
    x = 0;
    y = 0;
  endfunction
endclass

module top;
  initial begin
    pkt p = new();
    int y_changed = 0;
    int i;

    // Disable x, leave y enabled (unconstrained)
    p.x.rand_mode(0);
    p.x = 555;

    // Randomize multiple times — y should get random values (not stay at 0)
    for (i = 0; i < 10; i++) begin
      void'(p.randomize());
      if (p.y != 0) y_changed = 1;
    end

    // x should be preserved at 555 (rand_mode off)
    // CHECK: x_preserved=1
    $display("x_preserved=%0d", p.x == 555);

    // y should have been randomized to a non-zero value at least once
    // CHECK: y_randomized=1
    $display("y_randomized=%0d", y_changed);

    $finish;
  end
endmodule
