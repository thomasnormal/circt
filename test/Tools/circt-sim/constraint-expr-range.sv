// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top test 2>&1 | FileCheck %s
// Test that ConstraintExprOp-based constraints (x >= lo; x <= hi) are extracted
// and applied during randomize(). Without this, randomize() would produce
// unconstrained values.

// CHECK: in_range = 1
// CHECK: [circt-sim] Simulation completed

module test;
  class Packet;
    rand int x;
    constraint c_range { x >= 10; x <= 20; }
  endclass

  initial begin
    Packet p = new;
    int ok;
    ok = p.randomize();
    // x should be in [10, 20]
    $display("in_range = %0d", (p.x >= 10 && p.x <= 20) ? 1 : 0);
    $finish;
  end
endmodule
