// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top verif_noop_tb 2>&1 | FileCheck %s

// Test: verif.cover remains a no-op and verif.assume/verif.assert do not halt
// process execution in simulation when conditions hold.

// CHECK: before_cover
// CHECK: after_cover
// CHECK: before_assume
// CHECK: after_assume
// CHECK: [circt-sim] Simulation completed

module verif_noop_tb();
  logic a = 1;
  logic b = 0;

  cover #0 (a != 0);
  assume #0 (a != 0);

  initial begin
    $display("before_cover");
    #1;
    $display("after_cover");
    $display("before_assume");
    #1;
    $display("after_assume");
    $finish;
  end
endmodule
