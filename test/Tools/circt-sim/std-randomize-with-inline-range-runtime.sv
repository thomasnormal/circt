// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #26: std::randomize with inline constraints must
// respect the constrained domain.

module tb;
  int v;
  int fail = 0;

  initial begin
    // Exercise the inline range multiple times to catch ignored constraints.
    repeat (64) begin
      void'(std::randomize(v) with { v inside {[1:10]}; });
      if (v < 1 || v > 10)
        fail++;
    end

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d", fail);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
