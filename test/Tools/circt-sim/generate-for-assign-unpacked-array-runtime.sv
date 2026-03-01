// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #36: generate-for continuous assignments to unpacked
// array elements must drive each element.

module tb;
  genvar i;
  logic [3:0] sig[4];
  int fail = 0;

  generate
    for (i = 0; i < 4; i++) begin : g
      assign sig[i] = i;
    end
  endgenerate

  initial begin
    #1;
    for (int j = 0; j < 4; j++)
      if (sig[j] !== j[3:0])
        fail++;

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d sig=%0d,%0d,%0d,%0d",
               fail, sig[0], sig[1], sig[2], sig[3]);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
