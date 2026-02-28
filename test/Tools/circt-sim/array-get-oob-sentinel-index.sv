// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: ImportVerilog may lower out-of-bounds array accesses through a
// sentinel-index mux. The simulator must treat the sentinel path as OOB/X, not
// as a wrapped in-range element.

module top;
  int arr [0:15];
  int count;

  initial begin
    count = 0;
    for (int i = 0; i < 16; i++)
      arr[i] = 0;
    arr[0] = 1;

    // i == 16 is out of bounds and must not alias arr[0].
    for (int i = 0; i <= 16; i++)
      if (arr[i] > 0)
        count++;

    $display("COUNT=%0d A0=%0d A15=%0d", count, arr[0], arr[15]);
    // CHECK: COUNT=1 A0=1 A15=0
    #1 $finish;
  end
endmodule
