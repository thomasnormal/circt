// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #34: member writes on unpacked array-of-struct elements
// must persist.

module tb;
  typedef struct {
    int x;
    logic [7:0] y;
  } rec_t;

  rec_t arr[4];
  int fail = 0;

  initial begin
    arr[0].x = 100;
    arr[0].y = 8'hAA;
    arr[1].x = 200;
    arr[1].y = 8'hBB;

    if (arr[0].x !== 100) fail++;
    if (arr[0].y !== 8'hAA) fail++;
    if (arr[1].x !== 200) fail++;
    if (arr[1].y !== 8'hBB) fail++;

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d a0=(%0d,%h) a1=(%0d,%h)",
               fail, arr[0].x, arr[0].y, arr[1].x, arr[1].y);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
