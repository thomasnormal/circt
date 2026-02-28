// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=10000000 2>&1 | FileCheck %s

// Regression for issue #43: %h formatting should respect the source bit width.
module tb;
  int ival = 255;
  logic [7:0] bval = 8'hFF;
  string s;

  initial begin
    s = $sformatf("%h", ival);
    $display("INT=%s", s);

    s = $sformatf("%h", bval);
    $display("BYTE=%s", s);

    if ($sformatf("%h", ival) == "000000ff" && $sformatf("%h", bval) == "ff")
      $display("PASS");
    else
      $display("FAIL int=%s byte=%s", $sformatf("%h", ival),
               $sformatf("%h", bval));
    $finish;
  end

  // CHECK: INT=000000ff
  // CHECK: BYTE=ff
  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
