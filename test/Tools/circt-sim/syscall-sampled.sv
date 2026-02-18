// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $sampled, $rose, $fell, $stable, $changed, $past
module top;
  reg clk = 0;
  reg [7:0] val = 0;

  always #5 clk = ~clk;

  initial begin
    val = 8'h00;
    @(posedge clk);

    val = 8'hFF;
    @(posedge clk);

    // $past should return previous value
    // CHECK: past=0
    $display("past=%0d", $past(val));

    // $sampled returns sampled value
    // CHECK: sampled=255
    $display("sampled=%0d", $sampled(val));

    // $rose on val[0]: was 0, now 1 → true
    // CHECK: rose=1
    $display("rose=%0d", $rose(val[0]));

    // $changed: val changed from 0 to 255
    // CHECK: changed=1
    $display("changed=%0d", $changed(val));

    // Keep same value
    @(posedge clk);
    // $stable: val didn't change
    // CHECK: stable=1
    $display("stable=%0d", $stable(val));

    // $fell: val[0] was 1, still 1 → false
    // CHECK: fell=0
    $display("fell=%0d", $fell(val[0]));

    $finish;
  end
endmodule
