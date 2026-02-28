// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir 2>/dev/null
// RUN: circt-sim --top top %t.mlir --max-time 100000000 2>&1 | FileCheck %s

// Regression: automatic task waits on interface-backed clock must wake on
// clock edges (instead of stalling until max-time).
// CHECK: PASS
// CHECK-NOT: Main loop exit: maxTime reached

interface mem_if(input logic clk);
  logic flag;
endinterface

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  mem_if vif(.clk(clk));

  task automatic wait_one_tick();
    @(posedge vif.clk);
    vif.flag = 1'b1;
  endtask

  initial begin
    wait_one_tick();
    if (vif.flag)
      $display("PASS");
    else
      $display("FAIL");
    $finish;
  end
endmodule
