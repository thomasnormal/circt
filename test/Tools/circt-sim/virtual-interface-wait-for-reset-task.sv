// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression: virtual-interface task edge waits must observe live interface
// signal transitions. AVIP driver BFMs rely on this for waitForAresetn().

interface reset_if(input logic aclk, input logic aresetn);
  task automatic wait_for_reset_cycle();
    @(negedge aresetn);
    $display("NEG at %0t", $time);
    @(posedge aresetn);
    $display("POS at %0t", $time);
  endtask
endinterface

class reset_driver;
  virtual reset_if vif;

  function new(virtual reset_if vif);
    this.vif = vif;
  endfunction

  task run();
    vif.wait_for_reset_cycle();
    $display("PASS");
  endtask
endclass

module tb;
  logic clk = 0;
  logic rst_n = 1;
  reset_if ifc(.aclk(clk), .aresetn(rst_n));

  always #5 clk = ~clk;

  initial begin
    reset_driver drv = new(ifc);
    fork
      drv.run();
      begin
        #10 rst_n = 1'b0;
        #20 rst_n = 1'b1;
      end
    join
    $finish;
  end

  // CHECK: NEG at 10
  // CHECK: POS at 30
  // CHECK: PASS
endmodule
