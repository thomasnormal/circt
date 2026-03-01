// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression: nested interface signal propagation through module interface
// ports must preserve edge waits in virtual BFM tasks.

interface leaf_if(input logic aclk, input logic aresetn);
  logic data;
endinterface

interface mid_if(input logic aclk, input logic aresetn);
  leaf_if leaf(.aclk(aclk), .aresetn(aresetn));
endinterface

interface bfm_if(input logic aclk, input logic aresetn);
  task automatic wait_for_reset_cycle();
    @(negedge aresetn);
    $display("NEG at %0t", $time);
    @(posedge aresetn);
    $display("POS at %0t", $time);
  endtask
endinterface

package harness_pkg;
  virtual bfm_if shared_vif;
endpackage

module agent(mid_if m);
  import harness_pkg::*;
  bfm_if bfm(.aclk(m.leaf.aclk), .aresetn(m.leaf.aresetn));

  initial shared_vif = bfm;
endmodule

class reset_driver;
  virtual bfm_if vif;

  function new(virtual bfm_if vif);
    this.vif = vif;
  endfunction

  task run();
    vif.wait_for_reset_cycle();
    $display("PASS");
  endtask
endclass

module tb;
  import harness_pkg::*;
  logic clk = 0;
  logic rst_n = 1;

  mid_if m(.aclk(clk), .aresetn(rst_n));
  agent a(m);

  always #5 clk = ~clk;

  initial begin
    reset_driver drv;

    wait (shared_vif != null);
    drv = new(shared_vif);
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
