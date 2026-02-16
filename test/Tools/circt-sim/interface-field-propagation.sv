// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test that interface field changes propagate correctly from a parent module
// (driver) through interface ports to a child module (monitor). This exercises
// the forwardPropagateOnSignalChange path which ensures that when a parent
// interface signal is driven, child copies see the updated value.
//
// Previously, the monitor module would see stale (initial) values because
// signal changes on the parent interface were not forwarded to child copies.

interface bus_if(input clk);
  logic [7:0] addr;
  logic [7:0] data;
  logic       wr_en;
endinterface

// Driver drives interface fields on negedge
module driver(bus_if bif);
  initial begin
    bif.addr  = 8'd0;
    bif.data  = 8'd0;
    bif.wr_en = 1'b0;
    @(negedge bif.clk);
    bif.addr  = 8'hAA;
    bif.data  = 8'h55;
    bif.wr_en = 1'b1;
    @(negedge bif.clk);
    bif.addr  = 8'hBB;
    bif.data  = 8'hCC;
    bif.wr_en = 1'b0;
  end
endmodule

// Monitor reads interface fields on posedge (after driver sets on negedge)
module monitor(bus_if bif);
  initial begin
    @(posedge bif.clk);
    @(posedge bif.clk);
    // Driver set AA/55/1 on the negedge before this posedge
    $display("MON: addr=%h data=%h wr_en=%b", bif.addr, bif.data, bif.wr_en);
    @(posedge bif.clk);
    // Driver set BB/CC/0 on the negedge before this posedge
    $display("MON: addr=%h data=%h wr_en=%b", bif.addr, bif.data, bif.wr_en);
    $finish;
  end
endmodule

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  bus_if bus(clk);

  driver  drv(.bif(bus));
  monitor mon(.bif(bus));

  // CHECK: MON: addr=aa data=55 wr_en=1
  // CHECK: MON: addr=bb data=cc wr_en=0
endmodule
