/*
:name: uvm_interface_property_mini
:description: minimal UVM interface property for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_interface_property_smoke extends uvm_test;
  `uvm_component_utils(uvm_interface_property_smoke)

  function new(string name = "uvm_interface_property_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

interface test_if(input logic clk);
  logic sig;

  property p;
    @(posedge clk) sig |-> ##1 sig;
  endproperty
endinterface

module top(input logic clk, input logic in);
  test_if intf(clk);
  assign intf.sig = in;

  assert property (intf.p);
endmodule
