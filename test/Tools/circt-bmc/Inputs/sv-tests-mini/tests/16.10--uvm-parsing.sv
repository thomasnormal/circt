/*
:name: uvm_parsing_only
:description: parsing-only UVM SVA test
:type: simulation elaboration parsing
:tags: 16.10 uvm
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_parsing_test extends uvm_test;
  `uvm_component_utils(uvm_parsing_test)

  function new(string name = "uvm_parsing_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top;
  initial begin
    run_test();
  end
endmodule
