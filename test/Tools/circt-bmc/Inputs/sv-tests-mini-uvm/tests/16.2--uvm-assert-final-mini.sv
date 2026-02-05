/*
:name: uvm_assert_final_mini
:description: minimal UVM assert final for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_assert_final_smoke extends uvm_test;
  `uvm_component_utils(uvm_assert_final_smoke)

  function new(string name = "uvm_assert_final_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top(input logic a, input logic b);
  initial begin
    assert final (a == b);
  end
endmodule
