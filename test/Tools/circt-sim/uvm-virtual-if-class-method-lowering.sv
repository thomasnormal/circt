// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null

// Regression: lowering a class method that accesses a virtual interface field
// must not trip MLIR region-isolation verification.

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class mem_item extends uvm_sequence_item;
  `uvm_object_utils(mem_item)
  rand bit we;
  rand logic [3:0] addr;
  function new(string name = "mem_item");
    super.new(name);
  endfunction
endclass

interface mem_if (input logic clk);
  logic we;
  logic [3:0] addr;
endinterface

class mem_driver extends uvm_driver #(mem_item);
  `uvm_component_utils(mem_driver)
  virtual mem_if vif;

  function new(string name = "mem_driver", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    mem_item req;
    forever begin
      seq_item_port.get_next_item(req);
      @(posedge vif.clk);
      vif.we <= req.we;
      vif.addr <= req.addr;
      seq_item_port.item_done();
    end
  endtask
endclass

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  mem_if vif(clk);
endmodule
