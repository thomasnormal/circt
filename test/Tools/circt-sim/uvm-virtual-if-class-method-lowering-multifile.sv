// RUN: split-file %s %t
// RUN: circt-verilog --no-uvm-auto-include --single-unit --ir-llhd -I %t \
// RUN:   %t/uvm_pkg.sv %t/mem_if.sv %t/mem_driver.sv %t/tb_top.sv \
// RUN:   -o %t/out.mlir 2>&1
// REQUIRES: slang

// Regression (#14): lowering a class method that accesses a virtual interface
// field must not crash when the interface and class are defined in separate
// source files under --single-unit.

//--- uvm_pkg.sv
`timescale 1ns/1ps
package uvm_pkg;
  class uvm_object;
    function new(string name = "");
    endfunction
  endclass

  class uvm_sequence_item extends uvm_object;
    function new(string name = "");
      super.new(name);
    endfunction
  endclass

  class uvm_component extends uvm_object;
    function new(string name = "", uvm_component parent = null);
      super.new(name);
    endfunction
  endclass

  class uvm_phase;
  endclass

  class uvm_seq_item_pull_port #(type REQ = int);
    task get_next_item(output REQ req);
    endtask
    task item_done();
    endtask
  endclass

  class uvm_driver #(type REQ = int) extends uvm_component;
    uvm_seq_item_pull_port#(REQ) seq_item_port;
    function new(string name = "", uvm_component parent = null);
      super.new(name, parent);
      seq_item_port = new();
    endfunction
  endclass
endpackage

//--- uvm_macros.svh
`define uvm_object_utils(T)
`define uvm_component_utils(T)

//--- mem_if.sv
`timescale 1ns/1ps
interface mem_if(input logic clk);
  logic we;
  logic [3:0] addr;
endinterface

//--- mem_driver.sv
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

//--- tb_top.sv
`timescale 1ns/1ps
module tb_top;
  logic clk = 0;
  always #5 clk = ~clk;

  mem_if vif(.clk(clk));
endmodule
