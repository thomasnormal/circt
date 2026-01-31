// RUN: circt-verilog --ir-moore --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s | FileCheck %s

`timescale 1ns/1ps
`include "uvm_macros.svh"
import uvm_pkg::*;

class my_virtual_sequencer extends uvm_virtual_sequencer;
  `uvm_component_utils(my_virtual_sequencer)
  function new(string name = "my_virtual_sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

// CHECK: moore.class.classdecl @my_virtual_sequencer
