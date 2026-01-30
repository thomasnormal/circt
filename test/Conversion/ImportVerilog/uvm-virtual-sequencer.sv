// RUN: circt-verilog --ir-moore -I %S/../../../lib/Runtime/uvm \
// XFAIL: *
// RUN:   %S/../../../lib/Runtime/uvm/uvm_pkg.sv %s | FileCheck %s

`include "uvm_macros.svh"
import uvm_pkg::*;

class my_virtual_sequencer extends uvm_virtual_sequencer;
  `uvm_component_utils(my_virtual_sequencer)
  function new(string name = "my_virtual_sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

// CHECK: moore.class.classdecl @my_virtual_sequencer
