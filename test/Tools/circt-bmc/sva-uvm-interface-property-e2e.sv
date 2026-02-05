// XFAIL: *
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 5 --module=sva_uvm_interface_property - | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_interface_test extends uvm_test;
  `uvm_component_utils(uvm_sva_interface_test)

  function new(string name = "uvm_sva_interface_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

interface ifc(input logic clk);
  logic a;
  logic b;
  property p;
    @(posedge clk) a |-> b;
  endproperty
endinterface

module sva_uvm_interface_property(
    input logic clk,
    input logic a,
    input logic b);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;

  assert property (i.p);
endmodule

// CHECK-BMC: verif.bmc
// CHECK-BMC: loop
