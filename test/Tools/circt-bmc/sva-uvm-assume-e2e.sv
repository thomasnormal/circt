// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm-core/src --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:     --strip-llhd-processes --lower-to-bmc="top-module=sva_uvm_assume bound=2" | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_assume_test extends uvm_test;
  `uvm_component_utils(uvm_sva_assume_test)

  function new(string name = "uvm_sva_assume_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module sva_uvm_assume(
    input logic clk,
    input logic a,
    input logic b);
  assume property (@(posedge clk) a |-> b);
endmodule

// CHECK-BMC: verif.bmc
