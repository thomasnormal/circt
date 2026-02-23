// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm-core/src --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:     --strip-llhd-processes --lower-to-bmc="top-module=sva_uvm_expect bound=2" | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_expect_test extends uvm_test;
  `uvm_component_utils(uvm_sva_expect_test)

  function new(string name = "uvm_sva_expect_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module sva_uvm_expect(
    input logic clk,
    input logic a,
    input logic b);
  initial begin
    expect (@(posedge clk) a ##1 b);
  end
endmodule

// CHECK-BMC: verif.bmc
