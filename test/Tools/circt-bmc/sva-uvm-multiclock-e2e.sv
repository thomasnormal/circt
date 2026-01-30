// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:   --lower-to-bmc="top-module=sva_uvm_multiclock bound=5 allow-multi-clock" \
// RUN:   | FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang, uvm
// UVM multiclock test with UVM report function interception.

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_multiclock_test extends uvm_test;
  `uvm_component_utils(uvm_sva_multiclock_test)

  function new(string name = "uvm_sva_multiclock_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module sva_uvm_multiclock(
    input logic clk0,
    input logic clk1,
    input logic a,
    input logic b);
  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  property p1;
    @(posedge clk1) b |-> a;
  endproperty

  assert property (p0);
  assert property (p1);
endmodule

// CHECK-BMC: verif.bmc
// CHECK-BMC: loop
