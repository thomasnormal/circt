// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm-core/src --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:     --strip-llhd-processes --lower-to-bmc="top-module=sva_uvm_assert_final bound=2" | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_assert_final_test extends uvm_test;
  `uvm_component_utils(uvm_sva_assert_final_test)

  function new(string name = "uvm_sva_assert_final_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module sva_uvm_assert_final(
    input logic clk,
    input logic a);
  initial begin
    assert final (a);
  end
endmodule

// CHECK-BMC: verif.assert{{.*}}{bmc.final}
