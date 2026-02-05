// XFAIL: *
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 2 --module=sva_uvm_assert_final - | \
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
