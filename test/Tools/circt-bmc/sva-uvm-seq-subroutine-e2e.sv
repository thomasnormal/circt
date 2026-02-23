// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm-core/src --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers \
// RUN:     --strip-llhd-processes --lower-to-bmc="top-module=sva_uvm_seq_subroutine bound=5" | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_sva_seq_subroutine_test extends uvm_test;
  `uvm_component_utils(uvm_sva_seq_subroutine_test)

  function new(string name = "uvm_sva_seq_subroutine_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module sva_uvm_seq_subroutine(
    input logic clk,
    input logic valid,
    input logic [7:0] in,
    output logic [7:0] out);
  always_ff @(posedge clk) begin
    if (valid)
      out <= in + 8'd4;
  end

  sequence seq;
    @(posedge clk) valid ##1 (out == in + 8'd4, $display("seq"));
  endsequence

  assert property (seq);
endmodule

// CHECK-BMC: verif.bmc
// CHECK-BMC: loop
