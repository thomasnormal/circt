// XFAIL: *
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 5 --module=sva_uvm_seq_subroutine - | \
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
