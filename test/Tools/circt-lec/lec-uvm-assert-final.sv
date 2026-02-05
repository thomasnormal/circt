// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_lec_assert_final_test extends uvm_test;
  `uvm_component_utils(uvm_lec_assert_final_test)

  function new(string name = "uvm_lec_assert_final_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module modA(input logic a);
  initial begin
    assert final (a);
  end
endmodule

module modB(input logic a);
  initial begin
    assert final (a);
  end
endmodule

// CHECK: unsat
