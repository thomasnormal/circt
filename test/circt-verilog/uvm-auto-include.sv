// RUN: circt-verilog --uvm-path=%S/../../lib/Runtime/uvm %s
// REQUIRES: slang
// Test UVM auto-include with minimal stubs.

module top;
  import uvm_pkg::*;

  class my_item extends uvm_sequence_item;
    `uvm_object_utils(my_item)

    function new(string name = "my_item");
      super.new(name);
    endfunction
  endclass
endmodule
