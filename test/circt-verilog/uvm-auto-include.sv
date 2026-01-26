// RUN: circt-verilog --uvm-path=%S/../../lib/Runtime/uvm %s
// REQUIRES: slang
// XFAIL: *
// UVM runtime has compilation issues.

module top;
  import uvm_pkg::*;

  class my_item extends uvm_sequence_item;
    `uvm_object_utils(my_item)

    function new(string name = "my_item");
      super.new(name);
    endfunction
  endclass
endmodule
