// RUN: echo "module extra; endmodule" > %t-extra.sv
// RUN: circt-verilog --parse-only --uvm-path=%S/../../lib/Runtime/uvm-core %t-extra.sv %s
// REQUIRES: slang
//
// Regression test: multi-root UVM compiles must not abort when auto-UVM support
// is enabled. Keep the UVM macros include local to the source file instead of
// forcing uvm_macros.svh as a separate root input.

module top;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  class my_item extends uvm_sequence_item;
    `uvm_object_utils(my_item)

    function new(string name = "my_item");
      super.new(name);
    endfunction
  endclass
endmodule
