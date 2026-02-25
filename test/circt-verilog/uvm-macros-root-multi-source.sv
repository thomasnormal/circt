// RUN: echo "module extra; endmodule" > %t-extra.sv
// RUN: circt-verilog --parse-only --no-uvm-auto-include --uvm-path=%S/../../lib/Runtime/uvm-core -I %S/../../lib/Runtime/uvm-core/src %S/../../lib/Runtime/uvm-core/src/uvm_macros.svh %S/../../lib/Runtime/uvm-core/src/uvm_pkg.sv %t-extra.sv %s 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression test: passing uvm_macros.svh as a root input in a multi-source
// compile must not abort the frontend.

// CHECK: warning: skipping root input `uvm_macros.svh` in multi-source compile

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
