// RUN: circt-verilog %s --ir-llhd --timescale=1ns/1ns -Wno-implicit-conv -Wno-unknown-escape-code > %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=1000000 --max-wall-ms=120000 2>&1 | FileCheck %s

// Verify that seq_item_port.size() reflects native connect() state.
// CHECK: PORT_SIZE=1
// CHECK-NOT: DRVCONNECT
// CHECK-NOT: UVM_FATAL
// CHECK-NOT: UVM_ERROR

import uvm_pkg::*;
`include "uvm_macros.svh"

class my_item extends uvm_sequence_item;
  `uvm_object_utils(my_item)
  function new(string name="my_item");
    super.new(name);
  endfunction
endclass

class my_sequencer extends uvm_sequencer #(my_item);
  `uvm_component_utils(my_sequencer)
  function new(string name="my_sequencer", uvm_component parent=null);
    super.new(name, parent);
  endfunction
endclass

class my_driver extends uvm_driver #(my_item);
  `uvm_component_utils(my_driver)
  function new(string name="my_driver", uvm_component parent=null);
    super.new(name, parent);
  endfunction
endclass

module top;
  my_sequencer sqr;
  my_driver drv;
  initial begin
    sqr = new("sqr", null);
    drv = new("drv", null);
    drv.seq_item_port.connect(sqr.seq_item_export);
    $display("PORT_SIZE=%0d", drv.seq_item_port.size());
    $finish;
  end
endmodule
