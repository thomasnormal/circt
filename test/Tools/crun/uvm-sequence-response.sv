// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test get_response() matching via set_id_info().
// Sequence sends request, driver creates response with set_id_info, sequence gets it.

// CHECK: [TEST] response matching: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class rsp_item extends uvm_sequence_item;
    `uvm_object_utils(rsp_item)
    int data;
    function new(string name = "rsp_item");
      super.new(name);
    endfunction
  endclass

  class rsp_driver extends uvm_driver#(rsp_item);
    `uvm_component_utils(rsp_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      rsp_item req, rsp;
      forever begin
        seq_item_port.get_next_item(req);
        rsp = rsp_item::type_id::create("rsp");
        rsp.set_id_info(req);
        rsp.data = req.data + 100;
        seq_item_port.item_done(rsp);
      end
    endtask
  endclass

  class rsp_seq extends uvm_sequence#(rsp_item);
    `uvm_object_utils(rsp_seq)
    function new(string name = "rsp_seq");
      super.new(name);
    endfunction
    task body();
      rsp_item req, rsp;
      req = rsp_item::type_id::create("req");
      start_item(req);
      req.data = 42;
      finish_item(req);
      get_response(rsp);
      if (rsp != null && rsp.data == 142)
        `uvm_info("TEST", "response matching: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("response matching: FAIL (data=%0d)", rsp.data))
    endtask
  endclass

  class rsp_test extends uvm_test;
    `uvm_component_utils(rsp_test)
    uvm_sequencer#(rsp_item) sqr;
    rsp_driver drv;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      sqr = uvm_sequencer#(rsp_item)::type_id::create("sqr", this);
      drv = rsp_driver::type_id::create("drv", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction

    task run_phase(uvm_phase phase);
      rsp_seq s;
      phase.raise_objection(this);
      s = rsp_seq::type_id::create("s");
      s.start(sqr);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("rsp_test");
endmodule
