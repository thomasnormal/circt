// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_tlm_req_rsp_channel basic put/get operations.

// CHECK: [TEST] req_rsp channel put/get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class chan_item extends uvm_object;
    `uvm_object_utils(chan_item)
    int payload;
    function new(string name = "chan_item");
      super.new(name);
    endfunction
  endclass

  class channel_test extends uvm_test;
    `uvm_component_utils(channel_test)
    uvm_tlm_req_rsp_channel#(chan_item, chan_item) chan;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      chan = new("chan", this);
    endfunction

    task run_phase(uvm_phase phase);
      chan_item req_in, req_out;
      phase.raise_objection(this);

      req_in = chan_item::type_id::create("req_in");
      req_in.payload = 55;

      fork
        chan.put_request_export.put(req_in);
        begin
          chan.get_peek_request_export.get(req_out);
          if (req_out != null && req_out.payload == 55)
            `uvm_info("TEST", "req_rsp channel put/get: PASS", UVM_LOW)
          else
            `uvm_error("TEST", "req_rsp channel put/get: FAIL")
        end
      join

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("channel_test");
endmodule
