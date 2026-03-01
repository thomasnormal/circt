// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test uvm_tlm_req_rsp_channel for request-response pattern.
// One process puts a request and gets a response; another handles it.

// CHECK: [TEST] request received: PASS
// CHECK: [TEST] response received: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class req_txn extends uvm_sequence_item;
    `uvm_object_utils(req_txn)
    int addr;
    function new(string name = "req_txn");
      super.new(name);
    endfunction
  endclass

  class rsp_txn extends uvm_sequence_item;
    `uvm_object_utils(rsp_txn)
    int data;
    function new(string name = "rsp_txn");
      super.new(name);
    endfunction
  endclass

  class transport_test extends uvm_test;
    `uvm_component_utils(transport_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_req_rsp_channel #(req_txn, rsp_txn) chan;
      req_txn req, got_req;
      rsp_txn rsp, got_rsp;

      phase.raise_objection(this);

      chan = new("chan", this);

      fork
        // Requester: put request, then get response
        begin
          req = req_txn::type_id::create("req");
          req.addr = 'hAB;
          chan.put_request_export.put(req);
          chan.get_response_export.get(got_rsp);
        end
        // Responder: get request, put response
        begin
          chan.get_peek_request_export.get(got_req);
          rsp = rsp_txn::type_id::create("rsp");
          rsp.data = got_req.addr + 1;
          chan.put_response_export.put(rsp);
        end
        // Timeout guard
        begin
          #1000ns;
          `uvm_error("TEST", "TIMEOUT")
        end
      join_any
      disable fork;

      if (got_req != null && got_req.addr == 'hAB)
        `uvm_info("TEST", "request received: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "request received: FAIL")

      if (got_rsp != null && got_rsp.data == 'hAC)
        `uvm_info("TEST", "response received: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "response received: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("transport_test");
endmodule
