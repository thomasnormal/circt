// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_blocking_transport_imp transport() call.
// Creates an imp that handles transport, verifies response.

// CHECK: [TEST] transport response: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class xport_req extends uvm_object;
    `uvm_object_utils(xport_req)
    int cmd;
    function new(string name = "xport_req");
      super.new(name);
    endfunction
  endclass

  class xport_rsp extends uvm_object;
    `uvm_object_utils(xport_rsp)
    int status;
    function new(string name = "xport_rsp");
      super.new(name);
    endfunction
  endclass

  class xport_target extends uvm_component;
    `uvm_component_utils(xport_target)
    uvm_blocking_transport_imp#(xport_req, xport_rsp, xport_target) imp;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      imp = new("imp", this);
    endfunction

    task transport(xport_req req, output xport_rsp rsp);
      rsp = xport_rsp::type_id::create("rsp");
      rsp.status = req.cmd + 1;
    endtask
  endclass

  class xport_test extends uvm_test;
    `uvm_component_utils(xport_test)
    uvm_blocking_transport_port#(xport_req, xport_rsp) port;
    xport_target target;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      port = new("port", this);
      target = xport_target::type_id::create("target", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      port.connect(target.imp);
    endfunction

    task run_phase(uvm_phase phase);
      xport_req req;
      xport_rsp rsp;
      phase.raise_objection(this);
      req = xport_req::type_id::create("req");
      req.cmd = 10;
      port.transport(req, rsp);
      if (rsp != null && rsp.status == 11)
        `uvm_info("TEST", "transport response: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "transport response: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("xport_test");
endmodule
