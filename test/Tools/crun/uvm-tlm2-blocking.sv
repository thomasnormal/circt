// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test TLM-2.0 blocking transport interface.
// Verifies uvm_tlm_generic_payload and b_transport basic API.

// CHECK: [TEST] tlm2 b_transport: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class tlm2_target extends uvm_component;
    `uvm_component_utils(tlm2_target)
    uvm_tlm_b_target_socket #(tlm2_target) sock;
    int received_addr;
    bit received;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      sock = new("sock", this);
    endfunction

    task b_transport(uvm_tlm_generic_payload t, uvm_tlm_time delay);
      received_addr = t.get_address();
      received = 1;
      t.set_response_status(UVM_TLM_OK_RESPONSE);
    endtask
  endclass

  class tlm2_test extends uvm_test;
    `uvm_component_utils(tlm2_test)
    tlm2_target tgt;
    uvm_tlm_b_initiator_socket #() init_sock;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      tgt = tlm2_target::type_id::create("tgt", this);
      init_sock = new("init_sock", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      init_sock.connect(tgt.sock);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_generic_payload gp;
      uvm_tlm_time delay;
      byte unsigned data[];

      phase.raise_objection(this);

      gp = new("gp");
      data = new[4];
      data[0] = 8'hDE; data[1] = 8'hAD; data[2] = 8'hBE; data[3] = 8'hEF;
      gp.set_address(32'h1000);
      gp.set_data(data);
      gp.set_command(UVM_TLM_WRITE_COMMAND);
      gp.set_data_length(4);

      delay = new("delay");
      init_sock.b_transport(gp, delay);

      if (tgt.received && tgt.received_addr == 32'h1000 &&
          gp.get_response_status() == UVM_TLM_OK_RESPONSE)
        `uvm_info("TEST", "tlm2 b_transport: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "tlm2 b_transport: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("tlm2_test");
endmodule
