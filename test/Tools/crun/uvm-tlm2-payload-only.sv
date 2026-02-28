// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test uvm_tlm_generic_payload field access without socket transport.
// Just the payload object creation and set/get API.

// CHECK: [TEST] payload create: PASS
// CHECK: [TEST] set/get address: PASS
// CHECK: [TEST] set/get command: PASS
// CHECK: [TEST] set/get data length: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_tlm2_payload_test extends uvm_test;
    `uvm_component_utils(probe_tlm2_payload_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_gp gp;
      byte unsigned data[];

      phase.raise_objection(this);

      gp = new("gp");
      if (gp != null)
        `uvm_info("TEST", "payload create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "payload create: FAIL")

      gp.set_address(64'hABCD_0000);
      if (gp.get_address() == 64'hABCD_0000)
        `uvm_info("TEST", "set/get address: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("set/get address: FAIL (got %0h)", gp.get_address()))

      gp.set_command(UVM_TLM_WRITE_COMMAND);
      if (gp.get_command() == UVM_TLM_WRITE_COMMAND)
        `uvm_info("TEST", "set/get command: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "set/get command: FAIL")

      data = new[4];
      data[0] = 8'hDE; data[1] = 8'hAD; data[2] = 8'hBE; data[3] = 8'hEF;
      gp.set_data(data);
      gp.set_data_length(4);
      if (gp.get_data_length() == 4)
        `uvm_info("TEST", "set/get data length: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("set/get data length: FAIL (got %0d)", gp.get_data_length()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_tlm2_payload_test");
endmodule
