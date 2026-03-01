// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test uvm_reg_predictor creation and adapter assignment.
// No bus transactions — just setup API.

// CHECK: [TEST] adapter create: PASS
// CHECK: [TEST] predictor create: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_adapter extends uvm_reg_adapter;
    `uvm_object_utils(probe_adapter)

    function new(string name = "probe_adapter");
      super.new(name);
    endfunction

    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      return null;
    endfunction

    virtual function void bus2reg(uvm_sequence_item bus_item,
                                  ref uvm_reg_bus_op rw);
    endfunction
  endclass

  class probe_predictor_test extends uvm_test;
    `uvm_component_utils(probe_predictor_test)

    probe_adapter adapter;
    uvm_reg_predictor #(uvm_sequence_item) predictor;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Components must be created in build_phase (illegal after in UVM 1.1d)
      adapter = probe_adapter::type_id::create("adapter");
      predictor = uvm_reg_predictor #(uvm_sequence_item)::type_id::create("predictor", this);
      if (predictor != null)
        predictor.adapter = adapter;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      if (adapter != null)
        `uvm_info("TEST", "adapter create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "adapter create: FAIL")

      if (predictor != null)
        `uvm_info("TEST", "predictor create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "predictor create: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_predictor_test");
endmodule
