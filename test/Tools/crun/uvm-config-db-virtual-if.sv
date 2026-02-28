// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test config_db set/get with virtual interface.
// Defines a simple interface, passes it via config_db, retrieves in component.

// CHECK: [TEST] config_db virtual interface: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

interface simple_if;
  logic clk;
  logic [7:0] data;
endinterface

module tb_top;
  import uvm_pkg::*;

  simple_if sif();

  class vif_consumer extends uvm_component;
    `uvm_component_utils(vif_consumer)
    virtual simple_if m_vif;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      bit ok;
      ok = uvm_config_db#(virtual simple_if)::get(this, "", "vif", m_vif);
      if (ok && m_vif != null)
        `uvm_info("TEST", "config_db virtual interface: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config_db virtual interface: FAIL")
    endfunction
  endclass

  class vif_test extends uvm_test;
    `uvm_component_utils(vif_test)
    vif_consumer consumer;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_config_db#(virtual simple_if)::set(this, "consumer", "vif", sif);
      consumer = vif_consumer::type_id::create("consumer", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("vif_test");
endmodule
