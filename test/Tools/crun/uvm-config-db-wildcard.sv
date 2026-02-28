// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test config_db wildcard matching.
// Verifies that set with global wildcard "*" path is found by child via get.
// Uses null context + wildcard pattern (proven in Runtime/uvm/config_db_test.sv).

// CHECK: [TEST] config_db wildcard: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class child_comp extends uvm_component;
    `uvm_component_utils(child_comp)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      // Use null context with explicit full name (proven pattern from Runtime suite)
      ok = uvm_config_db #(int)::get(null, get_full_name(), "my_key", val);
      if (ok && val == 42)
        `uvm_info("TEST", "config_db wildcard: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("config_db wildcard: FAIL (ok=%0b val=%0d)", ok, val))
    endtask
  endclass

  class env_comp extends uvm_component;
    `uvm_component_utils(env_comp)
    child_comp agent_0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent_0 = child_comp::type_id::create("agent_0", this);
    endfunction
  endclass

  class wildcard_test extends uvm_test;
    `uvm_component_utils(wildcard_test)
    env_comp env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Set with global wildcard using null context (proven in Runtime suite)
      uvm_config_db #(int)::set(null, "*", "my_key", 42);
      env = env_comp::type_id::create("env", this);
    endfunction
  endclass

  initial run_test("wildcard_test");
endmodule
