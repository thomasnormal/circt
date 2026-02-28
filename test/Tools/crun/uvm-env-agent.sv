// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_env containing a uvm_agent with is_active field.

// CHECK: [TEST] agent is_active: UVM_ACTIVE
// CHECK: [TEST] agent parent is env: PASS
// CHECK: [TEST] env has 1 child: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_agent_ea extends uvm_agent;
    `uvm_component_utils(my_agent_ea)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      is_active = UVM_ACTIVE;
    endfunction
  endclass

  class my_env_ea extends uvm_env;
    `uvm_component_utils(my_env_ea)
    my_agent_ea agent;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent = my_agent_ea::type_id::create("agent", this);
    endfunction
  endclass

  class env_agent_test extends uvm_test;
    `uvm_component_utils(env_agent_test)
    my_env_ea env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = my_env_ea::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      if (env.agent.get_is_active() == UVM_ACTIVE)
        `uvm_info("TEST", "agent is_active: UVM_ACTIVE", UVM_LOW)
      else
        `uvm_error("TEST", "agent is_active: NOT UVM_ACTIVE")
      if (env.agent.get_parent() == env)
        `uvm_info("TEST", "agent parent is env: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "agent parent is env: FAIL")
      if (env.get_num_children() == 1)
        `uvm_info("TEST", "env has 1 child: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("env has %0d children", env.get_num_children()))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("env_agent_test");
endmodule
