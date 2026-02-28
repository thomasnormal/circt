// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test find_all("*agent*", comps) returns correct matches.
// Create hierarchy with env/agent/driver, call find_all, verify count.

// CHECK: [TEST] find_all agent count: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_driver_comp extends uvm_component;
    `uvm_component_utils(my_driver_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class my_agent_comp extends uvm_component;
    `uvm_component_utils(my_agent_comp)
    my_driver_comp drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      drv = my_driver_comp::type_id::create("drv", this);
    endfunction
  endclass

  class my_env_comp extends uvm_component;
    `uvm_component_utils(my_env_comp)
    my_agent_comp agent0;
    my_agent_comp agent1;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      agent0 = my_agent_comp::type_id::create("agent0", this);
      agent1 = my_agent_comp::type_id::create("agent1", this);
    endfunction
  endclass

  class find_all_test extends uvm_test;
    `uvm_component_utils(find_all_test)
    my_env_comp env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      env = my_env_comp::type_id::create("env", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_component comps[$];
      phase.raise_objection(this);

      find_all("*agent*", comps);
      // Should find agent0 and agent1
      if (comps.size() == 2)
        `uvm_info("TEST", "find_all agent count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("find_all agent count: FAIL (got %0d)", comps.size()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("find_all_test");
endmodule
