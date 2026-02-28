// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_root find and lookup methods.
// Verifies hierarchical component lookup.

// CHECK: [TEST] find driver: PASS
// CHECK: [TEST] lookup by path: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_driver extends uvm_component;
    `uvm_component_utils(my_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class my_agent extends uvm_component;
    `uvm_component_utils(my_agent)
    my_driver drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      drv = my_driver::type_id::create("drv", this);
    endfunction
  endclass

  class my_env extends uvm_env;
    `uvm_component_utils(my_env)
    my_agent agent;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent = my_agent::type_id::create("agent", this);
    endfunction
  endclass

  class root_find_test extends uvm_test;
    `uvm_component_utils(root_find_test)
    my_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = my_env::type_id::create("env", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_root root;
      uvm_component comp;
      uvm_component comps[$];
      phase.raise_objection(this);

      root = uvm_root::get();

      // Test 1: find
      root.find_all("*.drv", comps);
      if (comps.size() > 0)
        `uvm_info("TEST", "find driver: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "find driver: FAIL")

      // Test 2: lookup
      comp = root.lookup("uvm_test_top.env.agent.drv");
      if (comp != null)
        `uvm_info("TEST", "lookup by path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "lookup by path: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("root_find_test");
endmodule
