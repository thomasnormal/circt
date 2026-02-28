// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: factory override + config_db parameter passing through env/agent.

// CHECK: [TEST] agent got timeout=42 from config_db
// CHECK: [TEST] agent type is integ_custom_agent (factory override)
// CHECK: [TEST] env-config-factory: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_base_agent extends uvm_component;
    `uvm_component_utils(integ_base_agent)
    int timeout_val;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(int)::get(this, "", "timeout", timeout_val))
        `uvm_error("TEST", "config_db get failed for timeout")
    endfunction
    virtual function string agent_type_name();
      return "integ_base_agent";
    endfunction
  endclass

  class integ_custom_agent extends integ_base_agent;
    `uvm_component_utils(integ_custom_agent)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    virtual function string agent_type_name();
      return "integ_custom_agent";
    endfunction
  endclass

  class integ_ecf_env extends uvm_env;
    `uvm_component_utils(integ_ecf_env)
    integ_base_agent agt;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agt = integ_base_agent::type_id::create("agt", this);
    endfunction
  endclass

  class integ_ecf_test extends uvm_test;
    `uvm_component_utils(integ_ecf_test)
    integ_ecf_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      integ_base_agent::type_id::set_type_override(integ_custom_agent::get_type());
      uvm_config_db#(int)::set(this, "env.agt", "timeout", 42);
      env = integ_ecf_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      int pass = 1;
      phase.raise_objection(this);
      if (env.agt.timeout_val == 42)
        `uvm_info("TEST", "agent got timeout=42 from config_db", UVM_LOW)
      else begin
        `uvm_error("TEST", $sformatf("agent timeout=%0d, expected 42", env.agt.timeout_val))
        pass = 0;
      end
      if (env.agt.agent_type_name() == "integ_custom_agent")
        `uvm_info("TEST", "agent type is integ_custom_agent (factory override)", UVM_LOW)
      else begin
        `uvm_error("TEST", {"agent type is ", env.agt.agent_type_name()})
        pass = 0;
      end
      if (pass)
        `uvm_info("TEST", "env-config-factory: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "env-config-factory: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_ecf_test");
endmodule
