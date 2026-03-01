// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory set_inst_override_by_type with hierarchical path.
// Overrides base_comp with derived_comp at a specific instance path.

// CHECK: [TEST] factory inst override type: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class base_comp extends uvm_component;
    `uvm_component_utils(base_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    virtual function string whoami();
      return "base_comp";
    endfunction
  endclass

  class derived_comp extends base_comp;
    `uvm_component_utils(derived_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    virtual function string whoami();
      return "derived_comp";
    endfunction
  endclass

  class factory_inst_test extends uvm_test;
    `uvm_component_utils(factory_inst_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      base_comp c;
      set_inst_override_by_type("child", base_comp::get_type(), derived_comp::get_type());
      c = base_comp::type_id::create("child", this);
      if (c.whoami() == "derived_comp")
        `uvm_info("TEST", "factory inst override type: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("factory inst override type: FAIL (got %s)", c.whoami()))
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("factory_inst_test");
endmodule
