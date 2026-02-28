// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test config_db scope precedence: closest scope wins.

// CHECK: [TEST] child override wins: PASS
// CHECK: [TEST] null context vs test_top: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_child_comp extends uvm_component;
    `uvm_component_utils(edge_child_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      // Child sets same key — should override parent's value for child scope
      uvm_config_db#(int)::set(this, "", "scope_key", 200);
      ok = uvm_config_db#(int)::get(this, "", "scope_key", val);
      if (ok && val == 200)
        `uvm_info("TEST", "child override wins: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("child override wins: FAIL (val=%0d)", val))
    endtask
  endclass

  class edge_config_prec_test extends uvm_test;
    `uvm_component_utils(edge_config_prec_test)
    edge_child_comp child;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Parent sets key for child's scope
      uvm_config_db#(int)::set(this, "child", "scope_key", 100);
      child = edge_child_comp::type_id::create("child", this);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      phase.raise_objection(this);

      // Test null context vs uvm_test_top: set from null, then from this
      uvm_config_db#(int)::set(null, "uvm_test_top", "ctx_key", 10);
      uvm_config_db#(int)::set(this, "", "ctx_key", 20);
      ok = uvm_config_db#(int)::get(this, "", "ctx_key", val);
      if (ok && val == 20)
        `uvm_info("TEST", "null context vs test_top: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("null context vs test_top: FAIL (val=%0d)", val))

      #1;
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_config_prec_test");
endmodule
