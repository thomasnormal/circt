// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test config_db set/get with uvm_object type.
// Stores a custom object, retrieves it, and verifies fields.

// CHECK: [TEST] config_db object set/get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_cfg extends uvm_object;
    `uvm_object_utils(my_cfg)
    int value;
    string label;
    function new(string name = "my_cfg");
      super.new(name);
    endfunction
  endclass

  class config_db_obj_test extends uvm_test;
    `uvm_component_utils(config_db_obj_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_cfg cfg_in, cfg_out;
      bit ok;
      phase.raise_objection(this);

      cfg_in = my_cfg::type_id::create("cfg_in");
      cfg_in.value = 123;
      cfg_in.label = "test_label";

      uvm_config_db#(my_cfg)::set(this, "", "my_cfg", cfg_in);
      ok = uvm_config_db#(my_cfg)::get(this, "", "my_cfg", cfg_out);

      if (ok && cfg_out != null && cfg_out.value == 123 && cfg_out.label == "test_label")
        `uvm_info("TEST", "config_db object set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config_db object set/get: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("config_db_obj_test");
endmodule
