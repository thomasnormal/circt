// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: override A→B then A→C. Second override should replace first.

// CHECK: [TEST] double override second wins: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_base_obj extends uvm_object;
    `uvm_object_utils(neg_base_obj)
    function new(string name = "neg_base_obj");
      super.new(name);
    endfunction
    virtual function string whoami();
      return "base";
    endfunction
  endclass

  class neg_ovr_b extends neg_base_obj;
    `uvm_object_utils(neg_ovr_b)
    function new(string name = "neg_ovr_b");
      super.new(name);
    endfunction
    virtual function string whoami();
      return "ovr_b";
    endfunction
  endclass

  class neg_ovr_c extends neg_base_obj;
    `uvm_object_utils(neg_ovr_c)
    function new(string name = "neg_ovr_c");
      super.new(name);
    endfunction
    virtual function string whoami();
      return "ovr_c";
    endfunction
  endclass

  class neg_factory_dbl_ovr_test extends uvm_test;
    `uvm_component_utils(neg_factory_dbl_ovr_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      neg_base_obj obj;
      phase.raise_objection(this);

      // First override: base → B
      neg_base_obj::type_id::set_type_override(neg_ovr_b::get_type());
      // Second override: base → C (should replace)
      neg_base_obj::type_id::set_type_override(neg_ovr_c::get_type());

      obj = neg_base_obj::type_id::create("test_obj");
      if (obj.whoami() == "ovr_c")
        `uvm_info("TEST", "double override second wins: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("double override: FAIL (got %s)", obj.whoami()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_factory_dbl_ovr_test");
endmodule
