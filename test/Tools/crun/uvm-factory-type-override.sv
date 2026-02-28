// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test factory set_type_override_by_type.
// Verifies that creating base type returns derived instance after override.

// CHECK: [TEST] factory override: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class base_item extends uvm_object;
    `uvm_object_utils(base_item)
    function new(string name = "base_item");
      super.new(name);
    endfunction
    virtual function string whoami();
      return "base_item";
    endfunction
  endclass

  class derived_item extends base_item;
    `uvm_object_utils(derived_item)
    function new(string name = "derived_item");
      super.new(name);
    endfunction
    virtual function string whoami();
      return "derived_item";
    endfunction
  endclass

  class factory_test extends uvm_test;
    `uvm_component_utils(factory_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      base_item obj;
      uvm_factory factory;

      phase.raise_objection(this);

      factory = uvm_factory::get();
      factory.set_type_override_by_type(base_item::get_type(),
                                        derived_item::get_type());

      obj = base_item::type_id::create("obj");
      if (obj.whoami() == "derived_item")
        `uvm_info("TEST", "factory override: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("factory override: FAIL (got %s)", obj.whoami()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("factory_test");
endmodule
