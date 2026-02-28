// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: stress test — create 100 objects via factory, verify all created.

// CHECK: [TEST] created 100 objects via factory
// CHECK: [TEST] all names unique
// CHECK: [TEST] stress-create: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_stress_obj extends uvm_object;
    `uvm_object_utils(integ_stress_obj)
    int id;
    function new(string name = "integ_stress_obj");
      super.new(name);
    endfunction
  endclass

  class integ_stress_test extends uvm_test;
    `uvm_component_utils(integ_stress_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      integ_stress_obj objs[100];
      int count;
      int all_unique;
      string names[string];
      phase.raise_objection(this);
      count = 0;
      all_unique = 1;
      for (int i = 0; i < 100; i++) begin
        string nm = $sformatf("obj_%0d", i);
        objs[i] = integ_stress_obj::type_id::create(nm);
        objs[i].id = i;
        if (objs[i] != null) count++;
        if (names.exists(nm)) all_unique = 0;
        names[nm] = nm;
      end
      if (count == 100)
        `uvm_info("TEST", "created 100 objects via factory", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("created only %0d objects", count))
      if (all_unique)
        `uvm_info("TEST", "all names unique", UVM_LOW)
      else
        `uvm_error("TEST", "duplicate names found")
      // Clear references
      for (int i = 0; i < 100; i++)
        objs[i] = null;
      if (count == 100 && all_unique)
        `uvm_info("TEST", "stress-create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "stress-create: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_stress_test");
endmodule
