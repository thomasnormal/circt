// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test `uvm_field_string alone. May work even if field_array doesn't.

// CHECK: [TEST] copy string fields: PASS
// CHECK: [TEST] compare equal: PASS
// CHECK: [TEST] compare not equal: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_str_item extends uvm_object;
    string name_f;
    string desc_f;

    `uvm_object_utils_begin(probe_str_item)
      `uvm_field_string(name_f, UVM_ALL_ON)
      `uvm_field_string(desc_f, UVM_ALL_ON)
    `uvm_object_utils_end

    function new(string name = "probe_str_item");
      super.new(name);
    endfunction
  endclass

  class probe_str_test extends uvm_test;
    `uvm_component_utils(probe_str_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      probe_str_item a, b;
      phase.raise_objection(this);

      a = probe_str_item::type_id::create("a");
      b = probe_str_item::type_id::create("b");
      a.name_f = "hello";
      a.desc_f = "world";

      // Test copy
      b.copy(a);
      if (b.name_f == "hello" && b.desc_f == "world")
        `uvm_info("TEST", "copy string fields: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("copy string fields: FAIL (got %s, %s)", b.name_f, b.desc_f))

      // Test compare equal
      if (a.compare(b))
        `uvm_info("TEST", "compare equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare equal: FAIL")

      // Test compare not equal
      b.desc_f = "changed";
      if (!a.compare(b))
        `uvm_info("TEST", "compare not equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare not equal: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_str_test");
endmodule
