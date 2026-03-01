// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test `uvm_field_int alone (no array/enum/object macros).
// The int macro may work even if other field macros are broken.

// CHECK: [TEST] copy int fields: PASS
// CHECK: [TEST] compare equal: PASS
// CHECK: [TEST] compare not equal: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_int_item extends uvm_object;
    int addr;
    int data;
    int id;

    `uvm_object_utils_begin(probe_int_item)
      `uvm_field_int(addr, UVM_ALL_ON)
      `uvm_field_int(data, UVM_ALL_ON)
      `uvm_field_int(id, UVM_ALL_ON)
    `uvm_object_utils_end

    function new(string name = "probe_int_item");
      super.new(name);
    endfunction
  endclass

  class probe_int_test extends uvm_test;
    `uvm_component_utils(probe_int_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      probe_int_item a, b;
      phase.raise_objection(this);

      a = probe_int_item::type_id::create("a");
      b = probe_int_item::type_id::create("b");
      a.addr = 32'hDEAD;
      a.data = 32'hBEEF;
      a.id = 7;

      // Test copy
      b.copy(a);
      if (b.addr == 32'hDEAD && b.data == 32'hBEEF && b.id == 7)
        `uvm_info("TEST", "copy int fields: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "copy int fields: FAIL")

      // Test compare equal
      if (a.compare(b))
        `uvm_info("TEST", "compare equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare equal: FAIL")

      // Test compare not equal
      b.data = 32'hCAFE;
      if (!a.compare(b))
        `uvm_info("TEST", "compare not equal: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "compare not equal: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_int_test");
endmodule
