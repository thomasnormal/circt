// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test uvm_mem creation, configure, get_size, get_n_bits.
// No map needed — just object creation and query API.

// CHECK: [TEST] mem create: PASS
// CHECK: [TEST] mem size: PASS
// CHECK: [TEST] mem n_bits: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_ral_mem_test extends uvm_test;
    `uvm_component_utils(probe_ral_mem_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_mem mem;
      phase.raise_objection(this);

      // Create a 256-entry, 32-bit wide memory
      mem = new("probe_mem", 256, 32);
      if (mem != null)
        `uvm_info("TEST", "mem create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "mem create: FAIL")

      if (mem.get_size() == 256)
        `uvm_info("TEST", "mem size: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("mem size: FAIL (got %0d)", mem.get_size()))

      if (mem.get_n_bits() == 32)
        `uvm_info("TEST", "mem n_bits: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("mem n_bits: FAIL (got %0d)", mem.get_n_bits()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_ral_mem_test");
endmodule
