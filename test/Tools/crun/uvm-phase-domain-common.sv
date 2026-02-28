// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_domain::get_common_domain() returns a valid handle
// and its name is "common".

// CHECK: [TEST] common domain not null: PASS
// CHECK: [TEST] common domain name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class domain_common_test extends uvm_test;
    `uvm_component_utils(domain_common_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_domain d;
      d = uvm_domain::get_common_domain();
      if (d != null)
        `uvm_info("TEST", "common domain not null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "common domain not null: FAIL")

      if (d.get_name() == "common")
        `uvm_info("TEST", "common domain name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("common domain name: FAIL (got %s)", d.get_name()))
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("domain_common_test");
endmodule
