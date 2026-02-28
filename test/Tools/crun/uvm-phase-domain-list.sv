// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test uvm_domain::get_common_domain() API query.
// Just verify the common domain exists and has a name.

// CHECK: [TEST] common domain exists: PASS
// CHECK: [TEST] common domain name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_domain_test extends uvm_test;
    `uvm_component_utils(probe_domain_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_domain common;
      phase.raise_objection(this);

      common = uvm_domain::get_common_domain();
      if (common != null)
        `uvm_info("TEST", "common domain exists: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "common domain exists: FAIL")

      if (common != null && common.get_name() == "common")
        `uvm_info("TEST", "common domain name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("common domain name: FAIL (got %s)",
                   common != null ? common.get_name() : "null"))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_domain_test");
endmodule
