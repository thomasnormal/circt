// RUN: crun %s --top uvm_path_tb --uvm-path %S/../../../lib/Runtime/uvm-core -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test the explicit --uvm-path flag.
// Uses the repo-bundled UVM source tree via a relative path from the test dir.
// This exercises the code path where uvmPath is explicitly set (bypassing the
// sourceMentionsUvm heuristic).

// CHECK: [TEST] explicit uvm-path works
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps

module uvm_path_tb;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  class uvm_path_test extends uvm_test;
    `uvm_component_utils(uvm_path_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("TEST", "explicit uvm-path works", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("uvm_path_test");
endmodule
