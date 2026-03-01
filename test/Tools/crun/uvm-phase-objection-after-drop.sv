// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test: raise a second objection after dropping the first, while a concurrent
// component still holds its own objection. Verifies that the phase stays alive
// when any objection remains, and ends only after all are dropped.

// CHECK: [TEST] first raise dropped
// CHECK: [TEST] second raise done
// CHECK: [TEST] re-raise after drop: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  // A helper component that holds an objection for a longer time,
  // ensuring the phase doesn't end prematurely when the test drops its first one.
  class holder_comp extends uvm_component;
    `uvm_component_utils(holder_comp)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      #50ns;
      phase.drop_objection(this);
    endtask
  endclass

  class reraise_test extends uvm_test;
    `uvm_component_utils(reraise_test)

    holder_comp holder;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      holder = holder_comp::type_id::create("holder", this);
    endfunction

    task run_phase(uvm_phase phase);
      // First raise and drop — holder's objection keeps phase alive
      phase.raise_objection(this);
      #10ns;
      phase.drop_objection(this);
      `uvm_info("TEST", "first raise dropped", UVM_LOW)

      // Re-raise after drop (phase still alive due to holder)
      #10ns;
      phase.raise_objection(this);
      #10ns;
      `uvm_info("TEST", "second raise done", UVM_LOW)
      phase.drop_objection(this);

      `uvm_info("TEST", "re-raise after drop: PASS", UVM_LOW)
    endtask
  endclass

  initial run_test("reraise_test");
endmodule
