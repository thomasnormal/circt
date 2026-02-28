// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test config_db scalability: set and retrieve 20 different keys.

// CHECK: [TEST] all 20 keys retrieved: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_config_multi_test extends uvm_test;
    `uvm_component_utils(edge_config_multi_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      int pass_count;
      string key;

      phase.raise_objection(this);

      // Set 20 keys
      for (int i = 0; i < 20; i++) begin
        key = $sformatf("key_%0d", i);
        uvm_config_db#(int)::set(this, "", key, i * 10 + 7);
      end

      // Retrieve all 20
      pass_count = 0;
      for (int i = 0; i < 20; i++) begin
        key = $sformatf("key_%0d", i);
        ok = uvm_config_db#(int)::get(this, "", key, val);
        if (ok && val == i * 10 + 7)
          pass_count++;
        else
          `uvm_error("TEST", $sformatf("key_%0d: FAIL (ok=%0b val=%0d exp=%0d)", i, ok, val, i*10+7))
      end

      if (pass_count == 20)
        `uvm_info("TEST", "all 20 keys retrieved: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("only %0d/20 keys: FAIL", pass_count))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_config_multi_test");
endmodule
