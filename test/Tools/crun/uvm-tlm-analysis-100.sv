// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test analysis_port with 10 subscribers, 10 items = 100 deliveries.

// CHECK: [TEST] all 10 subscribers got 10 items each: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_ap_sub extends uvm_subscriber #(int);
    `uvm_component_utils(edge_ap_sub)
    int count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      count = 0;
    endfunction

    function void write(int t);
      count++;
    endfunction
  endclass

  class edge_tlm_analysis_test extends uvm_test;
    `uvm_component_utils(edge_tlm_analysis_test)
    uvm_analysis_port #(int) ap;
    edge_ap_sub subs[10];

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
      for (int i = 0; i < 10; i++)
        subs[i] = edge_ap_sub::type_id::create($sformatf("sub_%0d", i), this);
    endfunction

    function void connect_phase(uvm_phase phase);
      for (int i = 0; i < 10; i++)
        ap.connect(subs[i].analysis_export);
    endfunction

    task run_phase(uvm_phase phase);
      int total;
      phase.raise_objection(this);

      // Write 10 items
      for (int i = 0; i < 10; i++)
        ap.write(i);

      // Check all subscribers
      total = 0;
      for (int i = 0; i < 10; i++)
        total += subs[i].count;

      if (total == 100)
        `uvm_info("TEST", "all 10 subscribers got 10 items each: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("total deliveries %0d != 100: FAIL", total))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_tlm_analysis_test");
endmodule
