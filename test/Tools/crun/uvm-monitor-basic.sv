// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_monitor with analysis_port and subscriber.

// CHECK: [TEST] subscriber received 3 items
// CHECK: [TEST] monitor observation: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class mon_item extends uvm_object;
    `uvm_object_utils(mon_item)
    int value;
    function new(string name = "mon_item");
      super.new(name);
    endfunction
  endclass

  class mon_sub extends uvm_subscriber #(mon_item);
    `uvm_component_utils(mon_sub)
    int received;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      received = 0;
    endfunction
    function void write(mon_item t);
      received++;
    endfunction
  endclass

  class basic_monitor extends uvm_monitor;
    `uvm_component_utils(basic_monitor)
    uvm_analysis_port #(mon_item) ap;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
    endfunction
    task run_phase(uvm_phase phase);
      mon_item item;
      for (int i = 0; i < 3; i++) begin
        item = mon_item::type_id::create($sformatf("obs_%0d", i));
        item.value = i;
        ap.write(item);
      end
    endtask
  endclass

  class mon_test extends uvm_test;
    `uvm_component_utils(mon_test)
    basic_monitor monitor;
    mon_sub sub;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      monitor = basic_monitor::type_id::create("monitor", this);
      sub = mon_sub::type_id::create("sub", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      monitor.ap.connect(sub.analysis_export);
    endfunction
    function void report_phase(uvm_phase phase);
      if (sub.received == 3)
        `uvm_info("TEST", "subscriber received 3 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("subscriber received %0d items", sub.received))
      if (sub.received > 0)
        `uvm_info("TEST", "monitor observation: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "monitor observation: FAIL")
    endfunction
  endclass

  initial run_test("mon_test");
endmodule
