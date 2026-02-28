// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_analysis_port broadcast to multiple uvm_subscriber instances.
// NOTE: uvm_tlm_analysis_fifo via analysis_port is known broken — not tested.

// CHECK: [TEST] subscriber A received 3 items
// CHECK: [TEST] subscriber B received 3 items
// CHECK: [TEST] data integrity: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class simple_item extends uvm_object;
    `uvm_object_utils(simple_item)
    int value;
    function new(string name = "simple_item");
      super.new(name);
    endfunction
  endclass

  class my_subscriber extends uvm_subscriber #(simple_item);
    `uvm_component_utils(my_subscriber)

    int received[$];

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void write(simple_item t);
      received.push_back(t.value);
    endfunction
  endclass

  class ap_test extends uvm_test;
    `uvm_component_utils(ap_test)

    uvm_analysis_port #(simple_item) ap;
    my_subscriber sub_a;
    my_subscriber sub_b;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap    = new("ap", this);
      sub_a = my_subscriber::type_id::create("sub_a", this);
      sub_b = my_subscriber::type_id::create("sub_b", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      ap.connect(sub_a.analysis_export);
      ap.connect(sub_b.analysis_export);
    endfunction

    task run_phase(uvm_phase phase);
      simple_item item;

      phase.raise_objection(this);

      // Broadcast 3 items to both subscribers
      for (int i = 0; i < 3; i++) begin
        item = simple_item::type_id::create($sformatf("item_%0d", i));
        item.value = (i + 1) * 10;
        ap.write(item);
      end

      // Verify subscriber A
      if (sub_a.received.size() == 3)
        `uvm_info("TEST", "subscriber A received 3 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("subscriber A received %0d items", sub_a.received.size()))

      // Verify subscriber B
      if (sub_b.received.size() == 3)
        `uvm_info("TEST", "subscriber B received 3 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("subscriber B received %0d items", sub_b.received.size()))

      // Check data integrity
      if (sub_a.received[0] == 10 && sub_a.received[1] == 20 && sub_a.received[2] == 30)
        `uvm_info("TEST", "data integrity: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "data integrity: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ap_test");
endmodule
