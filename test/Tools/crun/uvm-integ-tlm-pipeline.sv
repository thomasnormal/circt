// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: multi-stage TLM pipeline A→B→C with transform.

// CHECK: [TEST] stage_c received 4 items
// CHECK: [TEST] stage_c values correct (doubled)
// CHECK: [TEST] tlm-pipeline: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_tlm_item extends uvm_object;
    `uvm_object_utils(integ_tlm_item)
    int value;
    function new(string name = "integ_tlm_item");
      super.new(name);
    endfunction
  endclass

  // Stage A: produces items
  class integ_tlm_stage_a extends uvm_component;
    `uvm_component_utils(integ_tlm_stage_a)
    uvm_analysis_port #(integ_tlm_item) ap;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
    endfunction
    function void produce(int val);
      integ_tlm_item item = integ_tlm_item::type_id::create("item");
      item.value = val;
      ap.write(item);
    endfunction
  endclass

  // Stage B: transforms (doubles) and forwards
  class integ_tlm_stage_b extends uvm_subscriber #(integ_tlm_item);
    `uvm_component_utils(integ_tlm_stage_b)
    uvm_analysis_port #(integ_tlm_item) ap_out;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap_out = new("ap_out", this);
    endfunction
    function void write(integ_tlm_item t);
      integ_tlm_item out = integ_tlm_item::type_id::create("out");
      out.value = t.value * 2;
      ap_out.write(out);
    endfunction
  endclass

  // Stage C: collects results
  class integ_tlm_stage_c extends uvm_subscriber #(integ_tlm_item);
    `uvm_component_utils(integ_tlm_stage_c)
    int values[$];
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void write(integ_tlm_item t);
      values.push_back(t.value);
    endfunction
  endclass

  class integ_tlm_test extends uvm_test;
    `uvm_component_utils(integ_tlm_test)
    integ_tlm_stage_a stage_a;
    integ_tlm_stage_b stage_b;
    integ_tlm_stage_c stage_c;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      stage_a = integ_tlm_stage_a::type_id::create("stage_a", this);
      stage_b = integ_tlm_stage_b::type_id::create("stage_b", this);
      stage_c = integ_tlm_stage_c::type_id::create("stage_c", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      stage_a.ap.connect(stage_b.analysis_export);
      stage_b.ap_out.connect(stage_c.analysis_export);
    endfunction
    task run_phase(uvm_phase phase);
      int pass;
      phase.raise_objection(this);
      for (int i = 1; i <= 4; i++)
        stage_a.produce(i * 5);
      pass = 1;
      if (stage_c.values.size() == 4)
        `uvm_info("TEST", "stage_c received 4 items", UVM_LOW)
      else begin
        `uvm_error("TEST", $sformatf("stage_c got %0d items", stage_c.values.size()))
        pass = 0;
      end
      // Expected: 10, 20, 30, 40
      if (stage_c.values.size() == 4 &&
          stage_c.values[0] == 10 && stage_c.values[1] == 20 &&
          stage_c.values[2] == 30 && stage_c.values[3] == 40) begin
        `uvm_info("TEST", "stage_c values correct (doubled)", UVM_LOW)
      end else begin
        `uvm_error("TEST", "stage_c values incorrect")
        pass = 0;
      end
      if (pass)
        `uvm_info("TEST", "tlm-pipeline: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "tlm-pipeline: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_tlm_test");
endmodule
