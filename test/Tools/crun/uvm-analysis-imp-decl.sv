// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_analysis_imp_decl for multiple analysis imp ports.

// CHECK: [TEST] write_port_a called 2 times
// CHECK: [TEST] write_port_b called 3 times
// CHECK: [TEST] dual imp: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  `uvm_analysis_imp_decl(_port_a)
  `uvm_analysis_imp_decl(_port_b)

  class aid_item extends uvm_object;
    `uvm_object_utils(aid_item)
    int value;
    function new(string name = "aid_item");
      super.new(name);
    endfunction
  endclass

  class dual_imp_comp extends uvm_component;
    `uvm_component_utils(dual_imp_comp)
    uvm_analysis_imp_port_a #(aid_item, dual_imp_comp) imp_a;
    uvm_analysis_imp_port_b #(aid_item, dual_imp_comp) imp_b;
    int count_a, count_b;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      count_a = 0;
      count_b = 0;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      imp_a = new("imp_a", this);
      imp_b = new("imp_b", this);
    endfunction
    function void write_port_a(aid_item t);
      count_a++;
    endfunction
    function void write_port_b(aid_item t);
      count_b++;
    endfunction
  endclass

  class aid_test extends uvm_test;
    `uvm_component_utils(aid_test)
    dual_imp_comp comp;
    uvm_analysis_port #(aid_item) ap_a;
    uvm_analysis_port #(aid_item) ap_b;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp = dual_imp_comp::type_id::create("comp", this);
      ap_a = new("ap_a", this);
      ap_b = new("ap_b", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      ap_a.connect(comp.imp_a);
      ap_b.connect(comp.imp_b);
    endfunction
    task run_phase(uvm_phase phase);
      aid_item item;
      phase.raise_objection(this);
      for (int i = 0; i < 2; i++) begin
        item = aid_item::type_id::create($sformatf("a_%0d", i));
        ap_a.write(item);
      end
      for (int i = 0; i < 3; i++) begin
        item = aid_item::type_id::create($sformatf("b_%0d", i));
        ap_b.write(item);
      end
      if (comp.count_a == 2)
        `uvm_info("TEST", "write_port_a called 2 times", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("write_port_a called %0d times", comp.count_a))
      if (comp.count_b == 3)
        `uvm_info("TEST", "write_port_b called 3 times", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("write_port_b called %0d times", comp.count_b))
      if (comp.count_a == 2 && comp.count_b == 3)
        `uvm_info("TEST", "dual imp: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "dual imp: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("aid_test");
endmodule
