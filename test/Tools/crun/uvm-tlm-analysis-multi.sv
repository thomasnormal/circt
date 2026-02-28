// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test multiple analysis ports with uvm_analysis_imp_decl.
// CHECK: [TEST] port_a write count: PASS
// CHECK: [TEST] port_b write count: PASS
// CHECK: [TEST] port_a data correct: PASS
// CHECK: [TEST] port_b data correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  `uvm_analysis_imp_decl(_from_a)
  `uvm_analysis_imp_decl(_from_b)

  class dual_sub extends uvm_component;
    `uvm_component_utils(dual_sub)
    uvm_analysis_imp_from_a #(int, dual_sub) imp_a;
    uvm_analysis_imp_from_b #(int, dual_sub) imp_b;
    int a_vals[$];
    int b_vals[$];
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      imp_a = new("imp_a", this);
      imp_b = new("imp_b", this);
    endfunction
    function void write_from_a(int t);
      a_vals.push_back(t);
    endfunction
    function void write_from_b(int t);
      b_vals.push_back(t);
    endfunction
  endclass

  class multi_ap_test extends uvm_test;
    `uvm_component_utils(multi_ap_test)
    uvm_analysis_port #(int) ap_a;
    uvm_analysis_port #(int) ap_b;
    dual_sub sub;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap_a = new("ap_a", this);
      ap_b = new("ap_b", this);
      sub = dual_sub::type_id::create("sub", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      ap_a.connect(sub.imp_a);
      ap_b.connect(sub.imp_b);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      ap_a.write(10);
      ap_a.write(20);
      ap_b.write(100);
      ap_b.write(200);
      ap_b.write(300);
      if (sub.a_vals.size() == 2)
        `uvm_info("TEST", "port_a write count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "port_a write count: FAIL")
      if (sub.b_vals.size() == 3)
        `uvm_info("TEST", "port_b write count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "port_b write count: FAIL")
      if (sub.a_vals[0] == 10 && sub.a_vals[1] == 20)
        `uvm_info("TEST", "port_a data correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "port_a data correct: FAIL")
      if (sub.b_vals[0] == 100 && sub.b_vals[1] == 200 && sub.b_vals[2] == 300)
        `uvm_info("TEST", "port_b data correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "port_b data correct: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("multi_ap_test");
endmodule
