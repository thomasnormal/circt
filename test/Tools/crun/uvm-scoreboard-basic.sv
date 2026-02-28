// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_scoreboard with analysis_imp.

// CHECK: [TEST] scoreboard collected 3 items
// CHECK: [TEST] scoreboard data check: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class sb_item extends uvm_object;
    `uvm_object_utils(sb_item)
    int value;
    function new(string name = "sb_item");
      super.new(name);
    endfunction
  endclass

  class basic_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(basic_scoreboard)
    uvm_analysis_imp #(sb_item, basic_scoreboard) ai;
    int collected[$];
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ai = new("ai", this);
    endfunction
    function void write(sb_item item);
      collected.push_back(item.value);
    endfunction
  endclass

  class sb_test extends uvm_test;
    `uvm_component_utils(sb_test)
    basic_scoreboard sb;
    uvm_analysis_port #(sb_item) ap;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sb = basic_scoreboard::type_id::create("sb", this);
      ap = new("ap", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      ap.connect(sb.ai);
    endfunction
    task run_phase(uvm_phase phase);
      sb_item item;
      phase.raise_objection(this);
      for (int i = 0; i < 3; i++) begin
        item = sb_item::type_id::create($sformatf("item_%0d", i));
        item.value = (i + 1) * 100;
        ap.write(item);
      end
      if (sb.collected.size() == 3)
        `uvm_info("TEST", "scoreboard collected 3 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("scoreboard collected %0d items", sb.collected.size()))
      if (sb.collected[0] == 100 && sb.collected[1] == 200 && sb.collected[2] == 300)
        `uvm_info("TEST", "scoreboard data check: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "scoreboard data check: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("sb_test");
endmodule
