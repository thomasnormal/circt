// RUN: crun %s --uvm-path=%S/../../../lib/Runtime/uvm-core --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test 5-level deep component hierarchy and get_full_name().

// CHECK: [TEST] level 1 path: PASS
// CHECK: [TEST] level 2 path: PASS
// CHECK: [TEST] level 3 path: PASS
// CHECK: [TEST] level 4 path: PASS
// CHECK: [TEST] level 5 path: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_sub_comp5 extends uvm_component;
    `uvm_component_utils(edge_sub_comp5)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class edge_sub_comp4 extends uvm_component;
    `uvm_component_utils(edge_sub_comp4)
    edge_sub_comp5 child;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      child = edge_sub_comp5::type_id::create("sub5", this);
    endfunction
  endclass

  class edge_sub_comp3 extends uvm_component;
    `uvm_component_utils(edge_sub_comp3)
    edge_sub_comp4 child;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      child = edge_sub_comp4::type_id::create("sub4", this);
    endfunction
  endclass

  class edge_sub_comp2 extends uvm_component;
    `uvm_component_utils(edge_sub_comp2)
    edge_sub_comp3 child;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      child = edge_sub_comp3::type_id::create("sub3", this);
    endfunction
  endclass

  class edge_deep_hier_test extends uvm_test;
    `uvm_component_utils(edge_deep_hier_test)
    edge_sub_comp2 lvl2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      lvl2 = edge_sub_comp2::type_id::create("lvl2", this);
    endfunction

    task run_phase(uvm_phase phase);
      string p;
      phase.raise_objection(this);

      // Level 1: test
      p = get_full_name();
      if (p == "uvm_test_top")
        `uvm_info("TEST", "level 1 path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("level 1: FAIL (%s)", p))

      // Level 2
      p = lvl2.get_full_name();
      if (p == "uvm_test_top.lvl2")
        `uvm_info("TEST", "level 2 path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("level 2: FAIL (%s)", p))

      // Level 3
      p = lvl2.child.get_full_name();
      if (p == "uvm_test_top.lvl2.sub3")
        `uvm_info("TEST", "level 3 path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("level 3: FAIL (%s)", p))

      // Level 4
      p = lvl2.child.child.get_full_name();
      if (p == "uvm_test_top.lvl2.sub3.sub4")
        `uvm_info("TEST", "level 4 path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("level 4: FAIL (%s)", p))

      // Level 5
      p = lvl2.child.child.child.get_full_name();
      if (p == "uvm_test_top.lvl2.sub3.sub4.sub5")
        `uvm_info("TEST", "level 5 path: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("level 5: FAIL (%s)", p))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_deep_hier_test");
endmodule
