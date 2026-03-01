// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: level 1 full name
// SIM: PASS: level 2 full name
// SIM: PASS: level 3 full name
// SIM: PASS: level 4 full name
// SIM: PASS: level 5 full name
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class level5_comp extends uvm_component;
  `uvm_component_utils(level5_comp)

  function new(string name = "level5_comp", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

class level4_comp extends uvm_component;
  `uvm_component_utils(level4_comp)
  level5_comp child;

  function new(string name = "level4_comp", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    child = level5_comp::type_id::create("sub5", this);
  endfunction
endclass

class level3_comp extends uvm_component;
  `uvm_component_utils(level3_comp)
  level4_comp child;

  function new(string name = "level3_comp", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    child = level4_comp::type_id::create("sub4", this);
  endfunction
endclass

class level2_comp extends uvm_component;
  `uvm_component_utils(level2_comp)
  level3_comp child;

  function new(string name = "level2_comp", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    child = level3_comp::type_id::create("sub3", this);
  endfunction
endclass

class component_full_name_semantic_test extends uvm_test;
  `uvm_component_utils(component_full_name_semantic_test)

  level2_comp lvl2;
  int fail_count;

  function new(string name = "component_full_name_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    lvl2 = level2_comp::type_id::create("lvl2", this);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond)
      $display("PASS: %s", msg);
    else begin
      fail_count++;
      $display("FAIL: %s", msg);
    end
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);

    check_result(get_full_name() == "uvm_test_top", "level 1 full name");
    check_result(lvl2.get_full_name() == "uvm_test_top.lvl2",
                 "level 2 full name");
    check_result(lvl2.child.get_full_name() == "uvm_test_top.lvl2.sub3",
                 "level 3 full name");
    check_result(lvl2.child.child.get_full_name() ==
                     "uvm_test_top.lvl2.sub3.sub4",
                 "level 4 full name");
    check_result(lvl2.child.child.child.get_full_name() ==
                     "uvm_test_top.lvl2.sub3.sub4.sub5",
                 "level 5 full name");

    phase.drop_objection(this);
  endtask

  function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("FULL_NAME_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("component_full_name_semantic_test");
endmodule
