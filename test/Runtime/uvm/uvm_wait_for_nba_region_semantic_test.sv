// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000000 +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PASS: waits through multi-#0 settling chain
// SIM: PASS: waits for NBA assignment visibility
// SIM: ALL TESTS PASSED
// SIM-NOT: UVM_ERROR

`timescale 1ns/1ps

import uvm_pkg::*;
`include "uvm_macros.svh"

class wait_for_nba_region_semantic_test extends uvm_test;
  `uvm_component_utils(wait_for_nba_region_semantic_test)

  int fail_count;
  int settle_chain;
  int nba_visible;

  function new(string name = "wait_for_nba_region_semantic_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void check_result(bit cond, string msg);
    if (cond)
      $display("PASS: %s", msg);
    else begin
      fail_count++;
      $display("FAIL: %s", msg);
    end
  endfunction

  virtual task run_phase(uvm_phase phase);
    phase.raise_objection(this);

    settle_chain = 0;
    fork
      begin
        #0;
        #0;
        settle_chain = 1;
      end
    join_none

    uvm_wait_for_nba_region();
    check_result(settle_chain == 1,
                 "waits through multi-#0 settling chain");

    nba_visible = 0;
    fork
      begin
        nba_visible <= 1;
      end
    join_none

    uvm_wait_for_nba_region();
    check_result(nba_visible == 1,
                 "waits for NBA assignment visibility");

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (fail_count == 0)
      $display("ALL TESTS PASSED");
    else
      `uvm_fatal("WAIT_NBA_SEM", $sformatf("Tests failed: %0d", fail_count))
  endfunction
endclass

module top;
  initial run_test("wait_for_nba_region_semantic_test");
endmodule
