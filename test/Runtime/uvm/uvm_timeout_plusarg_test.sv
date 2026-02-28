// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-llhd --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: not circt-sim %t.mlir --top uvm_timeout_plusarg_test --max-time=1000000000 +UVM_TIMEOUT=1,NO 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PH_TIMEOUT
// SIM: +UVM_TIMEOUT=1,NO
// SIM-NOT: NO_TIMEOUT

`include "uvm_macros.svh"
import uvm_pkg::*;

class timeout_test extends uvm_test;
  `uvm_component_utils(timeout_test)

  function new(string name = "timeout_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    #10;
    // This should not execute when +UVM_TIMEOUT is active.
    $display("NO_TIMEOUT");
    phase.drop_objection(this);
  endtask
endclass

module uvm_timeout_plusarg_test;
  initial run_test("timeout_test");
endmodule
