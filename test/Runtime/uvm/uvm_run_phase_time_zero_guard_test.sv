// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-llhd --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top uvm_run_phase_time_zero_guard_test --max-time=1000000000 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: RUNPHSTIME
// SIM: The run phase must start at time 0

`include "uvm_macros.svh"
import uvm_pkg::*;

class run_phase_time_zero_guard_test extends uvm_test;
  `uvm_component_utils(run_phase_time_zero_guard_test)

  function new(string name = "run_phase_time_zero_guard_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module uvm_run_phase_time_zero_guard_test;
  initial begin
    #1;
    run_test("run_phase_time_zero_guard_test");
  end
endmodule
