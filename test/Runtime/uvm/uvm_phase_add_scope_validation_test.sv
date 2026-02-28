// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-llhd --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: not circt-sim %t.mlir --top uvm_phase_add_scope_validation_test --max-time=1000000000 2>&1 | FileCheck %s --check-prefix=SIM

// SIM: PH_BAD_ADD
// SIM: cannot find with_phase 'sched_b' within node 'sched_a'

`include "uvm_macros.svh"
import uvm_pkg::*;

module uvm_phase_add_scope_validation_test;
  initial begin
    uvm_phase sched_a;
    uvm_phase sched_b;

    sched_a = new("sched_a", UVM_PHASE_SCHEDULE);
    sched_b = new("sched_b", UVM_PHASE_SCHEDULE);

    // Invalid: with_phase belongs to a different schedule graph.
    sched_a.add(uvm_build_phase::get(), .with_phase(sched_b));

    $display("NO_FATAL");
    $finish;
  end
endmodule
