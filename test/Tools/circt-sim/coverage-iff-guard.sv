// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: coverpoint iff guard evaluation.
// IEEE 1800-2017 Section 19.5 - coverpoint iff condition must be evaluated
// at sample time, and sampling must be suppressed when the condition is false.

// VERILOG-NOT: error

module top;
  int x;
  bit valid;
  covergroup cg;
    cp_x : coverpoint x iff (valid);
  endgroup

  initial begin
    static cg cg_inst = new;
    valid = 0;
    x = 5;
    cg_inst.sample();  // Should NOT be sampled (valid=0)
    valid = 1;
    x = 10;
    cg_inst.sample();  // Should be sampled (valid=1)
    // CHECK: DONE
    $display("DONE");
    // The coverage report should show exactly 1 hit for cp_x
    // (the second sample where valid=1, x=10).
    // CHECK: Coverage Report
    // CHECK: cp_x:
    // CHECK-SAME: 1 hits
    $finish;
  end
endmodule
