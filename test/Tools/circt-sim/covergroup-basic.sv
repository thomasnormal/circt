// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test basic covergroup/coverpoint functionality.
// Verifies that coverage sampling works and the runtime tracks values.

module top;
  int x;

  covergroup cg;
    coverpoint x;
  endgroup

  cg cg_inst = new;

  initial begin
    // Sample some values
    x = 5;
    cg_inst.sample();
    x = 15;
    cg_inst.sample();
    x = 25;
    cg_inst.sample();

    // CHECK: sampled
    $display("sampled");

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

// CHECK: Coverage Report
// CHECK: Covergroup: cg
// CHECK: x: 3 hits
