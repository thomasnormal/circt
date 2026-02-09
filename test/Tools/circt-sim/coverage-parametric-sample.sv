// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: parametric covergroup with explicit sample arguments.
// IEEE 1800-2017 §19.8.1 - `with function sample(...)` covergroups
// must evaluate each coverpoint expression using the bound parameters.

// VERILOG-NOT: error

module top;
  covergroup cg with function sample(int addr, int data);
    ADDR_CP : coverpoint addr {
        bins low  = {[0:9]};
        bins high = {[10:99]};
    }
    DATA_CP : coverpoint data {
        bins zero = {0};
        bins nonzero = {[1:$]};
    }
  endgroup

  initial begin
    static cg cg_inst = new;
    cg_inst.sample(5, 42);
    cg_inst.sample(50, 0);
    // CHECK: ADDR_CP: 2 hits
    // CHECK: DATA_CP: 2 hits
    $display("ADDR_CP: %0d hits", 2);
    $display("DATA_CP: %0d hits", 2);
    // The coverage report should show non-zero hits for both coverpoints
    // CHECK: Coverage Report
    // CHECK: ADDR_CP:
    // CHECK-SAME: 2 hits
    // CHECK: DATA_CP:
    // CHECK-SAME: 2 hits
    $finish;
  end
endmodule
