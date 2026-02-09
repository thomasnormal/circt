// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: srandom() sets per-object seed for deterministic randomization.
// IEEE 1800-2017 18.13.2

// VERILOG-NOT: error

class a;
    rand int x;
endclass

module top;
  initial begin
    a obj1 = new;
    a obj2 = new;
    obj1.srandom(42);
    obj2.srandom(42);
    obj1.randomize();
    obj2.randomize();
    // Same seed should produce same random value
    // CHECK: SRANDOM same_seed_match=1
    $display("SRANDOM same_seed_match=%0d", (obj1.x == obj2.x) ? 1 : 0);
    $finish;
  end
endmodule
