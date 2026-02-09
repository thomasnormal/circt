// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: get_randstate()/set_randstate() for per-object RNG state.
// IEEE 1800-2017 18.13

// VERILOG-NOT: error

class a;
    rand int x;
endclass

module top;
  initial begin
    automatic a obj = new;
    automatic string state;
    automatic int first_x;
    obj.srandom(42);
    // Capture state BEFORE randomize
    state = obj.get_randstate();
    obj.randomize();
    first_x = obj.x;
    // Restore the state captured before randomize
    obj.set_randstate(state);
    obj.randomize();
    // Should get same value since RNG state was restored to pre-randomize
    // CHECK: RANDSTATE match=1
    $display("RANDSTATE match=%0d", (obj.x == first_x) ? 1 : 0);
    $finish;
  end
endmodule
