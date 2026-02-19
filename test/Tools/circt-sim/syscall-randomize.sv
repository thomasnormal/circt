// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test randomize(), srandom, get_randstate, set_randstate
module top;
  class packet;
    rand int data;
    constraint c_range { data >= 0 && data < 256; }
  endclass

  initial begin
    packet p = new();
    int ok;
    int saved_data;
    string state;

    // Basic randomize
    ok = p.randomize();
    // CHECK: randomize_ok=1
    $display("randomize_ok=%0d", ok);
    // CHECK: data_in_range=1
    $display("data_in_range=%0d", (p.data >= 0) && (p.data < 256));

    // srandom — seed the RNG
    p.srandom(42);
    ok = p.randomize();
    // CHECK: randomize_seeded=1
    $display("randomize_seeded=%0d", ok);

    // get_randstate / set_randstate
    state = p.get_randstate();
    ok = p.randomize();
    saved_data = p.data;
    p.set_randstate(state);
    ok = p.randomize();
    // Same seed state should give same result
    // CHECK: randstate_restore=1
    $display("randstate_restore=%0d", p.data == saved_data);

    $finish;
  end
endmodule
