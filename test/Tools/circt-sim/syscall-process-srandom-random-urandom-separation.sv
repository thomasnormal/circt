// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// xrun parity: process::srandom(seed) reseeds $urandom, but legacy $random
// continues on its own stream.
module top;
  process p;
  int r0, r1, u0, u1;

  initial begin
    p = process::self();

    p.srandom(1);
    r0 = $random;
    p.srandom(1);
    r1 = $random;

    // CHECK: random_not_reseeded=1
    $display("random_not_reseeded=%0d", r0 != r1);
    // CHECK: random_first=303379748
    $display("random_first=%0d", r0);
    // CHECK: random_second=-1064739199
    $display("random_second=%0d", r1);

    p.srandom(1);
    u0 = $urandom;
    p.srandom(1);
    u1 = $urandom;

    // CHECK: urandom_reseeded=1
    $display("urandom_reseeded=%0d", u0 == u1);
    // CHECK: urandom_seed1=72793
    $display("urandom_seed1=%0d", u0);

    $finish;
  end
endmodule
