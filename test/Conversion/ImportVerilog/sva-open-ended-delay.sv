// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test open-ended sequence delay range ##[1:$].
module sva_open_ended_delay(
    input logic a,
    input logic b);
  sequence s;
    a ##[1:$] b;
  endsequence

  assert property (s);
endmodule

// CHECK-LABEL: moore.module @sva_open_ended_delay
// CHECK: ltl.delay {{.*}}, 1 : i1
// CHECK: ltl.concat
