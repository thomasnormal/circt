// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test for sequence event control: @seq
module SequenceEventControl;
  logic clk, a, b, c;

  sequence seq;
    @(posedge clk) a ##1 b ##1 c;
  endsequence

  // CHECK-LABEL: moore.procedure initial
  initial begin
    // CHECK: cf.br ^[[LOOP:[a-z0-9]+]](%{{.*}} : i1
    // CHECK: ^[[LOOP]](%{{.*}}: i1
    // CHECK: moore.wait_event
    // CHECK: moore.detect_event posedge
    // CHECK: comb.and
    // CHECK: comb.or
    // CHECK: cf.cond_br %{{.*}}, ^[[RESUME:[a-z0-9]+]], ^[[LOOP]]
    // CHECK: ^[[RESUME]]:
    @seq;
    a = 0;
  end
endmodule
