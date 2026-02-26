// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test intra-assignment event controls on non-blocking assignments.
// These forms should lower through wait_event + nonblocking_assign.

// CHECK-LABEL: moore.module @NBAEventSignal
// CHECK: moore.procedure initial
// CHECK: moore.read %x
// CHECK: moore.wait_event {
// CHECK: moore.detect_event any
// CHECK: moore.nonblocking_assign %x
module NBAEventSignal;
  int x;
  bit y;
  initial x <= @y x;
endmodule

// CHECK-LABEL: moore.module @NBAEventPosedge
// CHECK: moore.procedure initial
// CHECK: moore.read %x
// CHECK: moore.wait_event {
// CHECK: moore.detect_event posedge
// CHECK: moore.nonblocking_assign %x
module NBAEventPosedge;
  logic clk;
  int x;
  initial x <= @(posedge clk) x;
endmodule

// CHECK-LABEL: moore.module @NBAEventList
// CHECK: moore.procedure initial
// CHECK: moore.read %x
// CHECK: moore.wait_event {
// CHECK: moore.detect_event any
// CHECK: moore.detect_event any
// CHECK: moore.nonblocking_assign %x
module NBAEventList;
  logic a;
  logic b;
  int x;
  initial x <= @(a or b) x;
endmodule

// CHECK-LABEL: moore.module @NBAEventRepeat
// CHECK: moore.procedure initial
// CHECK: moore.read %x
// CHECK: moore.wait_event {
// CHECK: moore.detect_event posedge
// CHECK: moore.nonblocking_assign %x
module NBAEventRepeat;
  logic clk;
  int x;
  initial x <= repeat (2) @(posedge clk) x;
endmodule
