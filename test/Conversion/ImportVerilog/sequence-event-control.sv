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

// Test event-list sequence controls: @(seq1 or seq2)
module SequenceEventListControl;
  logic clk, a, b;

  sequence seq1;
    @(posedge clk) a;
  endsequence

  sequence seq2;
    @(posedge clk) b;
  endsequence

  // CHECK-LABEL: moore.module @SequenceEventListControl
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: %[[OR:.+]] = comb.or
  // CHECK: comb.and bin %{{.+}}, %[[OR]]
  // CHECK: cf.cond_br
  always @(seq1 or seq2) begin
    a <= ~a;
  end
endmodule

// Test mixed sequence/signal event list on equivalent clock edge.
module SequenceSignalEventListControl;
  logic clk, a, b, c;

  sequence seq;
    @(posedge clk) a;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListControl
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.read %b
  // CHECK: comb.or
  // CHECK: cf.cond_br
  always @(seq or posedge clk iff b) begin
    c <= ~c;
  end
endmodule

// Test sequence event control with iff guard: @(seq iff en)
module SequenceEventControlWithIff;
  logic clk, a, en, b;

  sequence seq;
    @(posedge clk) a;
  endsequence

  // CHECK-LABEL: moore.module @SequenceEventControlWithIff
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.read %en
  // CHECK: comb.and
  // CHECK: cf.cond_br
  always @(seq iff en) begin
    b <= ~b;
  end
endmodule

// Test sequence event-list entries with iff guards.
module SequenceEventListControlWithIff;
  logic clk, a, b, en1, en2, c;

  sequence seq1;
    @(posedge clk) a;
  endsequence

  sequence seq2;
    @(posedge clk) b;
  endsequence

  // CHECK-LABEL: moore.module @SequenceEventListControlWithIff
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.read %en1
  // CHECK: moore.read %en2
  // CHECK: comb.or
  // CHECK: cf.cond_br
  always @(seq1 iff en1 or seq2 iff en2) begin
    c <= ~c;
  end
endmodule

// Test sequence event-list controls on different clocks: @(seq1 or seq2)
module SequenceEventListDifferentClocks;
  logic clk1, clk2, a, b, c;

  sequence seq1;
    @(posedge clk1) a;
  endsequence

  sequence seq2;
    @(posedge clk2) b;
  endsequence

  // CHECK-LABEL: moore.module @SequenceEventListDifferentClocks
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event any
  // CHECK-DAG: moore.detect_event any
  // CHECK: cf.cond_br
  always @(seq1 or seq2) begin
    c <= ~c;
  end
endmodule

// Test mixed sequence/signal event-list controls on different clocks.
module SequenceSignalEventListDifferentClocks;
  logic clk1, clk2, a, b, c;

  sequence seq;
    @(posedge clk1) a;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListDifferentClocks
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event any
  // CHECK-DAG: moore.detect_event any
  // CHECK: moore.read %b
  // CHECK: cf.cond_br
  always @(seq or posedge clk2 iff b) begin
    c <= ~c;
  end
endmodule

// Test sequence .triggered in procedural event controls.
module SequenceTriggeredMethodControl;
  logic clk, a, b, c;

  sequence seq;
    @(posedge clk) a ##1 b;
  endsequence

  // CHECK-LABEL: moore.module @SequenceTriggeredMethodControl
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: ltl.triggered
  // CHECK: moore.detect_event posedge
  always @(posedge seq.triggered) begin
    c <= ~c;
  end
endmodule

// Test mixed sequence/signal event list with no-edge signal event.
module SequenceSignalEventListNoEdge;
  logic clk, a, b, c;

  sequence seq;
    @(posedge clk) a;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListNoEdge
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event any
  // CHECK-DAG: moore.detect_event any
  // CHECK: cf.cond_br
  always @(seq or b) begin
    c <= ~c;
  end
endmodule
