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
  // CHECK: moore.event_source_details =
  // CHECK-SAME: sequence_index = 0
  // CHECK-SAME: sequence_index = 1
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence[0]"
  // CHECK-SAME: "sequence[1]"
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
  // CHECK: moore.event_source_details =
  // CHECK-SAME: edge = "posedge"
  // CHECK-SAME: iff_name = "b"
  // CHECK-SAME: signal_name = "clk"
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence"
  // CHECK-SAME: "signal[0]:posedge:iff"
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
  // CHECK: moore.event_source_details =
  // CHECK-SAME: iff_name = "en1"
  // CHECK-SAME: iff_name = "en2"
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence[0]:iff"
  // CHECK-SAME: "sequence[1]:iff"
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
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence[0]"
  // CHECK-SAME: "sequence[1]"
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
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence"
  // CHECK-SAME: "signal[0]:posedge:iff"
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
  // CHECK: moore.event_source_details =
  // CHECK-SAME: edge = "both"
  // CHECK-SAME: signal_name = "b"
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence"
  // CHECK-SAME: "signal[0]:both"
  always @(seq or b) begin
    c <= ~c;
  end
endmodule

// Test mixed sequence/signal event list with expression-based signal/iff terms.
module SequenceSignalEventListExpr;
  logic clk, a, b, c, en, d;

  sequence seq;
    @(posedge clk) a;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListExpr
  // CHECK: moore.event_source_details =
  // CHECK-DAG: iff_expr = " (en | d)"
  // CHECK-DAG: signal_expr = " (b & c)"
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence"
  // CHECK-SAME: "signal[0]:posedge:iff"
  always @(seq or posedge (b & c) iff (en | d)) begin
    a <= ~a;
  end
endmodule

// Test mixed sequence/signal event list with select/reduction metadata that
// should be preserved structurally (not only as syntax text).
module SequenceSignalEventListStructuredExpr;
  logic clk, q;
  logic [3:0] bus;

  sequence seq;
    @(posedge clk) q;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListStructuredExpr
  // CHECK: moore.event_source_details =
  // CHECK-DAG: signal_name = "bus"
  // CHECK-DAG: signal_lsb = 2 : i32
  // CHECK-DAG: signal_msb = 2 : i32
  // CHECK-DAG: iff_name = "bus"
  // CHECK-DAG: iff_reduction = "and"
  // CHECK: moore.event_sources =
  // CHECK-SAME: "sequence"
  // CHECK-SAME: "signal[0]:posedge:iff"
  always @(seq or posedge bus[2] iff (&bus)) begin
    q <= ~q;
  end
endmodule

// Test mixed sequence/signal event list with inverted reduction metadata.
module SequenceSignalEventListStructuredInvertedReduction;
  logic clk, q;
  logic [3:0] bus;

  sequence seq;
    @(posedge clk) q;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListStructuredInvertedReduction
  // CHECK: moore.event_source_details =
  // CHECK-DAG: signal_name = "bus"
  // CHECK-DAG: signal_reduction = "xnor"
  // CHECK-DAG: iff_name = "bus"
  // CHECK-DAG: iff_reduction = "nor"
  always @(seq or posedge (^~bus) iff (~|bus)) begin
    q <= ~q;
  end
endmodule

// Test structured metadata for dynamic bit/part selects.
module SequenceSignalEventListStructuredDynamicSelect;
  logic clk, q;
  logic [7:0] bus;
  logic [2:0] i, j;

  sequence seq;
    @(posedge clk) q;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListStructuredDynamicSelect
  // CHECK: moore.event_source_details =
  // CHECK-DAG: signal_name = "bus"
  // CHECK-DAG: signal_dyn_index_name = "i"
  // CHECK-DAG: signal_dyn_sign = 1 : i32
  // CHECK-DAG: signal_dyn_offset = 0 : i32
  // CHECK-DAG: signal_dyn_width = 1 : i32
  // CHECK-DAG: iff_name = "bus"
  // CHECK-DAG: iff_dyn_index_name = "j"
  // CHECK-DAG: iff_dyn_sign = 1 : i32
  // CHECK-DAG: iff_dyn_offset = 0 : i32
  // CHECK-DAG: iff_dyn_width = 2 : i32
  always @(seq or posedge bus[i] iff bus[j +: 2]) begin
    q <= ~q;
  end
endmodule

// Test structured metadata for affine dynamic select expressions.
module SequenceSignalEventListStructuredDynamicSelectAffine;
  logic clk, q;
  logic [7:0] bus;
  logic [2:0] i, j;

  sequence seq;
    @(posedge clk) q;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListStructuredDynamicSelectAffine
  // CHECK: moore.event_source_details =
  // CHECK-DAG: signal_name = "bus"
  // CHECK-DAG: signal_dyn_index_name = "i"
  // CHECK-DAG: signal_dyn_sign = 1 : i32
  // CHECK-DAG: signal_dyn_offset = -1 : i32
  // CHECK-DAG: signal_dyn_width = 1 : i32
  // CHECK-DAG: iff_name = "bus"
  // CHECK-DAG: iff_dyn_index_name = "j"
  // CHECK-DAG: iff_dyn_sign = 1 : i32
  // CHECK-DAG: iff_dyn_offset = 1 : i32
  // CHECK-DAG: iff_dyn_width = 2 : i32
  always @(seq or posedge bus[i - 1] iff bus[(j + 1) +: 2]) begin
    q <= ~q;
  end
endmodule

// Test structured metadata for unary bitwise-not event terms.
module SequenceSignalEventListStructuredBitwiseNot;
  logic clk, q;
  logic [3:0] bus;

  sequence seq;
    @(posedge clk) q;
  endsequence

  // CHECK-LABEL: moore.module @SequenceSignalEventListStructuredBitwiseNot
  // CHECK: moore.event_source_details =
  // CHECK-DAG: signal_name = "bus"
  // CHECK-DAG: signal_lsb = 0 : i32
  // CHECK-DAG: signal_msb = 0 : i32
  // CHECK-DAG: signal_bitwise_not
  // CHECK-DAG: iff_name = "bus"
  // CHECK-DAG: iff_lsb = 2 : i32
  // CHECK-DAG: iff_msb = 2 : i32
  // CHECK-DAG: iff_bitwise_not
  always @(seq or posedge (~bus[0]) iff (~bus[2])) begin
    q <= ~q;
  end
endmodule
