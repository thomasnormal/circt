// RUN: circt-opt --lower-sva-to-ltl %s | FileCheck %s

// Test conversions for SVA sequence operations to LTL sequence operations.

hw.module @test_sequence_ops(in %clk: i1, in %a: i1, in %b: i1, in %c: i1) {

  //===--------------------------------------------------------------------===//
  // Sequence Delay
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.delay %a, 2 : i1
  %seq_delay = sva.seq.delay %a, 2 : i1

  // CHECK: ltl.delay %b, 1, 3 : i1
  %seq_delay_range = sva.seq.delay %b, 1, 3 : i1

  //===--------------------------------------------------------------------===//
  // Sequence Repeat
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.repeat %a, 3 : i1
  %seq_repeat = sva.seq.repeat %a, 3 : i1

  // CHECK: ltl.repeat %b, 1, 5 : i1
  %seq_repeat_range = sva.seq.repeat %b, 1, 5 : i1

  //===--------------------------------------------------------------------===//
  // Goto Repeat
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.goto_repeat %a, 2, 4 : i1
  %seq_goto = sva.seq.goto_repeat %a, 2, 4 : i1
  // CHECK: ltl.goto_repeat %a, 2 : i1
  %seq_goto_unbounded = sva.seq.goto_repeat %a, 2 : i1

  //===--------------------------------------------------------------------===//
  // Non-Consecutive Repeat
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.non_consecutive_repeat %b, 1, 3 : i1
  %seq_noncon = sva.seq.non_consecutive_repeat %b, 1, 3 : i1
  // CHECK: ltl.non_consecutive_repeat %b, 1 : i1
  %seq_noncon_unbounded = sva.seq.non_consecutive_repeat %b, 1 : i1

  //===--------------------------------------------------------------------===//
  // Sequence First Match
  //===--------------------------------------------------------------------===//

  // CHECK: [[DELAY:%[a-z0-9]+]] = ltl.delay %a, 2 : i1
  // CHECK: ltl.first_match [[DELAY]] : !ltl.sequence
  %seq0 = sva.seq.delay %a, 2 : i1
  %seq_first = sva.seq.first_match %seq0 : !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Concat
  //===--------------------------------------------------------------------===//

  // CHECK: [[D1:%[a-z0-9]+]] = ltl.delay %a, 2 : i1
  // CHECK: [[D2:%[a-z0-9]+]] = ltl.repeat %b, 3 : i1
  // CHECK: ltl.concat [[D1]], [[D2]] : !ltl.sequence, !ltl.sequence
  %seq1 = sva.seq.delay %a, 2 : i1
  %seq2 = sva.seq.repeat %b, 3 : i1
  %seq_concat = sva.seq.concat %seq1, %seq2 : !sva.sequence, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Or
  //===--------------------------------------------------------------------===//

  // CHECK: [[D3:%[a-z0-9]+]] = ltl.delay %a, 1 : i1
  // CHECK: [[D4:%[a-z0-9]+]] = ltl.delay %b, 2 : i1
  // CHECK: ltl.or [[D3]], [[D4]] : !ltl.sequence, !ltl.sequence
  %seq3 = sva.seq.delay %a, 1 : i1
  %seq4 = sva.seq.delay %b, 2 : i1
  %seq_or = sva.seq.or %seq3, %seq4 : !sva.sequence, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence And
  //===--------------------------------------------------------------------===//

  // CHECK: [[D5:%[a-z0-9]+]] = ltl.delay %a, 1 : i1
  // CHECK: [[D6:%[a-z0-9]+]] = ltl.delay %b, 1 : i1
  // CHECK: ltl.and [[D5]], [[D6]] : !ltl.sequence, !ltl.sequence
  %seq5 = sva.seq.delay %a, 1 : i1
  %seq6 = sva.seq.delay %b, 1 : i1
  %seq_and = sva.seq.and %seq5, %seq6 : !sva.sequence, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Intersect
  //===--------------------------------------------------------------------===//

  // CHECK: [[D7:%[a-z0-9]+]] = ltl.delay %a, 3 : i1
  // CHECK: [[D8:%[a-z0-9]+]] = ltl.delay %b, 3 : i1
  // CHECK: ltl.intersect [[D7]], [[D8]] : !ltl.sequence, !ltl.sequence
  %seq7 = sva.seq.delay %a, 3 : i1
  %seq8 = sva.seq.delay %b, 3 : i1
  %seq_intersect = sva.seq.intersect %seq7, %seq8 : !sva.sequence, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Throughout
  //===--------------------------------------------------------------------===//

  // CHECK: [[THROUGH_SEQ:%[a-z0-9_]+]] = ltl.repeat %a, 0 : i1
  // CHECK: ltl.intersect [[THROUGH_SEQ]], [[THROUGH_BASE:%[a-z0-9_]+]] : !ltl.sequence, !ltl.sequence
  %seq_through_base = sva.seq.delay %b, 2 : i1
  %seq_through = sva.seq.throughout %a, %seq_through_base : i1, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Within
  //===--------------------------------------------------------------------===//

  // CHECK: [[WITH_TRUE:%[a-z0-9_]+]] = hw.constant true
  // CHECK: [[WITH_REP:%[a-z0-9_]+]] = ltl.repeat [[WITH_TRUE]], 0 : i1
  // CHECK: [[WITH_REP_DELAY:%[a-z0-9_]+]] = ltl.delay [[WITH_REP]], 1, 0 : !ltl.sequence
  // CHECK: [[WITH_INNER_DELAY:%[a-z0-9_]+]] = ltl.delay [[WITH_INNER:%[a-z0-9_]+]], 1, 0 : !ltl.sequence
  // CHECK: [[WITH_COMBINED:%[a-z0-9_]+]] = ltl.concat [[WITH_REP_DELAY]], [[WITH_INNER_DELAY]], [[WITH_TRUE]] : !ltl.sequence, !ltl.sequence, i1
  // CHECK: ltl.intersect [[WITH_COMBINED]], [[WITH_OUTER:%[a-z0-9_]+]] : !ltl.sequence, !ltl.sequence
  %seq_within_inner = sva.seq.delay %a, 1 : i1
  %seq_within_outer = sva.seq.delay %c, 2 : i1
  %seq_within = sva.seq.within %seq_within_inner, %seq_within_outer : !sva.sequence, !sva.sequence

  //===--------------------------------------------------------------------===//
  // Sequence Matched / Triggered
  //===--------------------------------------------------------------------===//

  // CHECK: [[MATCH_SRC:%[a-z0-9_]+]] = ltl.delay %c, 0, 0 : i1
  // CHECK: ltl.matched [[MATCH_SRC]] : !ltl.sequence -> i1
  // CHECK: ltl.triggered [[MATCH_SRC]] : !ltl.sequence -> i1
  %seq_match_src = sva.seq.delay %c, 0, 0 : i1
  %seq_matched = sva.seq.matched %seq_match_src : !sva.sequence -> i1
  %seq_triggered = sva.seq.triggered %seq_match_src : !sva.sequence -> i1

  //===--------------------------------------------------------------------===//
  // Sequence Clock
  //===--------------------------------------------------------------------===//

  // CHECK: [[D9:%[a-z0-9]+]] = ltl.delay %a, 1 : i1
  // CHECK: ltl.clock [[D9]], posedge %clk : !ltl.sequence
  %seq9 = sva.seq.delay %a, 1 : i1
  %seq_clock_pos = sva.seq.clock %seq9, posedge %clk : !sva.sequence

  // CHECK: [[D10:%[a-z0-9]+]] = ltl.delay %b, 1 : i1
  // CHECK: ltl.clock [[D10]], negedge %clk : !ltl.sequence
  %seq10 = sva.seq.delay %b, 1 : i1
  %seq_clock_neg = sva.seq.clock %seq10, negedge %clk : !sva.sequence

  // CHECK: [[D11:%[a-z0-9]+]] = ltl.delay %c, 1 : i1
  // CHECK: ltl.clock [[D11]], edge %clk : !ltl.sequence
  %seq11 = sva.seq.delay %c, 1 : i1
  %seq_clock_both = sva.seq.clock %seq11, edge %clk : !sva.sequence

  hw.output
}
