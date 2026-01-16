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

  //===--------------------------------------------------------------------===//
  // Non-Consecutive Repeat
  //===--------------------------------------------------------------------===//

  // CHECK: ltl.non_consecutive_repeat %b, 1, 3 : i1
  %seq_noncon = sva.seq.non_consecutive_repeat %b, 1, 3 : i1

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
