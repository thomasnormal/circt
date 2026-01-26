// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test SVA sequence declarations and their usage in assertions

// CHECK-LABEL: moore.module @sva_sequence_test
module sva_sequence_test(
  input logic clk,
  input logic rst,
  input logic req,
  input logic ack,
  input logic data_valid,
  input logic data_ready
);

  // Basic sequence declaration with delay
  sequence req_ack_seq;
    req ##[1:4] ack;
  endsequence

  // Sequence with multiple signals
  sequence data_handshake_seq;
    data_valid ##1 data_ready;
  endsequence

  // Sequence with repetition
  sequence repeated_req_seq;
    req[*2:5];
  endsequence

  // Sequence with consecutive repetition
  sequence consecutive_ack_seq;
    ack[*3];
  endsequence

  // Sequence composition using another sequence
  sequence composite_seq;
    req_ack_seq ##1 data_handshake_seq;
  endsequence

  // Property using named sequence
  property p_req_ack;
    @(posedge clk) req |-> req_ack_seq;
  endproperty

  // Property with disable iff
  property p_req_ack_with_reset;
    @(posedge clk) disable iff (rst)
    req |-> req_ack_seq;
  endproperty

  // Property using sequence directly
  property p_data_handshake;
    @(posedge clk) data_valid |-> data_handshake_seq;
  endproperty

  // Assertions using named properties
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 0 : i1
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 3 : i1
  // CHECK-DAG: ltl.concat
  // CHECK-DAG: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_req_ack);

  // CHECK-DAG: ltl.or
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_req_ack_with_reset);

  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 0, 0 : i1
  // CHECK-DAG: ltl.concat
  // CHECK-DAG: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_data_handshake);

  // Direct assertion using sequence
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(@(posedge clk) req |-> req_ack_seq);

  // Cover property using sequence
  // CHECK: verif.{{(clocked_)?}}cover
  cover property(@(posedge clk) req_ack_seq);

  // Assume property using sequence
  // CHECK: verif.{{(clocked_)?}}assume
  assume property(@(posedge clk) rst |-> !req);

endmodule

// Test sequence with parameters (arguments)
// CHECK-LABEL: moore.module @sva_sequence_params_test
module sva_sequence_params_test(
  input logic clk,
  input logic a,
  input logic b,
  input logic c
);

  // Sequence with formal arguments
  sequence delay_seq(sig1, sig2, int delay_val);
    sig1 ##delay_val sig2;
  endsequence

  // Sequence with range argument
  sequence range_seq(sig, int min_delay, int max_delay);
    sig ##[min_delay:max_delay] sig;
  endsequence

  // Property using parameterized sequence
  property p_delay;
    @(posedge clk) a |-> delay_seq(a, b, 2);
  endproperty

  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 1, 0 : i1
  // CHECK-DAG: ltl.concat
  // CHECK-DAG: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_delay);

  // Direct use of parameterized sequence
  // CHECK-DAG: ltl.delay {{%[a-z0-9]+}}, 2, 0 : i1
  // CHECK-DAG: ltl.concat
  // CHECK-DAG: ltl.implication
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(@(posedge clk) a |-> delay_seq(a, c, 3));

endmodule

// Test sequence operators
// CHECK-LABEL: moore.module @sva_sequence_ops_test
module sva_sequence_ops_test(
  input logic clk,
  input logic x,
  input logic y,
  input logic z
);

  // Sequence with 'and' operator
  sequence and_seq;
    (x ##1 y) and (y ##1 z);
  endsequence

  // Sequence with 'or' operator
  sequence or_seq;
    (x ##1 y) or (y ##1 z);
  endsequence

  // Sequence with 'intersect' operator
  sequence intersect_seq;
    (x ##[1:3] y) intersect (y ##2 z);
  endsequence

  // Sequence with 'throughout'
  sequence throughout_seq;
    x throughout (y ##[1:4] z);
  endsequence

  // Sequence with 'within'
  sequence within_seq;
    (x ##1 y) within (z ##[1:5] z);
  endsequence

  // Sequence with first_match
  sequence first_match_seq;
    first_match(x ##[1:4] y);
  endsequence

  // Properties using the sequences
  property p_and;
    @(posedge clk) x |-> and_seq;
  endproperty

  property p_or;
    @(posedge clk) x |-> or_seq;
  endproperty

  property p_intersect;
    @(posedge clk) x |-> intersect_seq;
  endproperty

  property p_throughout;
    @(posedge clk) x |-> throughout_seq;
  endproperty

  property p_within;
    @(posedge clk) z |-> within_seq;
  endproperty

  property p_first_match;
    @(posedge clk) x |-> first_match_seq;
  endproperty

  // CHECK-DAG: ltl.and
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_and);

  // CHECK-DAG: ltl.or
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_or);

  // CHECK-DAG: ltl.intersect
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_intersect);

  // CHECK: ltl.repeat {{%[a-z0-9]+}}, 2, 3 : i1
  // CHECK: ltl.intersect
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_throughout);

  // CHECK: ltl.repeat {{%[a-z0-9]+}}, 0 : i1
  // CHECK: ltl.intersect
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_within);

  // CHECK: ltl.first_match
  // CHECK: verif.{{(clocked_)?}}assert
  assert property(p_first_match);

endmodule
