// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like | FileCheck %s --check-prefix=CHECK-NFA
// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like --externalize-registers --lower-to-bmc="top-module=nfa_sequence_match bound=5" | FileCheck %s --check-prefix=CHECK-BMC

//===----------------------------------------------------------------------===//
// LTLToCore NFA State Machine to BMC Integration Test
//
// This test verifies that LTLToCore's NFA-based sequence matching correctly
// integrates with BMC. The NFA builder creates state registers that track
// sequence matching progress across clock cycles.
//
// Key aspects being tested:
// 1. NFA state registers are created by LTLToCore
// 2. State registers are properly externalized by externalize-registers
// 3. BMC correctly handles the state tracking through its circuit region
// 4. bmc.final assertions are properly checked at the final step
//===----------------------------------------------------------------------===//

// CHECK-NFA-LABEL: hw.module @nfa_sequence_match
// LTLToCore creates NFA state registers for sequence matching
// Each state in the NFA has a corresponding compreg
// CHECK-NFA-DAG: seq.compreg{{.*}}ltl_state
// CHECK-NFA-DAG: seq.compreg{{.*}}ltl_implication_seen
// The safety check happens every cycle
// CHECK-NFA: verif.assert
// The liveness check (implication seen) has bmc.final attribute
// CHECK-NFA: verif.assert{{.*}}bmc.final

// CHECK-BMC-LABEL: func.func @nfa_sequence_match
// CHECK-BMC: verif.bmc bound 10 num_regs
// CHECK-BMC: init {
// CHECK-BMC: } loop {
// CHECK-BMC: } circuit {

hw.module @nfa_sequence_match(
  in %clk: !seq.clock,
  in %req: i1,
  in %ack: i1,
  out out: i1
) {
  // Property: req |-> ##[1:2] ack
  // When request is high, acknowledge must come within 1-2 cycles

  // Create delay variants for bounded delay
  %ack_d1 = ltl.delay %ack, 1, 0 : i1
  %ack_d2 = ltl.delay %ack, 2, 0 : i1

  // OR them together: ##1 ack || ##2 ack
  %ack_in_window = ltl.or %ack_d1, %ack_d2 : !ltl.sequence, !ltl.sequence

  // Create implication: req |-> (##1 ack || ##2 ack)
  %impl = ltl.implication %req, %ack_in_window : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %ack : i1
}

// Second module tests first_match which uses a different NFA construction
hw.module @nfa_first_match_test(
  in %clk: !seq.clock,
  in %start: i1,
  in %done: i1,
  out out: i1
) {
  // Property: start |-> first_match(##[1:3] done)
  // When start is high, wait for the FIRST occurrence of done within 1-3 cycles

  // Create bounded delay sequence
  %done_delayed = ltl.delay %done, 1, 2 : i1  // ##[1:3] done

  // First match selects earliest occurrence (deterministic matching)
  %first_done = ltl.first_match %done_delayed : !ltl.sequence

  // Create implication
  %impl = ltl.implication %start, %first_done : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %done : i1
}

// Third module tests repetition which requires consecutive cycle tracking
hw.module @nfa_repetition_test(
  in %clk: !seq.clock,
  in %valid: i1,
  in %commit: i1,
  out out: i1
) {
  // Property: commit |-> valid[*3]
  // When commit is high, valid must have been high for 3 consecutive cycles

  // Create repetition sequence
  %valid_3 = ltl.repeat %valid, 3, 0 : i1

  // Create implication
  %impl = ltl.implication %commit, %valid_3 : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %valid : i1
}
