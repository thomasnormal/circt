// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like | FileCheck %s --check-prefix=CHECK-LTL
// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like --externalize-registers --lower-to-bmc="top-module=ltl_delay_property bound=10" | FileCheck %s --check-prefix=CHECK-BMC
// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like --externalize-registers --lower-to-bmc="top-module=ltl_eventually_property bound=10" | FileCheck %s --check-prefix=CHECK-EVENTUALLY
// RUN: circt-opt %s --lower-sva-to-ltl --lower-ltl-to-core --lower-clocked-assert-like --externalize-registers --lower-to-bmc="top-module=ltl_implication_property bound=10" | FileCheck %s --check-prefix=CHECK-IMPL

//===----------------------------------------------------------------------===//
// LTLToCore to BMC Integration Tests
//
// This test file verifies that the LTLToCore pass correctly converts temporal
// properties to register-based checking logic that can be consumed by BMC.
//
// The circt-bmc pipeline applies these passes in order:
// 1. lower-sva-to-ltl: Convert SVA assertions to LTL dialect
// 2. lower-ltl-to-core: Convert LTL sequences/properties to register-based logic
// 3. lower-clocked-assert-like: Lower clocked assertions to regular assertions
// 4. externalize-registers: Move registers to module interface
// 5. lower-to-bmc: Create BMC problem from HW module
//
// The key integration point is that LTLToCore creates:
// - NFA-based state machines for sequence matching
// - Registers (seq.compreg) to track NFA states across clock cycles
// - Safety checks (verif.assert) for each time step
// - Final checks (with bmc.final attribute) for liveness properties
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Simple Delay Property
//
// This tests ltl.delay conversion through LTLToCore.
// The property: a |-> ##1 b (non-overlapping implication)
// LTLToCore should create register-based logic to track the delay.
//===----------------------------------------------------------------------===//

// CHECK-LTL-LABEL: hw.module @ltl_delay_property
// LTLToCore creates NFA state registers for sequence matching
// CHECK-LTL-DAG: seq.compreg{{.*}}ltl_state
// CHECK-LTL-DAG: seq.compreg{{.*}}ltl_implication_seen
// Safety assertion for each cycle
// CHECK-LTL: verif.assert
// Final assertion with bmc.final attribute for liveness checking
// CHECK-LTL: verif.assert{{.*}}bmc.final

// CHECK-BMC-LABEL: func.func @ltl_delay_property
// CHECK-BMC: verif.bmc bound 20
// CHECK-BMC-SAME: num_regs
// The circuit region contains the lowered LTL logic
// CHECK-BMC: circuit
hw.module @ltl_delay_property(
  in %clk: !seq.clock,
  in %a: i1,
  in %b: i1,
  out out: i1
) {
  // The property: when 'a' is true, 'b' must be true in the next cycle
  // This is encoded as a |-> ##1 b, or equivalently !a || X(b)

  // LTL delay sequence: ##1 b
  %delay_seq = ltl.delay %b, 1, 0 : i1

  // Implication: a |-> ##1 b
  %impl = ltl.implication %a, %delay_seq : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %b : i1
}

//===----------------------------------------------------------------------===//
// Test 2: Eventually Property (Liveness)
//
// Tests ltl.eventually conversion. This is a liveness property that requires
// special handling in BMC - it needs to be checked at the final step.
// LTLToCore creates a bmc.final assertion for such properties.
//===----------------------------------------------------------------------===//

// CHECK-EVENTUALLY-LABEL: func.func @ltl_eventually_property
// CHECK-EVENTUALLY: verif.bmc bound 20
// Eventually properties create bmc.final assertions that are checked at end
// CHECK-EVENTUALLY: circuit
hw.module @ltl_eventually_property(
  in %clk: !seq.clock,
  in %start: i1,
  in %done: i1,
  out out: i1
) {
  // The property: start |-> eventually(done)
  // When start is true, done must eventually become true

  // Eventually property
  %eventually_done = ltl.eventually %done : i1

  // Implication
  %impl = ltl.implication %start, %eventually_done : i1, !ltl.property

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %done : i1
}

//===----------------------------------------------------------------------===//
// Test 3: Same-Cycle Implication Property
//
// Tests ltl.implication with same-cycle consequent.
// The property: a |-> b (overlapping implication, a implies b same cycle)
//===----------------------------------------------------------------------===//

// CHECK-IMPL-LABEL: func.func @ltl_implication_property
// Same-cycle implication should become: !a || b
// CHECK-IMPL: verif.bmc bound 20
// CHECK-IMPL: circuit
hw.module @ltl_implication_property(
  in %clk: !seq.clock,
  in %a: i1,
  in %b: i1,
  out out: i1
) {
  // Same-cycle implication: a |-> b
  // This should become: !a || b

  %impl = ltl.implication %a, %b : i1, i1

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %b : i1
}

//===----------------------------------------------------------------------===//
// Test 4: Sequence Concatenation
//
// Tests ltl.concat conversion. Concatenation joins sequences end-to-end.
// For instantaneous booleans, concat is equivalent to AND.
//===----------------------------------------------------------------------===//

hw.module @ltl_concat_property(
  in %clk: !seq.clock,
  in %a: i1,
  in %b: i1,
  in %c: i1,
  out out: i1
) {
  // Concatenation: a ##1 b ##1 c
  // This creates a 3-cycle sequence

  %delay_b = ltl.delay %b, 1, 0 : i1
  %delay_c = ltl.delay %c, 1, 0 : i1

  %concat = ltl.concat %a, %delay_b, %delay_c : i1, !ltl.sequence, !ltl.sequence

  // Clock the sequence
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %concat, posedge %from_clk : !ltl.sequence

  // Assert the clocked sequence
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.sequence

  hw.output %c : i1
}

//===----------------------------------------------------------------------===//
// Test 5: Repetition
//
// Tests ltl.repeat conversion. Repetition requires tracking consecutive matches.
//===----------------------------------------------------------------------===//

hw.module @ltl_repeat_property(
  in %clk: !seq.clock,
  in %valid: i1,
  in %check: i1,
  out out: i1
) {
  // Repetition: valid[*3]
  // valid must hold for exactly 3 consecutive cycles

  %repeat_seq = ltl.repeat %valid, 3, 0 : i1

  // Only check when check signal is high
  %impl = ltl.implication %check, %repeat_seq : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %valid : i1
}

//===----------------------------------------------------------------------===//
// Test 6: Until Property
//
// Tests ltl.until conversion. p until q means p holds continuously until q.
//===----------------------------------------------------------------------===//

hw.module @ltl_until_property(
  in %clk: !seq.clock,
  in %p: i1,
  in %q: i1,
  out out: i1
) {
  // Until: p until q
  // p must hold at each cycle until q becomes true

  %until = ltl.until %p, %q : i1, i1

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %until, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %q : i1
}

//===----------------------------------------------------------------------===//
// Test 7: Complex Property - Request/Grant Protocol
//
// A realistic verification example combining multiple LTL operators.
// Property: (req && !grant) |-> ##[1:3] grant
// When there's a pending request, grant must arrive within 1-3 cycles.
//===----------------------------------------------------------------------===//

hw.module @ltl_protocol_property(
  in %clk: !seq.clock,
  in %req: i1,
  in %grant: i1,
  out out: i1
) {
  // Antecedent: req && !grant (pending request)
  %true = hw.constant true
  %not_grant = comb.xor %grant, %true : i1
  %pending = comb.and %req, %not_grant : i1

  // Consequent: ##[1:3] grant (grant within 1-3 cycles)
  // This is modeled as: ##1 grant || ##2 grant || ##3 grant
  %delay1 = ltl.delay %grant, 1, 0 : i1
  %delay2 = ltl.delay %grant, 2, 0 : i1
  %delay3 = ltl.delay %grant, 3, 0 : i1

  %within_3_cycles = ltl.or %delay1, %delay2, %delay3 : !ltl.sequence, !ltl.sequence, !ltl.sequence

  // Implication
  %impl = ltl.implication %pending, %within_3_cycles : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %grant : i1
}

//===----------------------------------------------------------------------===//
// Test 8: First Match
//
// Tests ltl.first_match conversion. Selects earliest matching occurrence.
//===----------------------------------------------------------------------===//

hw.module @ltl_first_match_property(
  in %clk: !seq.clock,
  in %trigger: i1,
  in %data: i1,
  out out: i1
) {
  // First match of a bounded delay sequence
  %delay_seq = ltl.delay %data, 1, 2 : i1  // ##[1:3] data
  %first = ltl.first_match %delay_seq : !ltl.sequence

  // Implication with trigger
  %impl = ltl.implication %trigger, %first : i1, !ltl.sequence

  // Clock the property
  %from_clk = seq.from_clock %clk
  %clocked = ltl.clock %impl, posedge %from_clk : !ltl.property

  // Assert the clocked property
  verif.clocked_assert %clocked, posedge %from_clk : !ltl.property

  hw.output %data : i1
}
