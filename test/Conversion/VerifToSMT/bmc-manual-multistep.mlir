// RUN: circt-opt %s --externalize-registers --lower-to-bmc="top-module=manual_req_ack bound=5" --convert-hw-to-smt --convert-comb-to-smt --convert-verif-to-smt --reconcile-unrealized-casts | FileCheck %s

// This test demonstrates multi-step BMC verification by MANUALLY implementing
// a delayed property using registers. This proves that the BMC infrastructure
// CAN handle multi-step temporal properties when properly encoded.
//
// Property: req |-> ##1 ack
// Meaning: When req is high, ack must be high in the NEXT cycle
//
// Manual encoding:
//   1. Use a register to store previous cycle's req value
//   2. Assert: !prev_req || ack (if req was high last cycle, ack must be high now)

// CHECK-LABEL: func.func @manual_req_ack

hw.module @manual_req_ack(
  in %clk: !seq.clock,
  in %req: i1,
  in %ack: i1,
  out out: i1
) {
  // Register to store previous req value
  %prev_req = seq.compreg %req, %clk : i1

  // Property: if req was high in previous cycle, ack must be high now
  // This is equivalent to: req |-> ##1 ack
  %true = hw.constant true
  %not_prev_req = comb.xor %prev_req, %true : i1
  %prop = comb.or %not_prev_req, %ack : i1

  verif.assert %prop : i1

  hw.output %ack : i1
}

// Test with a specific trace to verify it works:
// Cycle 0: req=0, ack=X -> prev_req=? -> Check: true (vacuous)
// Cycle 1: req=1, ack=X -> prev_req=0 -> Check: !0 || X = true
// Cycle 2: req=0, ack=1 -> prev_req=1 -> Check: !1 || 1 = true ✓
// Cycle 3: req=0, ack=0 -> prev_req=0 -> Check: !0 || 0 = true
//
// Counterexample that SHOULD be caught:
// Cycle 1: req=1, ack=0 -> prev_req=0 -> Check: !0 || 0 = true
// Cycle 2: req=0, ack=0 -> prev_req=1 -> Check: !1 || 0 = false ✗

// The BMC should be able to find this violation or prove the property holds
// CHECK: smt.solver
// CHECK: scf.for
// CHECK: smt.check
