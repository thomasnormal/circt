// REQUIRES: z3
// RUN: circt-bmc --run-smtlib -b 1 --module m_cover_hit %s | FileCheck %s --check-prefix=COVER-HIT
// RUN: circt-bmc --run-smtlib -b 1 --module m_cover_miss %s | FileCheck %s --check-prefix=COVER-MISS
// RUN: circt-bmc --run-smtlib -b 1 --module m_assert_fail %s | FileCheck %s --check-prefix=ASSERT-FAIL

// Mixed assert+cover BMC should be supported:
// - cover hit should produce SAT even when assertions hold
// - no cover hit and no assertion violation should be UNSAT
// - assertion violation should still produce SAT even when cover misses
//
// COVER-HIT: BMC_RESULT=SAT
// COVER-MISS: BMC_RESULT=UNSAT
// ASSERT-FAIL: BMC_RESULT=SAT

module {
  hw.module @m_cover_hit() {
    %true = hw.constant true
    verif.assert %true : i1
    verif.cover %true : i1
    hw.output
  }

  hw.module @m_cover_miss() {
    %true = hw.constant true
    %false = hw.constant false
    verif.assert %true : i1
    verif.cover %false : i1
    hw.output
  }

  hw.module @m_assert_fail() {
    %false = hw.constant false
    verif.assert %false : i1
    verif.cover %false : i1
    hw.output
  }
}
