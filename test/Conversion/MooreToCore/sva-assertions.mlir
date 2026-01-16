// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Immediate Assertions (assert, assume, cover)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @ImmediateAssertions
moore.module @ImmediateAssertions(in %cond : !moore.l1, in %cond2 : !moore.l1) {
  moore.procedure always {
    // CHECK: verif.assert %cond label "assert_cond" : i1
    moore.assert immediate %cond label "assert_cond" : l1

    // CHECK: verif.assume %cond label "" : i1
    moore.assume immediate %cond : l1

    // CHECK: verif.cover %cond label "" : i1
    moore.cover immediate %cond : l1

    moore.return
  }
}

// CHECK-LABEL: hw.module @DeferredAssertions
moore.module @DeferredAssertions(in %cond : !moore.l1) {
  moore.procedure always {
    // Observed deferred assertion (assert #0)
    // CHECK: verif.assert %cond label "" : i1
    moore.assert observed %cond : l1

    // Final deferred assertion (assert final)
    // CHECK: verif.assert %cond label "" : i1
    moore.assert final %cond : l1

    moore.return
  }
}

//===----------------------------------------------------------------------===//
// Concurrent Assertions with LTL Properties
//===----------------------------------------------------------------------===//

// Note: Concurrent assertions use verif.clocked_assert with LTL properties.
// The LTL dialect operations are preserved through MooreToCore.
// These operations are typically created during ImportVerilog when parsing
// SystemVerilog concurrent assertions like:
//   assert property(@(posedge clk) a |-> ##1 b);
//
// The MooreToCore pass primarily handles moore.assert/assume/cover operations.
// LTL and verif dialect operations are marked as legal and passed through.

