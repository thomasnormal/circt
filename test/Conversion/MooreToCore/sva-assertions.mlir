// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Immediate Assertions (assert, assume, cover)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @ImmediateAssertions
// CHECK-SAME: (in %cond : !hw.struct<value: i1, unknown: i1>
moore.module @ImmediateAssertions(in %cond : !moore.l1, in %cond2 : !moore.l1) {
  moore.procedure always {
    // CHECK: %[[VAL1:.*]] = hw.struct_extract %cond["value"]
    // CHECK: verif.assert %[[VAL1]] label "assert_cond" : i1
    moore.assert immediate %cond label "assert_cond" : l1

    // CHECK: %[[VAL2:.*]] = hw.struct_extract %cond["value"]
    // CHECK: verif.assume %[[VAL2]] label "" : i1
    moore.assume immediate %cond : l1

    // CHECK: %[[VAL3:.*]] = hw.struct_extract %cond["value"]
    // CHECK: verif.cover %[[VAL3]] label "" : i1
    moore.cover immediate %cond : l1

    moore.return
  }
}

// CHECK-LABEL: hw.module @DeferredAssertions
// CHECK-SAME: (in %cond : !hw.struct<value: i1, unknown: i1>)
moore.module @DeferredAssertions(in %cond : !moore.l1) {
  moore.procedure always {
    // Observed deferred assertion (assert #0)
    // CHECK: %[[VAL1:.*]] = hw.struct_extract %cond["value"]
    // CHECK: verif.assert %[[VAL1]] label "" : i1
    moore.assert observed %cond : l1

  // Final deferred assertion (assert final)
  // CHECK: %[[VAL2:.*]] = hw.struct_extract %cond["value"]
  // CHECK: verif.assert %[[VAL2]] label "" {bmc.final} : i1
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
