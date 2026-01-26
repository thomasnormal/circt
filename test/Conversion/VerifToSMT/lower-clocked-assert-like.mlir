// RUN: circt-opt %s --lower-clocked-assert-like | FileCheck %s

// Test lowering of clocked_assert with i1 property to assert
// CHECK-LABEL: hw.module @test_clocked_assert
hw.module @test_clocked_assert(in %clock : i1, in %prop : i1, in %enable : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.assert %{{.*}} if %enable : !ltl.sequence
  verif.clocked_assert %prop if %enable, posedge %clock : i1
  hw.output
}

// -----

// Test lowering of clocked_assert without enable
// CHECK-LABEL: hw.module @test_clocked_assert_no_enable
hw.module @test_clocked_assert_no_enable(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.assert %{{.*}} : !ltl.sequence
  verif.clocked_assert %prop, posedge %clock : i1
  hw.output
}

// -----

// Test lowering of clocked_assert with label
// CHECK-LABEL: hw.module @test_clocked_assert_label
hw.module @test_clocked_assert_label(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.assert %{{.*}} label "my_assert" : !ltl.sequence
  verif.clocked_assert %prop, posedge %clock label "my_assert" : i1
  hw.output
}

// -----

// Test lowering of clocked_assume with i1 property to assume
// CHECK-LABEL: hw.module @test_clocked_assume
hw.module @test_clocked_assume(in %clock : i1, in %prop : i1, in %enable : i1) {
  // CHECK-NOT: verif.clocked_assume
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.assume %{{.*}} if %enable : !ltl.sequence
  verif.clocked_assume %prop if %enable, posedge %clock : i1
  hw.output
}

// -----

// Test lowering of clocked_assume without enable
// CHECK-LABEL: hw.module @test_clocked_assume_no_enable
hw.module @test_clocked_assume_no_enable(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_assume
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.assume %{{.*}} : !ltl.sequence
  verif.clocked_assume %prop, posedge %clock : i1
  hw.output
}

// -----

// Test lowering of clocked_cover with i1 property to cover
// CHECK-LABEL: hw.module @test_clocked_cover
hw.module @test_clocked_cover(in %clock : i1, in %prop : i1, in %enable : i1) {
  // CHECK-NOT: verif.clocked_cover
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.cover %{{.*}} if %enable : !ltl.sequence
  verif.clocked_cover %prop if %enable, posedge %clock : i1
  hw.output
}

// -----

// Test lowering of clocked_cover without enable
// CHECK-LABEL: hw.module @test_clocked_cover_no_enable
hw.module @test_clocked_cover_no_enable(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_cover
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: verif.cover %{{.*}} : !ltl.sequence
  verif.clocked_cover %prop, posedge %clock : i1
  hw.output
}

// -----

// Test that negedge clocked assertions are also lowered
// CHECK-LABEL: hw.module @test_negedge
hw.module @test_negedge(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK: ltl.clock %prop, negedge %clock
  // CHECK: verif.assert %{{.*}} : !ltl.sequence
  verif.clocked_assert %prop, negedge %clock : i1
  hw.output
}

// -----

// Test that edge (both edges) clocked assertions are also lowered
// CHECK-LABEL: hw.module @test_edge
hw.module @test_edge(in %clock : i1, in %prop : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK: ltl.clock %prop, posedge %clock
  // CHECK: ltl.clock %prop, negedge %clock
  // CHECK: ltl.or
  // CHECK: verif.assert %{{.*}} : !ltl.sequence
  verif.clocked_assert %prop, edge %clock : i1
  hw.output
}

// -----

// Test all three operations together
// CHECK-LABEL: hw.module @test_all_ops
hw.module @test_all_ops(in %clock : i1, in %prop : i1, in %enable : i1) {
  // CHECK-NOT: verif.clocked_assert
  // CHECK-NOT: verif.clocked_assume
  // CHECK-NOT: verif.clocked_cover
  // CHECK-DAG: ltl.clock %prop, posedge %clock
  // CHECK-DAG: verif.assert %{{.*}} if %enable : !ltl.sequence
  // CHECK-DAG: verif.assume %{{.*}} if %enable : !ltl.sequence
  // CHECK-DAG: verif.cover %{{.*}} if %enable : !ltl.sequence
  verif.clocked_assert %prop if %enable, posedge %clock : i1
  verif.clocked_assume %prop if %enable, posedge %clock : i1
  verif.clocked_cover %prop if %enable, posedge %clock : i1
  hw.output
}
