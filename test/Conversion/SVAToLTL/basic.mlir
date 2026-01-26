// RUN: circt-opt --lower-sva-to-ltl %s | FileCheck %s

hw.module @test(in %clk: i1, in %req: i1, in %gnt: i1) {
  // Test sequence delay conversion
  // CHECK: ltl.delay %req, 2 : i1
  %seq0 = sva.seq.delay %req, 2 : i1

  // Test sequence repeat conversion
  // CHECK: ltl.repeat {{%.+}}, 3 : i1
  %seq1 = sva.seq.repeat %req, 3 : i1

  // Test sequence concat conversion
  // CHECK: ltl.concat {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
  %seq2 = sva.seq.concat %seq0, %seq1 : !sva.sequence, !sva.sequence

  // Test sequence or conversion
  // CHECK: ltl.or {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
  %seq3 = sva.seq.or %seq0, %seq1 : !sva.sequence, !sva.sequence

  // Test property not conversion
  // CHECK: ltl.not %req : i1
  %prop0 = sva.prop.not %req : i1

  // Test property implication conversion (overlapping)
  // CHECK: ltl.implication %req, %gnt : i1, i1
  %prop1 = sva.prop.implication %req, %gnt overlapping : i1, i1

  // Test property implication conversion (non-overlapping - delays antecedent)
  // CHECK: ltl.delay %req, 1, 0 : i1
  // CHECK: ltl.implication {{%.+}}, %gnt : !ltl.sequence, i1
  %prop2 = sva.prop.implication %req, %gnt : i1, i1

  // Test property eventually conversion
  // CHECK: ltl.eventually %gnt : i1
  %prop3 = sva.prop.eventually %gnt : i1

  // Test assert property conversion
  // CHECK: verif.assert {{%.+}} : !ltl.property
  sva.assert %prop0 : !sva.property

  // Test clocked assert property conversion
  // CHECK: verif.clocked_assert {{%.+}}, posedge %clk : !ltl.property
  sva.clocked_assert %prop0, posedge %clk : !sva.property

  // Test assume property conversion
  // CHECK: verif.assume %gnt : i1
  sva.assume %gnt : i1

  // Test cover property conversion
  // CHECK: verif.cover %req : i1
  sva.cover %req : i1

  hw.output
}
