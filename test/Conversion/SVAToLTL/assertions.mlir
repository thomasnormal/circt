// RUN: circt-opt --lower-sva-to-ltl %s | FileCheck %s

// Test conversions for SVA assertion directives to Verif operations.

hw.module @test_assertions(in %clk: i1, in %a: i1, in %b: i1, in %enable: i1) {

  //===--------------------------------------------------------------------===//
  // Assert Property
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT1:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.assert [[NOT1]] : !ltl.property
  %prop0 = sva.prop.not %a : i1
  sva.assert %prop0 : !sva.property

  //===--------------------------------------------------------------------===//
  // Assert Property with Label
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT2:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: verif.assert [[NOT2]] label "my_assert" : !ltl.property
  %prop1 = sva.prop.not %b : i1
  sva.assert %prop1 label "my_assert" : !sva.property

  //===--------------------------------------------------------------------===//
  // Assert Property with Enable
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT3:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.assert [[NOT3]] if %enable : !ltl.property
  %prop2 = sva.prop.not %a : i1
  sva.assert %prop2 if %enable : !sva.property

  //===--------------------------------------------------------------------===//
  // Assume Property
  //===--------------------------------------------------------------------===//

  // CHECK: verif.assume %a : i1
  sva.assume %a : i1

  // CHECK: verif.assume %b label "my_assume" : i1
  sva.assume %b label "my_assume" : i1

  // CHECK: verif.assume %a if %enable : i1
  sva.assume %a if %enable : i1

  //===--------------------------------------------------------------------===//
  // Cover Property
  //===--------------------------------------------------------------------===//

  // CHECK: verif.cover %a : i1
  sva.cover %a : i1

  // CHECK: verif.cover %b label "my_cover" : i1
  sva.cover %b label "my_cover" : i1

  // CHECK: verif.cover %a if %enable : i1
  sva.cover %a if %enable : i1

  //===--------------------------------------------------------------------===//
  // Clocked Assert Property
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT4:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_assert [[NOT4]], posedge %clk : !ltl.property
  %prop3 = sva.prop.not %a : i1
  sva.clocked_assert %prop3, posedge %clk : !sva.property

  // CHECK: [[NOT5:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: verif.clocked_assert [[NOT5]], negedge %clk : !ltl.property
  %prop4 = sva.prop.not %b : i1
  sva.clocked_assert %prop4, negedge %clk : !sva.property

  // CHECK: [[NOT6:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_assert [[NOT6]], edge %clk : !ltl.property
  %prop5 = sva.prop.not %a : i1
  sva.clocked_assert %prop5, edge %clk : !sva.property

  // CHECK: [[NOT7:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_assert [[NOT7]], posedge %clk label "clk_assert" : !ltl.property
  %prop6 = sva.prop.not %a : i1
  sva.clocked_assert %prop6, posedge %clk label "clk_assert" : !sva.property

  // CHECK: [[NOT8:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_assert [[NOT8]] if %enable, posedge %clk : !ltl.property
  %prop7 = sva.prop.not %a : i1
  sva.clocked_assert %prop7, posedge %clk if %enable : !sva.property

  //===--------------------------------------------------------------------===//
  // Clocked Assume Property
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT9:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_assume [[NOT9]], posedge %clk : !ltl.property
  %prop8 = sva.prop.not %a : i1
  sva.clocked_assume %prop8, posedge %clk : !sva.property

  // CHECK: [[NOT10:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: verif.clocked_assume [[NOT10]], negedge %clk : !ltl.property
  %prop9 = sva.prop.not %b : i1
  sva.clocked_assume %prop9, negedge %clk : !sva.property

  //===--------------------------------------------------------------------===//
  // Clocked Cover Property
  //===--------------------------------------------------------------------===//

  // CHECK: [[NOT11:%[a-z0-9]+]] = ltl.not %a : i1
  // CHECK: verif.clocked_cover [[NOT11]], posedge %clk : !ltl.property
  %prop10 = sva.prop.not %a : i1
  sva.clocked_cover %prop10, posedge %clk : !sva.property

  // CHECK: [[NOT12:%[a-z0-9]+]] = ltl.not %b : i1
  // CHECK: verif.clocked_cover [[NOT12]], negedge %clk : !ltl.property
  %prop11 = sva.prop.not %b : i1
  sva.clocked_cover %prop11, negedge %clk : !sva.property

  hw.output
}
