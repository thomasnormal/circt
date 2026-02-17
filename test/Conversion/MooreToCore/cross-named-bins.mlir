// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test for MooreToCore lowering of cross coverage with named bins.
// This tests the lowering of cross coverage items including:
// - Cross coverage creation with __moore_cross_create
// - Named cross bins with __moore_cross_add_named_bin
// - Cross sampling with __moore_cross_sample
// - Negated binsof expressions (!binsof syntax)

// CHECK-DAG: llvm.mlir.global internal @__cg_handle_CrossCG
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cg_name_CrossCG("CrossCG
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cp_name_CrossCG_addr("addr
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cp_name_CrossCG_cmd("cmd
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__cross_name_CrossCG_addr_x_cmd("addr_x_cmd
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__crossbin_name_CrossCG_addr_x_cmd_low_addr("low_addr
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__crossbin_name_CrossCG_addr_x_cmd_ignore_zero("ignore_zero

// Globals for NegatedBinsCG (tests negate attribute)
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__crossbin_name_NegatedBinsCG_ab_not_zero("not_zero
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__crossbin_name_NegatedBinsCG_ab_not_high("not_high

// Globals for MixedNegateCG (tests mixed negate/non-negate)
// CHECK-DAG: llvm.mlir.global{{.*}}constant @__crossbin_name_MixedNegateCG_xy_edge_case("edge_case

// CHECK-DAG: llvm.func @__moore_covergroup_create(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @__moore_coverpoint_init(!llvm.ptr, i32, !llvm.ptr)
// CHECK-DAG: llvm.func @__moore_coverpoint_sample(!llvm.ptr, i32, i64)
// CHECK-DAG: llvm.func @__moore_cross_create(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__moore_cross_add_named_bin(!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__moore_cross_sample(!llvm.ptr, !llvm.ptr, i32)

// Covergroup declaration with cross coverage and named bins
moore.covergroup.decl @CrossCG {
  moore.coverpoint.decl @addr : !moore.i8 {}
  moore.coverpoint.decl @cmd : !moore.i4 {}
  moore.covercross.decl @addr_x_cmd targets [@addr, @cmd] {
    moore.crossbin.decl @low_addr kind<bins> {
      moore.binsof @addr intersect [0, 1, 2, 3]
    }
    moore.crossbin.decl @ignore_zero kind<ignore_bins> {
      moore.binsof @addr intersect [0]
      moore.binsof @cmd intersect [0]
    }
  }
}

// CHECK-LABEL: func @TestCrossInst
func.func @TestCrossInst() -> !moore.covergroup<@CrossCG> {
  // CHECK: llvm.call @__cg_init_CrossCG() : () -> ()
  // CHECK: [[HANDLE_PTR:%.+]] = llvm.mlir.addressof @__cg_handle_CrossCG : !llvm.ptr
  // CHECK: [[HANDLE:%.+]] = llvm.load [[HANDLE_PTR]] : !llvm.ptr -> !llvm.ptr
  // CHECK: return [[HANDLE]] : !llvm.ptr
  %cg = moore.covergroup.inst @CrossCG : !moore.covergroup<@CrossCG>
  return %cg : !moore.covergroup<@CrossCG>
}

// CHECK-LABEL: func @TestCrossSample
// CHECK-SAME: (%[[CG:.*]]: !llvm.ptr, %[[ADDR:.*]]: i8, %[[CMD:.*]]: i4)
func.func @TestCrossSample(%cg: !moore.covergroup<@CrossCG>, %addr: !moore.i8, %cmd: !moore.i4) {
  // CHECK: %[[ADDR_EXT:.*]] = arith.extui %[[ADDR]] : i8 to i64
  // CHECK: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.call @__moore_coverpoint_sample(%[[CG]], %[[IDX0]], %[[ADDR_EXT]]) : (!llvm.ptr, i32, i64) -> ()
  // CHECK: %[[CMD_EXT:.*]] = arith.extui %[[CMD]] : i4 to i64
  // CHECK: %[[IDX1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @__moore_coverpoint_sample(%[[CG]], %[[IDX1]], %[[CMD_EXT]]) : (!llvm.ptr, i32, i64) -> ()
  // CHECK: llvm.call @__moore_cross_sample(%[[CG]], {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> ()
  moore.covergroup.sample %cg(%addr, %cmd) : !moore.covergroup<@CrossCG> (!moore.i8, !moore.i4)
  return
}

// Test cross coverage without named bins (automatic bins)
moore.covergroup.decl @SimpleCrossCG {
  moore.coverpoint.decl @a : !moore.i4 {}
  moore.coverpoint.decl @b : !moore.i4 {}
  moore.covercross.decl @ab targets [@a, @b] {
  }
}

// CHECK-LABEL: func @TestSimpleCrossInst
func.func @TestSimpleCrossInst() -> !moore.covergroup<@SimpleCrossCG> {
  // CHECK: llvm.call @__cg_init_SimpleCrossCG() : () -> ()
  %cg = moore.covergroup.inst @SimpleCrossCG : !moore.covergroup<@SimpleCrossCG>
  return %cg : !moore.covergroup<@SimpleCrossCG>
}

// Test cross coverage with illegal bins
moore.covergroup.decl @IllegalBinsCG {
  moore.coverpoint.decl @x : !moore.i4 {}
  moore.coverpoint.decl @y : !moore.i4 {}
  moore.covercross.decl @xy targets [@x, @y] {
    moore.crossbin.decl @bad_combo kind<illegal_bins> {
      moore.binsof @x intersect [15]
      moore.binsof @y intersect [15]
    }
  }
}

// CHECK-LABEL: func @TestIllegalBinsCrossInst
func.func @TestIllegalBinsCrossInst() -> !moore.covergroup<@IllegalBinsCG> {
  // CHECK: llvm.call @__cg_init_IllegalBinsCG() : () -> ()
  %cg = moore.covergroup.inst @IllegalBinsCG : !moore.covergroup<@IllegalBinsCG>
  return %cg : !moore.covergroup<@IllegalBinsCG>
}

// Test three-way cross coverage
moore.covergroup.decl @TripleCrossCG {
  moore.coverpoint.decl @p1 : !moore.i4 {}
  moore.coverpoint.decl @p2 : !moore.i4 {}
  moore.coverpoint.decl @p3 : !moore.i4 {}
  moore.covercross.decl @triple targets [@p1, @p2, @p3] {
    moore.crossbin.decl @corner kind<bins> {
      moore.binsof @p1 intersect [0]
      moore.binsof @p2 intersect [0]
      moore.binsof @p3 intersect [0]
    }
  }
}

// CHECK-LABEL: func @TestTripleCrossInst
func.func @TestTripleCrossInst() -> !moore.covergroup<@TripleCrossCG> {
  // CHECK: llvm.call @__cg_init_TripleCrossCG() : () -> ()
  %cg = moore.covergroup.inst @TripleCrossCG : !moore.covergroup<@TripleCrossCG>
  return %cg : !moore.covergroup<@TripleCrossCG>
}

// Test multiple crosses in same covergroup
moore.covergroup.decl @MultiCrossCG {
  moore.coverpoint.decl @cp1 : !moore.i4 {}
  moore.coverpoint.decl @cp2 : !moore.i4 {}
  moore.coverpoint.decl @cp3 : !moore.i4 {}
  moore.covercross.decl @cross12 targets [@cp1, @cp2] {
  }
  moore.covercross.decl @cross23 targets [@cp2, @cp3] {
    moore.crossbin.decl @high kind<bins> {
      moore.binsof @cp2 intersect [15]
    }
  }
}

// CHECK-LABEL: func @TestMultiCrossInst
func.func @TestMultiCrossInst() -> !moore.covergroup<@MultiCrossCG> {
  // CHECK: llvm.call @__cg_init_MultiCrossCG() : () -> ()
  %cg = moore.covergroup.inst @MultiCrossCG : !moore.covergroup<@MultiCrossCG>
  return %cg : !moore.covergroup<@MultiCrossCG>
}

// Test cross coverage with negated binsof expressions (!binsof syntax)
// This tests the negate attribute on BinsOfOp
// The negate field is used in the MooreCrossBinsofFilter struct to invert matching.
// The globals for NegatedBinsCG are checked at the top of the file.
moore.covergroup.decl @NegatedBinsCG {
  moore.coverpoint.decl @a : !moore.i4 {}
  moore.coverpoint.decl @b : !moore.i4 {}
  moore.covercross.decl @ab targets [@a, @b] {
    // bins not_zero: match when a is NOT 0 (using negate)
    moore.crossbin.decl @not_zero kind<bins> {
      moore.binsof @a intersect [0] negate
    }
    // ignore_bins not_high: ignore when both are NOT in [12,13,14,15] (complex negation)
    moore.crossbin.decl @not_high kind<ignore_bins> {
      moore.binsof @a intersect [12, 13, 14, 15] negate
      moore.binsof @b intersect [12, 13, 14, 15] negate
    }
  }
}

// CHECK-LABEL: func @TestNegatedBinsInst
func.func @TestNegatedBinsInst() -> !moore.covergroup<@NegatedBinsCG> {
  // CHECK: llvm.call @__cg_init_NegatedBinsCG() : () -> ()
  %cg = moore.covergroup.inst @NegatedBinsCG : !moore.covergroup<@NegatedBinsCG>
  return %cg : !moore.covergroup<@NegatedBinsCG>
}

// Test cross coverage with mixed negated and non-negated binsof
// The globals for MixedNegateCG are checked at the top of the file.
moore.covergroup.decl @MixedNegateCG {
  moore.coverpoint.decl @x : !moore.i8 {}
  moore.coverpoint.decl @y : !moore.i8 {}
  moore.covercross.decl @xy targets [@x, @y] {
    // bins edge_case: x is in [0,1] AND y is NOT in [0,1]
    moore.crossbin.decl @edge_case kind<bins> {
      moore.binsof @x intersect [0, 1]
      moore.binsof @y intersect [0, 1] negate
    }
  }
}

// CHECK-LABEL: func @TestMixedNegateInst
func.func @TestMixedNegateInst() -> !moore.covergroup<@MixedNegateCG> {
  // CHECK: llvm.call @__cg_init_MixedNegateCG() : () -> ()
  %cg = moore.covergroup.inst @MixedNegateCG : !moore.covergroup<@MixedNegateCG>
  return %cg : !moore.covergroup<@MixedNegateCG>
}
