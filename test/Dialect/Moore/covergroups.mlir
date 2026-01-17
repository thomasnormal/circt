// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

// Test basic covergroup declaration
// CHECK-LABEL: moore.covergroup.decl @empty_cg {
// CHECK-NEXT: }
moore.covergroup.decl @empty_cg {
}

// Test covergroup with coverpoint declarations
// CHECK-LABEL: moore.covergroup.decl @cg_with_coverpoints {
// CHECK-NEXT:   moore.coverpoint.decl @state_cp : i4 {
// CHECK-NEXT:   }
// CHECK-NEXT:   moore.coverpoint.decl @data_cp : i8 {
// CHECK-NEXT:   }
// CHECK-NEXT: }
moore.covergroup.decl @cg_with_coverpoints {
  moore.coverpoint.decl @state_cp : i4 {
  }
  moore.coverpoint.decl @data_cp : i8 {
  }
}

// Test covergroup with cross coverage declaration
// CHECK-LABEL: moore.covergroup.decl @cg_with_cross {
// CHECK-NEXT:   moore.coverpoint.decl @x_cp : i4 {
// CHECK-NEXT:   }
// CHECK-NEXT:   moore.coverpoint.decl @y_cp : i4 {
// CHECK-NEXT:   }
// CHECK-NEXT:   moore.covercross.decl @xy_cross targets [@x_cp, @y_cp] {
// CHECK-NEXT:   }
// CHECK-NEXT: }
moore.covergroup.decl @cg_with_cross {
  moore.coverpoint.decl @x_cp : i4 {
  }
  moore.coverpoint.decl @y_cp : i4 {
  }
  moore.covercross.decl @xy_cross targets [@x_cp, @y_cp] {
  }
}

// Test multiple covergroups
// CHECK-LABEL: moore.covergroup.decl @cg1 {
moore.covergroup.decl @cg1 {
  moore.coverpoint.decl @cp1 : i16 {
  }
}

// CHECK-LABEL: moore.covergroup.decl @cg2 {
moore.covergroup.decl @cg2 {
  moore.coverpoint.decl @cp2 : i32 {
  }
  moore.covercross.decl @cross1 targets [@cp2] {
  }
}

// Test cross coverage with bins using binsof/intersect
// CHECK-LABEL: moore.covergroup.decl @cg_with_cross_bins {
// CHECK-NEXT:   moore.coverpoint.decl @addr_cp : i8 {
// CHECK-NEXT:   }
// CHECK-NEXT:   moore.coverpoint.decl @cmd_cp : i4 {
// CHECK-NEXT:   }
// CHECK-NEXT:   moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp] {
// CHECK-NEXT:     moore.crossbin.decl @low_addr kind<bins> {
// CHECK-NEXT:       moore.binsof @addr_cp intersect [0, 1, 2, 3]
// CHECK-NEXT:     }
// CHECK-NEXT:     moore.crossbin.decl @ignore_zero kind<ignore_bins> {
// CHECK-NEXT:       moore.binsof @addr_cp intersect [0]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
moore.covergroup.decl @cg_with_cross_bins {
  moore.coverpoint.decl @addr_cp : i8 {
  }
  moore.coverpoint.decl @cmd_cp : i4 {
  }
  moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp] {
    moore.crossbin.decl @low_addr kind<bins> {
      moore.binsof @addr_cp intersect [0, 1, 2, 3]
    }
    moore.crossbin.decl @ignore_zero kind<ignore_bins> {
      moore.binsof @addr_cp intersect [0]
    }
  }
}

// Test covergroup instantiation
moore.covergroup.decl @inst_test_cg {
  moore.coverpoint.decl @value_cp : i8 {
  }
}

// CHECK-LABEL: func.func @test_covergroup_inst
func.func @test_covergroup_inst() {
  // CHECK: moore.covergroup.inst @inst_test_cg : <@inst_test_cg>
  %cg = moore.covergroup.inst @inst_test_cg : !moore.covergroup<@inst_test_cg>
  return
}

// Test covergroup sampling
moore.covergroup.decl @sample_test_cg {
  moore.coverpoint.decl @sample_cp1 : i8 {
  }
  moore.coverpoint.decl @sample_cp2 : i4 {
  }
}

// CHECK-LABEL: func.func @test_covergroup_sample
// CHECK-SAME: (%[[CG:.*]]: !moore.covergroup<@sample_test_cg>, %[[VAL1:.*]]: !moore.l8, %[[VAL2:.*]]: !moore.l4)
func.func @test_covergroup_sample(%cg: !moore.covergroup<@sample_test_cg>, %val1: !moore.l8, %val2: !moore.l4) {
  // CHECK: moore.covergroup.sample %[[CG]](%[[VAL1]], %[[VAL2]]) : <@sample_test_cg>(!moore.l8, !moore.l4)
  moore.covergroup.sample %cg(%val1, %val2) : !moore.covergroup<@sample_test_cg>(!moore.l8, !moore.l4)
  return
}

// Test covergroup get_coverage
moore.covergroup.decl @coverage_test_cg {
  moore.coverpoint.decl @coverage_cp : i16 {
  }
}

// CHECK-LABEL: func.func @test_covergroup_get_coverage
// CHECK-SAME: (%[[CG:.*]]: !moore.covergroup<@coverage_test_cg>)
func.func @test_covergroup_get_coverage(%cg: !moore.covergroup<@coverage_test_cg>) -> !moore.f64 {
  // CHECK: moore.covergroup.get_coverage %[[CG]] : <@coverage_test_cg> -> f64
  %coverage = moore.covergroup.get_coverage %cg : !moore.covergroup<@coverage_test_cg> -> !moore.f64
  // CHECK: return %{{.*}} : !moore.f64
  return %coverage : !moore.f64
}
