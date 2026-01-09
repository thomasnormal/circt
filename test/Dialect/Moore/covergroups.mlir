// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

// Test basic covergroup declaration
// CHECK-LABEL: moore.covergroup.decl @empty_cg {
// CHECK-NEXT: }
moore.covergroup.decl @empty_cg {
}

// Test covergroup with sample event
// CHECK-LABEL: moore.covergroup.decl @sampled_cg
// CHECK-SAME: sample_event<posedge> %clk : !moore.i1 {
// CHECK-NEXT: }
moore.module @top(in %clk: !moore.i1, in %data: !moore.i8, in %addr: !moore.i4) {
  moore.covergroup.decl @sampled_cg sample_event<posedge> %clk : !moore.i1 {
  }
  moore.output
}

// Test covergroup with coverpoints
// CHECK-LABEL: moore.covergroup.decl @cg_with_coverpoints {
// CHECK-NEXT:   moore.covergroup.coverpoint @state_cp
moore.module @test_coverpoints(in %state: !moore.i4, in %data: !moore.i8) {
  moore.covergroup.decl @cg_with_coverpoints {
    moore.covergroup.coverpoint @state_cp %state : !moore.i4 {
      // CHECK: moore.covergroup.bins "idle" values [0]
      moore.covergroup.bins "idle" values [0]
      // CHECK: moore.covergroup.bins "running" values [1, 2, 3]
      moore.covergroup.bins "running" values [1, 2, 3]
      // CHECK: moore.covergroup.bins "error" values [15] kind illegal
      moore.covergroup.bins "error" values [15] kind illegal
      // CHECK: moore.covergroup.bins "ignored" values [14] kind ignore
      moore.covergroup.bins "ignored" values [14] kind ignore
    }
    // CHECK: moore.covergroup.coverpoint @data_cp
    moore.covergroup.coverpoint @data_cp %data : !moore.i8 auto_bin_max 16 {
    }
  }
  moore.output
}

// Test range bins
// CHECK-LABEL: moore.covergroup.decl @cg_range_bins {
moore.module @test_range_bins(in %value: !moore.i16) {
  moore.covergroup.decl @cg_range_bins {
    moore.covergroup.coverpoint @value_cp %value : !moore.i16 {
      // CHECK: moore.covergroup.bins_range "low" from 0 to 100
      moore.covergroup.bins_range "low" from 0 to 100
      // CHECK: moore.covergroup.bins_range "mid" from 101 to 200 array
      moore.covergroup.bins_range "mid" from 101 to 200 array
      // CHECK: moore.covergroup.bins_range "high" from 201 to 255 kind illegal
      moore.covergroup.bins_range "high" from 201 to 255 kind illegal
    }
  }
  moore.output
}

// Test transition bins
// CHECK-LABEL: moore.covergroup.decl @cg_transitions {
moore.module @test_transitions(in %fsm_state: !moore.i4) {
  moore.covergroup.decl @cg_transitions {
    moore.covergroup.coverpoint @fsm_cp %fsm_state : !moore.i4 {
      // CHECK: moore.covergroup.transition_bins "startup" transitions {{\[}}{{\[}}0, 1, 2{{\]}}{{\]}}
      moore.covergroup.transition_bins "startup" transitions [[0, 1, 2]]
      // CHECK: moore.covergroup.transition_bins "reset_paths" transitions {{\[}}{{\[}}3, 0{{\]}}, {{\[}}2, 0{{\]}}{{\]}}
      moore.covergroup.transition_bins "reset_paths" transitions [[3, 0], [2, 0]]
    }
  }
  moore.output
}

// Test cross coverage
// CHECK-LABEL: moore.covergroup.decl @cg_cross {
moore.module @test_cross(in %addr: !moore.i4, in %data: !moore.i8) {
  moore.covergroup.decl @cg_cross {
    // CHECK: moore.covergroup.coverpoint @addr_cp
    moore.covergroup.coverpoint @addr_cp %addr : !moore.i4 {
    }
    // CHECK: moore.covergroup.coverpoint @data_cp
    moore.covergroup.coverpoint @data_cp %data : !moore.i8 {
    }
    // CHECK: moore.covergroup.cross @addr_data_cross coverpoints [@addr_cp, @data_cp]
    moore.covergroup.cross @addr_data_cross coverpoints [@addr_cp, @data_cp] {
    }
  }
  moore.output
}

// Test cross with explicit bins
// CHECK-LABEL: moore.covergroup.decl @cg_cross_bins {
moore.module @test_cross_bins(in %x: !moore.i4, in %y: !moore.i4) {
  moore.covergroup.decl @cg_cross_bins {
    moore.covergroup.coverpoint @x_cp %x : !moore.i4 {
    }
    moore.covergroup.coverpoint @y_cp %y : !moore.i4 {
    }
    moore.covergroup.cross @xy_cross coverpoints [@x_cp, @y_cp] {
      // CHECK: moore.covergroup.cross_bins "corners" select {{\[}}{{\[}}0, 0{{\]}}, {{\[}}0, 15{{\]}}, {{\[}}15, 0{{\]}}, {{\[}}15, 15{{\]}}{{\]}}
      moore.covergroup.cross_bins "corners" select [[0, 0], [0, 15], [15, 0], [15, 15]]
      // CHECK: moore.covergroup.cross_bins "illegal_combo" select {{\[}}{{\[}}7, 7{{\]}}{{\]}} kind illegal
      moore.covergroup.cross_bins "illegal_combo" select [[7, 7]] kind illegal
    }
  }
  moore.output
}

// Test covergroup instantiation and sampling
// CHECK-LABEL: moore.module @test_instance
moore.module @test_instance(in %clk: !moore.i1, in %data: !moore.i8) {
  moore.covergroup.decl @my_cg {
  }

  // CHECK: %cg = moore.covergroup.instance "cg1" @my_cg
  %cg = moore.covergroup.instance "cg1" @my_cg

  // CHECK: moore.covergroup.sample %cg
  moore.covergroup.sample %cg

  moore.output
}

// Test multiple sample event kinds
// CHECK-LABEL: moore.covergroup.decl @cg_negedge
// CHECK-SAME: sample_event<negedge>
moore.module @test_negedge(in %clk: !moore.i1) {
  moore.covergroup.decl @cg_negedge sample_event<negedge> %clk : !moore.i1 {
  }
  moore.output
}

// CHECK-LABEL: moore.covergroup.decl @cg_edge
// CHECK-SAME: sample_event<edge>
moore.module @test_edge(in %clk: !moore.i1) {
  moore.covergroup.decl @cg_edge sample_event<edge> %clk : !moore.i1 {
  }
  moore.output
}
