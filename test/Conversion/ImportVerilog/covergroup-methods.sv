// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s
// REQUIRES: slang

// Test covergroup method call lowering (IEEE 1800-2017 Section 19.8)
// Verifies that sample() and get_coverage() are lowered to specialized Moore ops.

module covergroup_methods_test;
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // Define a covergroup
  covergroup cg @(posedge clk);
    coverpoint data;
    coverpoint addr;
  endgroup

  // Instantiate the covergroup
  cg cg_inst = new();

  real coverage_val;

  // Test sample() method in module context - should emit moore.covergroup.sample
  // CHECK: moore.covergroup.sample {{.*}} : <@cg>
  // Test get_coverage() method in module context - should emit moore.covergroup.get_coverage
  // CHECK: moore.covergroup.get_coverage {{.*}} : <@cg>
  initial begin
    cg_inst.sample();
    coverage_val = cg_inst.get_coverage();
  end

endmodule

// CHECK: moore.covergroup.decl @cg
// CHECK: moore.covergroup.decl @val_cg

// Test covergroup methods in a class context
class coverage_class;
  logic [7:0] value;
  logic [3:0] status;

  covergroup val_cg;
    coverpoint value;
    coverpoint status;
  endgroup

  function new();
    val_cg = new();
  endfunction

  // Method that samples the covergroup
  // CHECK: moore.covergroup.sample {{.*}} : <@val_cg>
  function void do_sample();
    val_cg.sample();
  endfunction

  // Method that gets coverage
  // CHECK: moore.covergroup.get_coverage {{.*}} : <@val_cg>
  function real get_cov();
    return val_cg.get_coverage();
  endfunction

endclass

module class_covergroup_test;
  coverage_class cc;

  initial begin
    cc = new();
    cc.do_sample();
    $display("Coverage: %f", cc.get_cov());
  end
endmodule
