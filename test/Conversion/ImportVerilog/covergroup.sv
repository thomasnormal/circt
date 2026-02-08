// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test basic covergroup parsing and conversion to Moore dialect ops.

module test_covergroups;
  logic [7:0] addr;
  logic [3:0] data;
  logic       valid;

  // CHECK: moore.covergroup.decl @cg1 sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
  // CHECK:   }
  // CHECK: }
  covergroup cg1 @(posedge valid);
    coverpoint addr;
  endgroup

  // CHECK: moore.covergroup.decl @cg2 sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4 {
  // CHECK:   }
  // CHECK: }
  covergroup cg2 @(posedge valid);
    addr_cp: coverpoint addr;
    data_cp: coverpoint data;
  endgroup

  // CHECK: moore.covergroup.decl @cg3 sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @a_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @d_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @axd targets [@a_cp, @d_cp]
  // CHECK: }
  covergroup cg3 @(posedge valid);
    a_cp: coverpoint addr;
    d_cp: coverpoint data;
    axd: cross a_cp, d_cp;
  endgroup

  // Test coverpoint iff with parentheses (standard form).
  // CHECK: moore.covergroup.decl @cg4 sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 iff<"valid"> {
  // CHECK:   }
  // CHECK: }
  covergroup cg4 @(posedge valid);
    coverpoint addr iff (valid);
  endgroup

  // Test coverpoint iff without parentheses (extension).
  // CHECK: moore.covergroup.decl @cg5 sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 iff<{{.*}}valid"> {
  // CHECK:   }
  // CHECK: }
  covergroup cg5 @(posedge valid);
    coverpoint addr iff valid;
  endgroup

endmodule
