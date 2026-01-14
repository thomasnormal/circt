// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test basic covergroup parsing and conversion to Moore dialect ops.

module test_covergroups;
  logic [7:0] addr;
  logic [3:0] data;
  logic       valid;

  // CHECK: moore.covergroup.decl @cg1 {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8
  // CHECK: }
  covergroup cg1 @(posedge valid);
    coverpoint addr;
  endgroup

  // CHECK: moore.covergroup.decl @cg2 {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8
  // CHECK:   moore.coverpoint.decl @data_cp : !moore.l4
  // CHECK: }
  covergroup cg2 @(posedge valid);
    addr_cp: coverpoint addr;
    data_cp: coverpoint data;
  endgroup

  // CHECK: moore.covergroup.decl @cg3 {
  // CHECK:   moore.coverpoint.decl @a_cp : !moore.l8
  // CHECK:   moore.coverpoint.decl @d_cp : !moore.l4
  // CHECK:   moore.covercross.decl @axd targets [@a_cp, @d_cp]
  // CHECK: }
  covergroup cg3 @(posedge valid);
    a_cp: coverpoint addr;
    d_cp: coverpoint data;
    axd: cross a_cp, d_cp;
  endgroup

endmodule
