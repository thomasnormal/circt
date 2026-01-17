// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test cross coverage support per IEEE 1800-2017 Section 19.6.
// Cross coverage measures the Cartesian product of multiple coverpoints.

module test_cross_coverage;
  logic clk;
  logic [7:0] addr;
  logic [3:0] cmd;
  logic       valid;
  logic [1:0] mode;

  // CHECK: moore.covergroup.decl @cg sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @cmd_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp]
  // CHECK: }
  covergroup cg @(posedge clk);
    coverpoint addr;
    coverpoint cmd;
    cross addr, cmd;  // Cross coverage of addr and cmd
  endgroup

  // CHECK: moore.covergroup.decl @cg_named sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @cmd_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @addr_cmd_cross targets [@addr_cp, @cmd_cp]
  // CHECK: }
  covergroup cg_named @(posedge clk);
    addr_cp: coverpoint addr;
    cmd_cp: coverpoint cmd;
    addr_cmd_cross: cross addr_cp, cmd_cp;  // Named cross coverage
  endgroup

  // CHECK: moore.covergroup.decl @cg_triple sampling_event<"@(posedge valid)"> {
  // CHECK:   moore.coverpoint.decl @a_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @c_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @m_cp : !moore.l2 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @triple_cross targets [@a_cp, @c_cp, @m_cp]
  // CHECK: }
  covergroup cg_triple @(posedge valid);
    a_cp: coverpoint addr;
    c_cp: coverpoint cmd;
    m_cp: coverpoint mode;
    triple_cross: cross a_cp, c_cp, m_cp;  // Three-way cross coverage
  endgroup

  // CHECK: moore.covergroup.decl @cg_multi_cross sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @cp1 : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @cp2 : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @cp3 : !moore.l2 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @cross1 targets [@cp1, @cp2]
  // CHECK:   moore.covercross.decl @cross2 targets [@cp2, @cp3]
  // CHECK: }
  covergroup cg_multi_cross @(posedge clk);
    cp1: coverpoint addr;
    cp2: coverpoint cmd;
    cp3: coverpoint mode;
    cross1: cross cp1, cp2;  // First cross
    cross2: cross cp2, cp3;  // Second cross using shared coverpoint
  endgroup

endmodule
