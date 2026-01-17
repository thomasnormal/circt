// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

// Test covergroups declared inside classes (IEEE 1800-2017 Section 19.3)

// CHECK: moore.class.classdecl @basic_coverage {
// CHECK:   moore.class.propertydecl @data : !moore.l8
// CHECK:   moore.class.propertydecl @addr : !moore.l4
// CHECK:   moore.class.propertydecl @cg_basic : !moore.covergroup<@cg_basic>
// CHECK: }
// CHECK: moore.covergroup.decl @cg_basic {
// CHECK:   moore.coverpoint.decl @data_cp : !moore.l8 {
// CHECK:     moore.coverbin.decl @low kind<bins>
// CHECK:     moore.coverbin.decl @high kind<bins>
// CHECK:   }
// CHECK:   moore.coverpoint.decl @addr_cp : !moore.l4 {
// CHECK:   }
// CHECK: }
class basic_coverage;
  logic [7:0] data;
  logic [3:0] addr;

  // Covergroup declared inside a class
  covergroup cg_basic;
    data_cp: coverpoint data {
      bins low = {[0:127]};
      bins high = {[128:255]};
    }
    addr_cp: coverpoint addr;
  endgroup

endclass

// CHECK: moore.class.classdecl @coverage_with_event {
// CHECK:   moore.class.propertydecl @clk : !moore.l1
// CHECK:   moore.class.propertydecl @state : !moore.l4
// CHECK:   moore.class.propertydecl @state_cg : !moore.covergroup<@state_cg>
// CHECK: }
// CHECK: moore.covergroup.decl @state_cg sampling_event<"@(posedge clk)"> {
// CHECK:   moore.coverpoint.decl @state_cp : !moore.l4 {
// CHECK:     moore.coverbin.decl @idle kind<bins> values [0]
// CHECK:     moore.coverbin.decl @active kind<bins> values [1]
// CHECK:     moore.coverbin.decl @done kind<bins> values [2]
// CHECK:     moore.coverbin.decl @invalid kind<illegal_bins>
// CHECK:   }
// CHECK: }
class coverage_with_event;
  logic clk;
  logic [3:0] state;

  // Covergroup with sampling event
  covergroup state_cg @(posedge clk);
    state_cp: coverpoint state {
      bins idle = {0};
      bins active = {1};
      bins done = {2};
      illegal_bins invalid = {[3:15]};
    }
  endgroup

endclass

// CHECK: moore.class.classdecl @coverage_with_constructor {
// CHECK:   moore.class.propertydecl @value : !moore.l8
// CHECK:   moore.class.propertydecl @val_cg : !moore.covergroup<@val_cg>
// CHECK: }
// CHECK: moore.covergroup.decl @val_cg {
// CHECK:   moore.coverpoint.decl @value_cp : !moore.l8 {
// CHECK:   }
// CHECK: }
class coverage_with_constructor;
  logic [7:0] value;

  covergroup val_cg;
    coverpoint value;
  endgroup

endclass

// CHECK: moore.class.classdecl @coverage_with_cross {
// CHECK:   moore.class.propertydecl @opcode : !moore.l4
// CHECK:   moore.class.propertydecl @address : !moore.l8
// CHECK:   moore.class.propertydecl @trans_cg : !moore.covergroup<@trans_cg>
// CHECK: }
// CHECK: moore.covergroup.decl @trans_cg {
// CHECK:   moore.coverpoint.decl @op_cp : !moore.l4 {
// CHECK:   }
// CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
// CHECK:   }
// CHECK:   moore.covercross.decl @op_addr_cross targets [@op_cp, @addr_cp]
// CHECK: }
class coverage_with_cross;
  logic [3:0] opcode;
  logic [7:0] address;

  covergroup trans_cg;
    op_cp: coverpoint opcode;
    addr_cp: coverpoint address;
    op_addr_cross: cross op_cp, addr_cp;
  endgroup

endclass

module top;
endmodule
