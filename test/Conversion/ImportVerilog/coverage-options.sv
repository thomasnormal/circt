// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test coverage options support for covergroups, coverpoints, and crosses.
// Based on IEEE 1800-2017 Section 19.7 Coverage options.

module test_coverage_options;
  logic [7:0] data;
  logic [3:0] addr;
  logic clk;

  // CHECK: moore.covergroup.decl @cg_with_options
  // CHECK-SAME: goal<90>
  // CHECK-SAME: comment<"Main coverage group">
  // CHECK-SAME: per_instance
  covergroup cg_with_options @(posedge clk);
    option.per_instance = 1;
    option.goal = 90;
    option.comment = "Main coverage group";

    // CHECK: moore.coverpoint.decl @data_cp : !moore.l8
    // CHECK-SAME: comment<"Data coverage">
    // CHECK-SAME: at_least<5>
    data_cp: coverpoint data {
      option.comment = "Data coverage";
      option.at_least = 5;
      bins low = {[0:63]};
      bins mid = {[64:191]};
      bins high = {[192:255]};
    }

    // CHECK: moore.coverpoint.decl @addr_cp : !moore.l4
    // CHECK-SAME: weight<2>
    // CHECK-SAME: goal<95>
    // CHECK-SAME: comment<"Address coverage">
    addr_cp: coverpoint addr {
      option.weight = 2;
      option.goal = 95;
      option.comment = "Address coverage";
      bins all_values[] = {[0:15]};
    }

    // CHECK: moore.covercross.decl @data_x_addr
    // CHECK-SAME: comment<"Data x Address cross">
    data_x_addr: cross data_cp, addr_cp {
      option.comment = "Data x Address cross";
    }
  endgroup

  // Test type_option support
  // CHECK: moore.covergroup.decl @cg_type_options
  // CHECK-SAME: at_least<10>
  // CHECK-SAME: type_weight<3>
  // CHECK-SAME: type_goal<85>
  // CHECK-SAME: type_comment<"Type-level options test">
  covergroup cg_type_options @(posedge clk);
    type_option.weight = 3;
    type_option.goal = 85;
    type_option.comment = "Type-level options test";

    option.at_least = 10;

    // CHECK: moore.coverpoint.decl @simple_cp : !moore.l8
    simple_cp: coverpoint data;
  endgroup

  // Test multiple coverpoints with different options
  // CHECK: moore.covergroup.decl @cg_mixed_options
  // CHECK-SAME: weight<5>
  covergroup cg_mixed_options @(posedge clk);
    option.weight = 5;

    // CHECK: moore.coverpoint.decl @cp1 : !moore.l4
    // CHECK-SAME: weight<1>
    cp1: coverpoint data[3:0] {
      option.weight = 1;
      bins b1 = {[0:7]};
      bins b2 = {[8:15]};
    }

    // CHECK: moore.coverpoint.decl @cp2 : !moore.l4
    // CHECK-SAME: weight<3>
    // CHECK-SAME: at_least<2>
    cp2: coverpoint data[7:4] {
      option.weight = 3;
      option.at_least = 2;
      bins low = {[0:3]};
      bins high = {[12:15]};
    }

    // CHECK: moore.coverpoint.decl @cp3 : !moore.l4
    // CHECK-SAME: auto_bin_max<32>
    cp3: coverpoint addr {
      option.auto_bin_max = 32;
    }
  endgroup

  // Test coverpoint with iff condition (IEEE 1800-2017 Section 19.5)
  logic enable;
  // CHECK: moore.covergroup.decl @cg_with_iff
  covergroup cg_with_iff @(posedge clk);
    // CHECK: moore.coverpoint.decl @data_iff_cp : !moore.l8 iff<"enable">
    data_iff_cp: coverpoint data iff (enable) {
      bins low = {[0:127]};
      bins high = {[128:255]};
    }
  endgroup

  cg_with_options cg1 = new();
  cg_type_options cg2 = new();
  cg_mixed_options cg3 = new();
  cg_with_iff cg4 = new();

endmodule
