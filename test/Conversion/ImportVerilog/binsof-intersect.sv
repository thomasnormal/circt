// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test binsof/intersect semantics for cross coverage bins.
// This tests the infrastructure for SystemVerilog cross coverage bin selection.
// See IEEE 1800-2017 Section 19.6.1 for specification.

module test_binsof_intersect;
  logic clk;
  logic [7:0] addr;
  logic [3:0] cmd;

  // CHECK: moore.covergroup.decl @cg_with_cross_bins sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @cmd_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp] {
  // CHECK:     moore.crossbin.decl @low_addr kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @high_addr kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @ignore_zero kind<ignore_bins> {
  // CHECK:       moore.binsof @addr_cp intersect [0]
  // CHECK:       moore.binsof @cmd_cp intersect [0]
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  covergroup cg_with_cross_bins @(posedge clk);
    addr_cp: coverpoint addr;
    cmd_cp: coverpoint cmd;
    addr_x_cmd: cross addr_cp, cmd_cp {
      bins low_addr = binsof(addr_cp) intersect {[0:10]};
      bins high_addr = binsof(addr_cp) intersect {[245:255]};
      ignore_bins ignore_zero = binsof(addr_cp) intersect {0} && binsof(cmd_cp) intersect {0};
    }
  endgroup

  // CHECK: moore.covergroup.decl @cg_simple_cross sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @a_cp : !moore.l8 {
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @b_cp : !moore.l4 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @a_cp_x_b_cp targets [@a_cp, @b_cp] {
  // CHECK:   }
  // CHECK: }
  covergroup cg_simple_cross @(posedge clk);
    a_cp: coverpoint addr;
    b_cp: coverpoint cmd;
    cross a_cp, b_cp;  // Cross without explicit bins (automatic bins)
  endgroup

endmodule
