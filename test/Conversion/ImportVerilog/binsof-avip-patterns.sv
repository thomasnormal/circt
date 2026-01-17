// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test AVIP-style binsof/intersect patterns for cross coverage bins.
// These patterns are commonly found in verification IP coverage models.
// See IEEE 1800-2017 Section 19.6.1 for specification.

module test_avip_coverage_patterns;
  logic clk;
  logic [7:0] addr;
  logic [3:0] cmd;
  logic [1:0] mode;
  logic       valid;
  logic       error;

  // Pattern 1: Simple cross with single binsof/intersect
  // CHECK: moore.covergroup.decl @simple_cross_bins
  // CHECK:   moore.covercross.decl @cmd_x_mode targets [@cmd_cp, @mode_cp] {
  // CHECK:     moore.crossbin.decl @read_ops kind<bins> {
  // CHECK:       moore.binsof @cmd_cp intersect [0, 1]
  // CHECK:     }
  // CHECK:   }
  covergroup simple_cross_bins @(posedge clk);
    cmd_cp: coverpoint cmd;
    mode_cp: coverpoint mode;
    cmd_x_mode: cross cmd_cp, mode_cp {
      bins read_ops = binsof(cmd_cp) intersect {[0:1]};
    }
  endgroup

  // Pattern 2: Multiple bins with different ranges
  // CHECK: moore.covergroup.decl @multi_range_cross
  // CHECK:   moore.covercross.decl @addr_x_cmd targets [@addr_cp, @cmd_cp] {
  // CHECK:     moore.crossbin.decl @low kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @mid kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @high kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
  // CHECK:     }
  // CHECK:   }
  covergroup multi_range_cross @(posedge clk);
    addr_cp: coverpoint addr;
    cmd_cp: coverpoint cmd;
    addr_x_cmd: cross addr_cp, cmd_cp {
      bins low = binsof(addr_cp) intersect {[0:15]};
      bins mid = binsof(addr_cp) intersect {[100:110]};
      bins high = binsof(addr_cp) intersect {[240:255]};
    }
  endgroup

  // Pattern 3: Illegal and ignore bins
  // CHECK: moore.covergroup.decl @illegal_ignore_bins
  // CHECK:   moore.covercross.decl @addr_x_valid targets [@addr_cp, @valid_cp] {
  // CHECK:     moore.crossbin.decl @valid_high kind<bins> {
  // CHECK:       moore.binsof @valid_cp intersect [1]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @ignore_reserved kind<ignore_bins> {
  // CHECK:       moore.binsof @addr_cp intersect [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239]
  // CHECK:     }
  // CHECK:     moore.crossbin.decl @bad_combo kind<illegal_bins> {
  // CHECK:       moore.binsof @addr_cp intersect [255]
  // CHECK:       moore.binsof @valid_cp intersect [0]
  // CHECK:     }
  // CHECK:   }
  covergroup illegal_ignore_bins @(posedge clk);
    addr_cp: coverpoint addr;
    valid_cp: coverpoint valid;
    addr_x_valid: cross addr_cp, valid_cp {
      bins valid_high = binsof(valid_cp) intersect {1};
      ignore_bins ignore_reserved = binsof(addr_cp) intersect {[200:239]};
      illegal_bins bad_combo = binsof(addr_cp) intersect {255} && binsof(valid_cp) intersect {0};
    }
  endgroup

  // Pattern 4: Combined AND expression
  // CHECK: moore.covergroup.decl @and_expr_cross
  // CHECK:   moore.covercross.decl @full_cross targets [@addr_cp, @cmd_cp] {
  // CHECK:     moore.crossbin.decl @special_combo kind<bins> {
  // CHECK:       moore.binsof @addr_cp intersect [0]
  // CHECK:       moore.binsof @cmd_cp intersect [15]
  // CHECK:     }
  // CHECK:   }
  covergroup and_expr_cross @(posedge clk);
    addr_cp: coverpoint addr;
    cmd_cp: coverpoint cmd;
    full_cross: cross addr_cp, cmd_cp {
      bins special_combo = binsof(addr_cp) intersect {0} && binsof(cmd_cp) intersect {15};
    }
  endgroup

endmodule
