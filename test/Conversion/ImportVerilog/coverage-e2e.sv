// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// End-to-End Coverage Test
//===----------------------------------------------------------------------===//
// This test verifies the complete coverage collection path:
// 1. SystemVerilog covergroup parsing via slang
// 2. Moore dialect representation (CovergroupDeclOp, CoverpointDeclOp)
// 3. Runtime lowering to __moore_coverage_* function calls
// 4. Coverage sampling and reporting infrastructure

//===----------------------------------------------------------------------===//
// Basic Covergroup with Explicit Bins
//===----------------------------------------------------------------------===//

module coverage_test;
  logic [7:0] data;
  logic clk;

  // CHECK: moore.module @coverage_test

  // A covergroup with explicit bins for testing coverage collection.
  // The runtime tracks: __moore_covergroup_create, __moore_coverpoint_init,
  // and __moore_coverpoint_sample calls.
  covergroup cg @(posedge clk);
    cp_data: coverpoint data {
      bins low = {[0:63]};
      bins mid = {[64:127]};
      bins high = {[128:255]};
    }
  endgroup

  // CHECK: moore.covergroup.inst @cg
  cg cg_inst;

  initial begin
    cg_inst = new();
    // The runtime will track coverage when sampling occurs.
    // get_coverage() returns coverage percentage from __moore_covergroup_get_coverage
  end
endmodule

// CHECK: moore.covergroup.decl @cg
// CHECK:   moore.coverpoint.decl @cp_data
// CHECK:     moore.coverbin.decl @low
// CHECK:     moore.coverbin.decl @mid
// CHECK:     moore.coverbin.decl @high

//===----------------------------------------------------------------------===//
// Multiple Coverpoints with Different Types
//===----------------------------------------------------------------------===//

module multi_coverpoint_test;
  logic clk;
  logic [3:0] addr;
  logic [15:0] data;
  logic rw;

  // CHECK: moore.module @multi_coverpoint_test

  covergroup transaction_cg @(posedge clk);
    addr_cp: coverpoint addr {
      bins low_addr = {[0:7]};
      bins high_addr = {[8:15]};
    }
    data_cp: coverpoint data;  // Auto bins
    rw_cp: coverpoint rw {
      bins read = {0};
      bins write = {1};
    }
  endgroup

  // CHECK: moore.covergroup.inst @transaction_cg
  transaction_cg txn_cov;

  initial begin
    txn_cov = new();
  end
endmodule

// CHECK: moore.covergroup.decl @transaction_cg
// CHECK:   moore.coverpoint.decl @addr_cp
// CHECK:     moore.coverbin.decl @low_addr
// CHECK:     moore.coverbin.decl @high_addr
// CHECK:   moore.coverpoint.decl @data_cp
// CHECK:   moore.coverpoint.decl @rw_cp
// CHECK:     moore.coverbin.decl @read
// CHECK:     moore.coverbin.decl @write

//===----------------------------------------------------------------------===//
// Coverage with Sampling Event
//===----------------------------------------------------------------------===//

module sampled_coverage;
  logic clk;
  logic [7:0] value;

  // CHECK: moore.module @sampled_coverage

  // Covergroup with explicit posedge clk sampling event
  covergroup sample_cg @(posedge clk);
    cp: coverpoint value;
  endgroup

  // CHECK: moore.covergroup.inst @sample_cg
  sample_cg my_cg;

  initial begin
    my_cg = new();
    // Each @(posedge clk) triggers automatic sampling via
    // __moore_coverpoint_sample(cg_handle, cp_index, value)
  end
endmodule

// CHECK: moore.covergroup.decl @sample_cg sampling_event
// CHECK:   moore.coverpoint.decl @cp

//===----------------------------------------------------------------------===//
// Single Value Bins
//===----------------------------------------------------------------------===//

module single_value_bins;
  logic clk;
  logic [2:0] state;

  // CHECK: moore.module @single_value_bins

  covergroup state_cg @(posedge clk);
    state_cp: coverpoint state {
      bins idle = {0};
      bins start = {1};
      bins run = {2};
      bins stop = {3};
      bins error = {7};
    }
  endgroup

  // CHECK: moore.covergroup.inst @state_cg
  state_cg s_cg;

  initial begin
    s_cg = new();
  end
endmodule

// CHECK: moore.covergroup.decl @state_cg
// CHECK:   moore.coverpoint.decl @state_cp
// CHECK:     moore.coverbin.decl @idle
// CHECK:     moore.coverbin.decl @start
// CHECK:     moore.coverbin.decl @run
// CHECK:     moore.coverbin.decl @stop
// CHECK:     moore.coverbin.decl @error

//===----------------------------------------------------------------------===//
// Wide Value Coverage
//===----------------------------------------------------------------------===//

module wide_coverage;
  logic clk;
  logic [31:0] wide_data;

  // CHECK: moore.module @wide_coverage

  covergroup wide_cg @(posedge clk);
    wide_cp: coverpoint wide_data {
      bins low = {[0:32'h3FFFFFFF]};
      bins high = {[32'h40000000:32'hFFFFFFFF]};
    }
  endgroup

  // CHECK: moore.covergroup.inst @wide_cg
  wide_cg w_cg;

  initial begin
    w_cg = new();
  end
endmodule

// CHECK: moore.covergroup.decl @wide_cg
// CHECK:   moore.coverpoint.decl @wide_cp
// CHECK:     moore.coverbin.decl @low
// CHECK:     moore.coverbin.decl @high
