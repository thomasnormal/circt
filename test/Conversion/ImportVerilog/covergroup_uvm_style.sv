// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test UVM-style covergroup patterns commonly used in verification.
// IEEE 1800-2017 Section 19 "Functional coverage"

// CHECK: moore.module @uvm_covergroup_test
// CHECK:   moore.covergroup.inst @transaction_cg
// CHECK:   moore.covergroup.inst @fsm_cg

module uvm_covergroup_test;
  logic        clk;
  logic [7:0]  opcode;
  logic [15:0] address;
  logic [31:0] data;
  logic        read_write;

  // Test a transaction covergroup (common UVM pattern)
  // CHECK: moore.covergroup.decl @transaction_cg sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @op_cp : !moore.l8 {
  // CHECK:     moore.coverbin.decl @read_ops kind<bins> values [1, 2, 3]
  // CHECK:     moore.coverbin.decl @write_ops kind<bins> values [16, 17, 18]
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @addr_cp : !moore.l16 {
  // CHECK:     moore.coverbin.decl @low_addr kind<bins>
  // CHECK:     moore.coverbin.decl @mid_addr kind<bins>
  // CHECK:     moore.coverbin.decl @high_addr kind<bins>
  // CHECK:   }
  // CHECK:   moore.coverpoint.decl @rw_cp : !moore.l1 {
  // CHECK:   }
  // CHECK:   moore.covercross.decl @op_addr_cross targets [@op_cp, @addr_cp]
  // CHECK: }
  covergroup transaction_cg @(posedge clk);
    op_cp: coverpoint opcode {
      bins read_ops = {8'h01, 8'h02, 8'h03};
      bins write_ops = {8'h10, 8'h11, 8'h12};
    }
    addr_cp: coverpoint address {
      bins low_addr = {[0:16'h3FFF]};
      bins mid_addr = {[16'h4000:16'hBFFF]};
      bins high_addr = {[16'hC000:16'hFFFF]};
    }
    rw_cp: coverpoint read_write;
    op_addr_cross: cross op_cp, addr_cp;
  endgroup

  // Test a state machine covergroup
  // Note: Range bins like {[4:15]} don't evaluate to constant lists, so
  // the illegal_bins for invalid states appears without explicit values.
  // CHECK: moore.covergroup.decl @fsm_cg sampling_event<"@(posedge clk)"> {
  // CHECK:   moore.coverpoint.decl @state_cp : !moore.l4 {
  // CHECK:     moore.coverbin.decl @idle kind<bins> values [0]
  // CHECK:     moore.coverbin.decl @running kind<bins> values [1]
  // CHECK:     moore.coverbin.decl @waiting kind<bins> values [2]
  // CHECK:     moore.coverbin.decl @done kind<bins> values [3]
  // CHECK:     moore.coverbin.decl @invalid kind<illegal_bins>
  // CHECK:   }
  // CHECK: }
  logic [3:0] state;
  covergroup fsm_cg @(posedge clk);
    state_cp: coverpoint state {
      bins idle = {0};
      bins running = {1};
      bins waiting = {2};
      bins done = {3};
      illegal_bins invalid = {[4:15]};
    }
  endgroup

  // Instantiate the covergroups (typical UVM usage)
  transaction_cg trans_cov = new();
  fsm_cg fsm_cov = new();

endmodule
