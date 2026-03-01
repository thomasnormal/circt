//===----------------------------------------------------------------------===//
// UVM Coverage Infrastructure Test
//===----------------------------------------------------------------------===//
// Tests the UVM functional coverage support including:
// - object-based coverage collectors with covergroups
// - uvm_mem_mam (Memory Allocation Manager)
// - uvm_mem_region
// - Covergroup integration with UVM
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm-core %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top tb_coverage --max-time=1000000000 +UVM_TESTNAME=mam_test +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM-MAM
// RUN: circt-sim %t.mlir --top tb_coverage --max-time=1000000000 +UVM_TESTNAME=coverage_db_test +UVM_VERBOSITY=UVM_NONE 2>&1 | FileCheck %s --check-prefix=SIM-COVDB
//
// SIM-MAM: UVM_COVERAGE_MAM_PASS
// SIM-MAM-NOT: UVM_ERROR
// SIM-MAM: [circt-sim] Simulation completed
//
// SIM-COVDB: UVM_COVERAGE_DB_PASS
// SIM-COVDB-NOT: UVM_ERROR
// SIM-COVDB: [circt-sim] Simulation completed

`timescale 1ns/1ps

`include "uvm_macros.svh"

package coverage_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Concrete coverage class for testing
  //==========================================================================
  // UVM 1.2 does not define a concrete uvm_coverage base object in this runtime.
  // Keep coverage state in a regular uvm_object and sample via embedded covergroups.
  class my_coverage extends uvm_object;
    `uvm_object_utils(my_coverage)

    // Sample counter for testing
    int sample_count;

    // Covergroup embedded in the class
    bit [7:0] data_value;
    bit [3:0] cmd_value;

    covergroup data_cg;
      data_cp: coverpoint data_value {
        bins low = {[0:63]};
        bins mid = {[64:191]};
        bins high = {[192:255]};
      }
      cmd_cp: coverpoint cmd_value {
        bins read = {4'h0, 4'h1};
        bins write = {4'h2, 4'h3};
        bins other = default;
      }
    endgroup

    function new(string name = "my_coverage");
      super.new(name);
      sample_count = 0;
      data_cg = new();
    endfunction

    // Override sample method
    virtual function void sample();
      sample_count++;
      data_cg.sample();
    endfunction

    // Override to provide coverage percentage
    virtual function real get_coverage_pct();
      return data_cg.get_coverage();
    endfunction

    // Helper to set values and sample
    function void collect(bit [7:0] data, bit [3:0] cmd);
      data_value = data;
      cmd_value = cmd;
      sample();
    endfunction

  endclass

  //==========================================================================
  // Transaction with embedded covergroup
  //==========================================================================
  class covered_tx extends uvm_sequence_item;
    `uvm_object_utils(covered_tx)

    rand bit [15:0] addr;
    rand bit [31:0] data;
    rand bit        write;

    // Transaction covergroup
    covergroup tx_cg with function sample(bit [15:0] a, bit [31:0] d, bit w);
      addr_cp: coverpoint a {
        bins low_addr = {[16'h0000:16'h3FFF]};
        bins mid_addr = {[16'h4000:16'hBFFF]};
        bins high_addr = {[16'hC000:16'hFFFF]};
      }
      write_cp: coverpoint w {
        bins reads = {0};
        bins writes = {1};
      }
      // Cross coverage
      addr_write_cross: cross addr_cp, write_cp;
    endgroup

    function new(string name = "covered_tx");
      super.new(name);
      tx_cg = new();
    endfunction

    virtual function void do_copy(uvm_object rhs);
      covered_tx rhs_tx;
      super.do_copy(rhs);
      if (!$cast(rhs_tx, rhs))
        `uvm_fatal("CAST", "Cast failed")
      addr = rhs_tx.addr;
      data = rhs_tx.data;
      write = rhs_tx.write;
    endfunction

    function void sample_coverage();
      tx_cg.sample(addr, data, write);
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=%0h data=%0h write=%0b", addr, data, write);
    endfunction

  endclass

  //==========================================================================
  // Coverage collector component
  //==========================================================================
  class coverage_collector extends uvm_subscriber #(covered_tx);
    `uvm_component_utils(coverage_collector)

    my_coverage cov;
    int transaction_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      transaction_count = 0;
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      cov = my_coverage::type_id::create("cov");
    endfunction

    virtual function void write(covered_tx t);
      transaction_count++;
      t.sample_coverage();
      cov.collect(t.data[7:0], t.addr[3:0]);
      `uvm_info("COV", $sformatf("Sampled tx #%0d: %s", transaction_count, t.convert2string()), UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("COV", $sformatf("Coverage samples: %0d", cov.sample_count), UVM_NONE)
      `uvm_info("COV", $sformatf("Coverage percentage: %.2f%%", cov.get_coverage_pct()), UVM_NONE)
    endfunction

  endclass

  //==========================================================================
  // Memory Allocation Manager Test
  //==========================================================================
  class mam_test extends uvm_test;
    `uvm_component_utils(mam_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_mem_mam_cfg cfg;
      uvm_mem_mam mam;
      uvm_mem_region r1, r2, r3;
      uvm_mem_region regions[$];
      bit pass;

      pass = 1;

      phase.raise_objection(this, "Testing MAM");

      // Create and configure MAM
      cfg = new;
      cfg.start_offset = 0;
      cfg.end_offset = 1023;
      cfg.n_bytes = 4;
      cfg.mode = uvm_mem_mam::GREEDY;

      mam = new("mam", cfg);

      `uvm_info("MAM", $sformatf("Initial state: %s", mam.convert2string()), UVM_LOW)

      // Test deterministic reserve_region allocations.
      r1 = mam.reserve_region(0, 64);
      if (r1 == null) begin
        `uvm_error("MAM", "Failed to allocate r1")
        pass = 0;
      end else
        `uvm_info("MAM", $sformatf("Allocated r1: %s", r1.convert2string()), UVM_LOW)

      r2 = mam.reserve_region(16, 128);
      if (r2 == null) begin
        `uvm_error("MAM", "Failed to allocate r2")
        pass = 0;
      end else
        `uvm_info("MAM", $sformatf("Allocated r2: %s", r2.convert2string()), UVM_LOW)

      // Test reserve_region at specific address
      r3 = mam.reserve_region(512, 256);
      if (r3 == null) begin
        `uvm_error("MAM", "Failed to reserve r3")
        pass = 0;
      end else
        `uvm_info("MAM", $sformatf("Reserved r3: %s", r3.convert2string()), UVM_LOW)

      `uvm_info("MAM", $sformatf("After allocations: %s", mam.convert2string()), UVM_LOW)

      // Enumerate allocated regions through MAM iterator API.
      begin
        uvm_mem_region region;
        region = mam.for_each(1);
        while (region != null) begin
          regions.push_back(region);
          region = mam.for_each();
        end
      end
      if (regions.size() != 3) begin
        `uvm_error("MAM", $sformatf("Expected 3 allocated regions, got %0d", regions.size()))
        pass = 0;
      end
      `uvm_info("MAM", $sformatf("Number of allocated regions: %0d", regions.size()), UVM_LOW)

      // Test release
      mam.release_region(r2);
      `uvm_info("MAM", $sformatf("After releasing r2: %s", mam.convert2string()), UVM_LOW)

      // Test overlapping reservation (should fail)
      begin
        uvm_mem_region r_overlap;
        r_overlap = mam.reserve_region(500, 100);
        if (r_overlap != null) begin
          `uvm_error("MAM", "Overlapping reservation should have failed!")
          pass = 0;
        end else
          `uvm_info("MAM", "Correctly rejected overlapping reservation", UVM_LOW)
      end

      // Release all
      mam.release_all_regions();
      `uvm_info("MAM", $sformatf("After release_all: %s", mam.convert2string()), UVM_LOW)

      begin
        int remaining_regions;
        uvm_mem_region region;
        remaining_regions = 0;
        region = mam.for_each(1);
        while (region != null) begin
          remaining_regions++;
          region = mam.for_each();
        end
        if (remaining_regions != 0) begin
          `uvm_error("MAM", $sformatf("release_all_regions left %0d regions", remaining_regions))
          pass = 0;
        end
      end

      if (pass)
        `uvm_info("MAM", "UVM_COVERAGE_MAM_PASS", UVM_NONE)

      phase.drop_objection(this, "MAM test done");
    endtask

  endclass

  //==========================================================================
  // Coverage Database Test
  //==========================================================================
  class coverage_db_test extends uvm_test;
    `uvm_component_utils(coverage_db_test)

    coverage_collector collector;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      collector = coverage_collector::type_id::create("collector", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      covered_tx tx;
      bit pass;
      pass = 1;

      phase.raise_objection(this, "Testing coverage DB");

      // Create and sample some transactions
      repeat (10) begin
        tx = covered_tx::type_id::create("tx");
        if (!tx.randomize()) begin
          `uvm_error("TX", "Randomization failed")
          pass = 0;
        end
        collector.write(tx);
      end

      if (collector.transaction_count != 10) begin
        `uvm_error("COVDB", $sformatf("Expected 10 transactions, got %0d", collector.transaction_count))
        pass = 0;
      end
      if (collector.cov.sample_count != 10) begin
        `uvm_error("COVDB", $sformatf("Expected 10 coverage samples, got %0d", collector.cov.sample_count))
        pass = 0;
      end

      `uvm_info("COVDB", $sformatf("Collected coverage samples: %0d",
                                    collector.cov.sample_count), UVM_LOW)
      `uvm_info("COVDB", $sformatf("Collector coverage: %.2f%%",
                                    collector.cov.get_coverage_pct()), UVM_LOW)

      if (pass)
        `uvm_info("COVDB", "UVM_COVERAGE_DB_PASS", UVM_NONE)

      phase.drop_objection(this, "Coverage DB test done");
    endtask

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_coverage;
  import uvm_pkg::*;
  import coverage_test_pkg::*;

  // Simple clock for covergroup sampling
  logic clk;
  logic [7:0] data;
  logic [3:0] addr;

  // Module-level covergroup
  covergroup simple_cg @(posedge clk);
    option.per_instance = 1;
    data_cp: coverpoint data {
      bins zeros = {0};
      bins ones = {8'hFF};
      bins others = default;
    }
    addr_cp: coverpoint addr;
  endgroup

  simple_cg cg_inst = new();

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Stimulus
  initial begin
    data = 0;
    addr = 0;
    repeat (20) begin
      @(posedge clk);
      data = $urandom();
      addr = $urandom();
    end
  end

  // Run UVM test
  initial begin
    `uvm_info("TB", "Starting UVM coverage test", UVM_NONE)
    run_test();
  end

  // Timeout
  initial begin
    #10000;
    $display("Test timeout - ending simulation");
    $finish;
  end

endmodule
