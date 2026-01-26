// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s
// REQUIRES: slang
// XFAIL: *
// UVM runtime has compilation issues affecting TLM analysis port.

//===----------------------------------------------------------------------===//
// TLM Analysis Port Test - Iteration 108 Track D
//===----------------------------------------------------------------------===//
//
// This test validates UVM TLM analysis port patterns commonly used in
// production AVIPs for component communication. Patterns tested:
//
// 1. uvm_analysis_port declaration and construction
// 2. Connection to analysis_export (subscriber pattern)
// 3. write() method for broadcasting transactions
// 4. uvm_subscriber base class with custom write()
// 5. Monitor -> Coverage connection pattern
//
// Pattern sources: mbit AHB, APB, AXI4, SPI AVIPs
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"

import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Transaction class for testing
//===----------------------------------------------------------------------===//

class apb_transaction extends uvm_sequence_item;
  `uvm_object_utils(apb_transaction)

  rand bit [31:0] paddr;
  rand bit [31:0] pwdata;
  rand bit [31:0] prdata;
  rand bit pwrite;
  rand bit pslverr;
  rand bit [3:0] pstrb;

  function new(string name = "apb_transaction");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("paddr=0x%08h pwdata=0x%08h prdata=0x%08h pwrite=%b",
                     paddr, pwdata, prdata, pwrite);
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_transaction

//===----------------------------------------------------------------------===//
// Monitor with Analysis Port (Pattern from mbit APB AVIP)
//===----------------------------------------------------------------------===//

class apb_master_monitor extends uvm_monitor;
  `uvm_component_utils(apb_master_monitor)

  // Analysis port declaration - key TLM component
  uvm_analysis_port #(apb_transaction) apb_master_analysis_port;

  function new(string name = "apb_master_monitor", uvm_component parent = null);
    super.new(name, parent);
    // Analysis port constructed in new() as per UVM convention
    apb_master_analysis_port = new("apb_master_analysis_port", this);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    `uvm_info(get_type_name(), "Build phase complete", UVM_HIGH)
  endfunction

  // Simulated run_phase that would normally sample from interface
  virtual task run_phase(uvm_phase phase);
    apb_transaction txn;
    super.run_phase(phase);

    repeat (5) begin
      txn = apb_transaction::type_id::create("txn");
      void'(txn.randomize());
      `uvm_info(get_type_name(),
                $sformatf("Sending via analysis_port: %s", txn.convert2string()),
                UVM_HIGH)
      // Broadcast transaction to all connected subscribers
      apb_master_analysis_port.write(txn);
    end
  endtask
endclass

// CHECK: moore.class.classdecl @apb_master_monitor extends @"uvm_pkg::uvm_monitor"

//===----------------------------------------------------------------------===//
// Coverage using uvm_subscriber (Pattern from mbit APB AVIP)
//===----------------------------------------------------------------------===//

class apb_master_coverage extends uvm_subscriber #(apb_transaction);
  `uvm_component_utils(apb_master_coverage)

  // Transaction counter for verification
  int transaction_count;
  int write_count;
  int read_count;

  // Covergroup for functional coverage
  covergroup apb_cg with function sample(apb_transaction txn);
    option.per_instance = 1;

    PADDR_CP: coverpoint txn.paddr[7:0] {
      bins low_addr = {[0:63]};
      bins mid_addr = {[64:191]};
      bins high_addr = {[192:255]};
    }

    PWRITE_CP: coverpoint txn.pwrite {
      bins write_op = {1};
      bins read_op = {0};
    }

    PSLVERR_CP: coverpoint txn.pslverr {
      bins no_error = {0};
      bins error = {1};
    }

    // Cross coverage
    ADDR_X_WRITE: cross PADDR_CP, PWRITE_CP;
  endgroup

  function new(string name = "apb_master_coverage", uvm_component parent = null);
    super.new(name, parent);
    apb_cg = new();
  endfunction

  // write() is called automatically when analysis_port.write() is invoked
  // This is the key uvm_subscriber callback
  virtual function void write(apb_transaction t);
    transaction_count++;
    if (t.pwrite)
      write_count++;
    else
      read_count++;

    `uvm_info(get_type_name(),
              $sformatf("Coverage received txn #%0d: %s",
                        transaction_count, t.convert2string()),
              UVM_HIGH)
    apb_cg.sample(t);
  endfunction

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(),
              $sformatf("Coverage: %0.2f%% (%0d transactions, %0d writes, %0d reads)",
                        apb_cg.get_coverage(), transaction_count, write_count, read_count),
              UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_master_coverage extends @"uvm_pkg::uvm_subscriber"

//===----------------------------------------------------------------------===//
// Simple Checker using uvm_subscriber
//===----------------------------------------------------------------------===//

class apb_checker extends uvm_subscriber #(apb_transaction);
  `uvm_component_utils(apb_checker)

  int error_count;

  function new(string name = "apb_checker", uvm_component parent = null);
    super.new(name, parent);
    error_count = 0;
  endfunction

  virtual function void write(apb_transaction t);
    // Simple protocol check: pslverr should only occur on valid transactions
    if (t.pslverr && t.paddr == 0) begin
      `uvm_error(get_type_name(), "Unexpected slave error on address 0")
      error_count++;
    end
  endfunction

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    if (error_count == 0)
      `uvm_info(get_type_name(), "All protocol checks passed", UVM_LOW)
    else
      `uvm_error(get_type_name(), $sformatf("%0d protocol errors detected", error_count))
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_checker extends @"uvm_pkg::uvm_subscriber"

//===----------------------------------------------------------------------===//
// Agent demonstrating port connections (Pattern from mbit AVIPs)
//===----------------------------------------------------------------------===//

class apb_master_agent extends uvm_agent;
  `uvm_component_utils(apb_master_agent)

  apb_master_monitor monitor;
  apb_master_coverage coverage;
  apb_checker checker;

  function new(string name = "apb_master_agent", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    monitor = apb_master_monitor::type_id::create("monitor", this);
    coverage = apb_master_coverage::type_id::create("coverage", this);
    checker = apb_checker::type_id::create("checker", this);
  endfunction

  // Key pattern: connect monitor's analysis_port to subscriber's analysis_export
  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);

    // Monitor broadcasts to multiple subscribers (1-to-N pattern)
    // Both coverage and checker receive the same transactions
    monitor.apb_master_analysis_port.connect(coverage.analysis_export);
    monitor.apb_master_analysis_port.connect(checker.analysis_export);

    `uvm_info(get_type_name(), "Analysis port connections complete", UVM_HIGH)
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_master_agent extends @"uvm_pkg::uvm_agent"

//===----------------------------------------------------------------------===//
// Environment with hierarchical analysis export (advanced pattern)
//===----------------------------------------------------------------------===//

class apb_env extends uvm_env;
  `uvm_component_utils(apb_env)

  apb_master_agent agent;

  // Hierarchical export - allows upper-level components to connect
  uvm_analysis_export #(apb_transaction) analysis_export;

  function new(string name = "apb_env", uvm_component parent = null);
    super.new(name, parent);
    analysis_export = new("analysis_export", this);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    agent = apb_master_agent::type_id::create("agent", this);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    // Expose agent's analysis port at env level via export
    // This allows test-level components to connect without knowing internal structure
    // Pattern: monitor.port -> env.export -> test_level_subscriber
    // Not connecting here since we have no upper-level subscriber
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_env extends @"uvm_pkg::uvm_env"

//===----------------------------------------------------------------------===//
// Test demonstrating complete analysis port flow
//===----------------------------------------------------------------------===//

class apb_analysis_port_test extends uvm_test;
  `uvm_component_utils(apb_analysis_port_test)

  apb_env env;

  function new(string name = "apb_analysis_port_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = apb_env::type_id::create("env", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);
    phase.raise_objection(this);
    #100;  // Allow some simulation time
    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), "Analysis port test complete", UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @apb_analysis_port_test extends @"uvm_pkg::uvm_test"

//===----------------------------------------------------------------------===//
// Top module for test
//===----------------------------------------------------------------------===//

module tlm_analysis_port_test_top;
  initial begin
    $display("TLM Analysis Port pattern test - Iteration 108 Track D");
    run_test("apb_analysis_port_test");
  end
endmodule

// CHECK: moore.module @tlm_analysis_port_test_top
