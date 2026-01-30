// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s
// REQUIRES: slang
// XFAIL: *

//===----------------------------------------------------------------------===//
// TLM FIFO Test - Iteration 108 Track D
//===----------------------------------------------------------------------===//
//
// This test validates UVM TLM FIFO patterns commonly used in production AVIPs
// for scoreboard and transaction buffering. Patterns tested:
//
// 1. uvm_tlm_analysis_fifo declaration (single and array)
// 2. Connection via analysis_export
// 3. Blocking get() operations
// 4. size() and is_empty() queries
// 5. Scoreboard pattern with master/slave FIFO comparison
// 6. Dynamic array of FIFOs (indexed by number of agents)
//
// Pattern sources: mbit AHB, APB, AXI4 scoreboard implementations
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"

import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Transaction classes for testing
//===----------------------------------------------------------------------===//

class ahb_master_transaction extends uvm_sequence_item;
  `uvm_object_utils(ahb_master_transaction)

  rand bit [31:0] haddr;
  rand bit [31:0] hwdata;
  rand bit [31:0] hrdata;
  rand bit hwrite;
  rand bit [2:0] hsize;
  rand bit [2:0] hburst;

  function new(string name = "ahb_master_transaction");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("haddr=0x%08h hwdata=0x%08h hwrite=%b hsize=%0d",
                     haddr, hwdata, hwrite, hsize);
  endfunction
endclass

class ahb_slave_transaction extends uvm_sequence_item;
  `uvm_object_utils(ahb_slave_transaction)

  rand bit [31:0] haddr;
  rand bit [31:0] hwdata;
  rand bit [31:0] hrdata;
  rand bit hwrite;
  rand bit hreadyout;
  rand bit [1:0] hresp;

  function new(string name = "ahb_slave_transaction");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("haddr=0x%08h hwdata=0x%08h hrdata=0x%08h hwrite=%b",
                     haddr, hwdata, hrdata, hwrite);
  endfunction
endclass

// CHECK: moore.class.classdecl @ahb_master_transaction
// CHECK: moore.class.classdecl @ahb_slave_transaction

//===----------------------------------------------------------------------===//
// Monitor with analysis port (transaction producer)
//===----------------------------------------------------------------------===//

class ahb_master_monitor extends uvm_monitor;
  `uvm_component_utils(ahb_master_monitor)

  uvm_analysis_port #(ahb_master_transaction) ahb_master_analysis_port;

  function new(string name = "ahb_master_monitor", uvm_component parent = null);
    super.new(name, parent);
    ahb_master_analysis_port = new("ahb_master_analysis_port", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    ahb_master_transaction txn;
    super.run_phase(phase);

    repeat (10) begin
      txn = ahb_master_transaction::type_id::create("txn");
      void'(txn.randomize());
      `uvm_info(get_type_name(),
                $sformatf("Master monitor sending: %s", txn.convert2string()),
                UVM_HIGH)
      ahb_master_analysis_port.write(txn);
      #10;
    end
  endtask
endclass

class ahb_slave_monitor extends uvm_monitor;
  `uvm_component_utils(ahb_slave_monitor)

  uvm_analysis_port #(ahb_slave_transaction) ahb_slave_analysis_port;

  function new(string name = "ahb_slave_monitor", uvm_component parent = null);
    super.new(name, parent);
    ahb_slave_analysis_port = new("ahb_slave_analysis_port", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    ahb_slave_transaction txn;
    super.run_phase(phase);

    repeat (10) begin
      txn = ahb_slave_transaction::type_id::create("txn");
      void'(txn.randomize());
      `uvm_info(get_type_name(),
                $sformatf("Slave monitor sending: %s", txn.convert2string()),
                UVM_HIGH)
      ahb_slave_analysis_port.write(txn);
      #10;
    end
  endtask
endclass

// CHECK: moore.class.classdecl @ahb_master_monitor extends @"uvm_pkg::uvm_monitor"
// CHECK: moore.class.classdecl @ahb_slave_monitor extends @"uvm_pkg::uvm_monitor"

//===----------------------------------------------------------------------===//
// Scoreboard with Analysis FIFOs (Pattern from mbit AHB AVIP)
//===----------------------------------------------------------------------===//

// Parameterized constants for multi-agent support
parameter int NO_OF_MASTERS = 2;
parameter int NO_OF_SLAVES = 2;

class ahb_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(ahb_scoreboard)

  // Dynamic array of analysis FIFOs - key pattern from mbit AVIPs
  // Allows scaling with number of agents
  uvm_tlm_analysis_fifo #(ahb_master_transaction) ahb_master_analysis_fifo[];
  uvm_tlm_analysis_fifo #(ahb_slave_transaction) ahb_slave_analysis_fifo[];

  // Statistics counters
  int master_transaction_count;
  int slave_transaction_count;
  int verified_hwdata_count;
  int failed_hwdata_count;
  int verified_haddr_count;
  int failed_haddr_count;

  function new(string name = "ahb_scoreboard", uvm_component parent = null);
    super.new(name, parent);

    // Allocate FIFO arrays based on number of agents
    ahb_master_analysis_fifo = new[NO_OF_MASTERS];
    ahb_slave_analysis_fifo = new[NO_OF_SLAVES];

    // Create individual FIFOs with indexed names
    foreach (ahb_master_analysis_fifo[i]) begin
      ahb_master_analysis_fifo[i] =
        new($sformatf("ahb_master_analysis_fifo[%0d]", i), this);
    end

    foreach (ahb_slave_analysis_fifo[i]) begin
      ahb_slave_analysis_fifo[i] =
        new($sformatf("ahb_slave_analysis_fifo[%0d]", i), this);
    end
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    `uvm_info(get_type_name(),
              $sformatf("Scoreboard built with %0d master and %0d slave FIFOs",
                        NO_OF_MASTERS, NO_OF_SLAVES),
              UVM_MEDIUM)
  endfunction

  // Main comparison task - runs forever comparing transactions
  virtual task run_phase(uvm_phase phase);
    ahb_master_transaction master_txn;
    ahb_slave_transaction slave_txn;

    super.run_phase(phase);

    forever begin
      // Get transactions from each master FIFO
      for (int j = 0; j < NO_OF_MASTERS; j++) begin
        ahb_master_analysis_fifo[j].get(master_txn);  // Blocking get
        master_transaction_count++;
        `uvm_info(get_type_name(),
                  $sformatf("Got master[%0d] txn: %s", j, master_txn.convert2string()),
                  UVM_HIGH)
      end

      // Get transactions from each slave FIFO
      for (int i = 0; i < NO_OF_SLAVES; i++) begin
        ahb_slave_analysis_fifo[i].get(slave_txn);  // Blocking get
        slave_transaction_count++;
        `uvm_info(get_type_name(),
                  $sformatf("Got slave[%0d] txn: %s", i, slave_txn.convert2string()),
                  UVM_HIGH)
      end

      // Compare transactions (simplified - actual AVIPs have more complex logic)
      if (master_txn.hwrite == 1) begin
        // Write transaction comparison
        if (master_txn.hwdata == slave_txn.hwdata) begin
          verified_hwdata_count++;
          `uvm_info(get_type_name(), "HWDATA MATCH", UVM_HIGH)
        end else begin
          failed_hwdata_count++;
          `uvm_error(get_type_name(),
                     $sformatf("HWDATA MISMATCH: master=0x%08h slave=0x%08h",
                               master_txn.hwdata, slave_txn.hwdata))
        end

        if (master_txn.haddr == slave_txn.haddr) begin
          verified_haddr_count++;
          `uvm_info(get_type_name(), "HADDR MATCH", UVM_HIGH)
        end else begin
          failed_haddr_count++;
          `uvm_error(get_type_name(),
                     $sformatf("HADDR MISMATCH: master=0x%08h slave=0x%08h",
                               master_txn.haddr, slave_txn.haddr))
        end
      end
    end
  endtask

  // Check phase verifies FIFO draining
  virtual function void check_phase(uvm_phase phase);
    super.check_phase(phase);

    `uvm_info(get_type_name(), "--- SCOREBOARD CHECK PHASE ---", UVM_HIGH)

    // Verify transaction counts match
    if (master_transaction_count == slave_transaction_count) begin
      `uvm_info(get_type_name(),
                $sformatf("Transaction counts match: %0d", master_transaction_count),
                UVM_HIGH)
    end else begin
      `uvm_error(get_type_name(),
                 $sformatf("Transaction count mismatch: master=%0d slave=%0d",
                           master_transaction_count, slave_transaction_count))
    end

    // Check FIFOs are empty
    for (int i = 0; i < NO_OF_MASTERS; i++) begin
      if (ahb_master_analysis_fifo[i].size() == 0) begin
        `uvm_info(get_type_name(),
                  $sformatf("Master FIFO[%0d] is empty", i), UVM_HIGH)
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("Master FIFO[%0d] not empty: %0d items remaining",
                             i, ahb_master_analysis_fifo[i].size()))
      end
    end

    for (int i = 0; i < NO_OF_SLAVES; i++) begin
      if (ahb_slave_analysis_fifo[i].size() == 0) begin
        `uvm_info(get_type_name(),
                  $sformatf("Slave FIFO[%0d] is empty", i), UVM_HIGH)
      end else begin
        `uvm_error(get_type_name(),
                   $sformatf("Slave FIFO[%0d] not empty: %0d items remaining",
                             i, ahb_slave_analysis_fifo[i].size()))
      end
    end
  endfunction

  // Report phase summarizes results
  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);

    `uvm_info(get_type_name(), "--- SCOREBOARD REPORT ---", UVM_LOW)
    `uvm_info(get_type_name(),
              $sformatf("Master transactions: %0d", master_transaction_count), UVM_LOW)
    `uvm_info(get_type_name(),
              $sformatf("Slave transactions: %0d", slave_transaction_count), UVM_LOW)
    `uvm_info(get_type_name(),
              $sformatf("HWDATA verified: %0d failed: %0d",
                        verified_hwdata_count, failed_hwdata_count), UVM_LOW)
    `uvm_info(get_type_name(),
              $sformatf("HADDR verified: %0d failed: %0d",
                        verified_haddr_count, failed_haddr_count), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @ahb_scoreboard extends @"uvm_pkg::uvm_scoreboard"

//===----------------------------------------------------------------------===//
// Simple TLM FIFO test (non-analysis)
//===----------------------------------------------------------------------===//

class fifo_unit_test extends uvm_component;
  `uvm_component_utils(fifo_unit_test)

  // Basic TLM FIFO (not analysis FIFO)
  uvm_tlm_fifo #(ahb_master_transaction) basic_fifo;

  function new(string name = "fifo_unit_test", uvm_component parent = null);
    super.new(name, parent);
    // Create bounded FIFO with size 16
    basic_fifo = new("basic_fifo", this, 16);
  endfunction

  virtual task run_phase(uvm_phase phase);
    ahb_master_transaction txn;
    ahb_master_transaction retrieved_txn;

    super.run_phase(phase);

    // Test FIFO operations
    `uvm_info(get_type_name(), "Testing basic FIFO operations", UVM_MEDIUM)

    // Check initial state
    if (basic_fifo.is_empty()) begin
      `uvm_info(get_type_name(), "FIFO is initially empty - PASS", UVM_MEDIUM)
    end

    // Put some transactions
    for (int i = 0; i < 5; i++) begin
      txn = ahb_master_transaction::type_id::create($sformatf("txn_%0d", i));
      txn.haddr = i * 4;
      txn.hwdata = i * 100;
      basic_fifo.put(txn);
      `uvm_info(get_type_name(),
                $sformatf("Put txn %0d, FIFO size = %0d", i, basic_fifo.size()),
                UVM_HIGH)
    end

    // Verify size
    if (basic_fifo.size() == 5) begin
      `uvm_info(get_type_name(), "FIFO size is 5 - PASS", UVM_MEDIUM)
    end

    // Get transactions back
    for (int i = 0; i < 5; i++) begin
      basic_fifo.get(retrieved_txn);
      `uvm_info(get_type_name(),
                $sformatf("Got txn: haddr=0x%08h hwdata=0x%08h",
                          retrieved_txn.haddr, retrieved_txn.hwdata),
                UVM_HIGH)
    end

    // Verify empty
    if (basic_fifo.is_empty()) begin
      `uvm_info(get_type_name(), "FIFO is empty after gets - PASS", UVM_MEDIUM)
    end
  endtask
endclass

// CHECK: moore.class.classdecl @fifo_unit_test extends @"uvm_pkg::uvm_component"

//===----------------------------------------------------------------------===//
// Environment demonstrating FIFO connections
//===----------------------------------------------------------------------===//

class ahb_env extends uvm_env;
  `uvm_component_utils(ahb_env)

  ahb_master_monitor master_monitor[];
  ahb_slave_monitor slave_monitor[];
  ahb_scoreboard scoreboard;
  fifo_unit_test fifo_test;

  function new(string name = "ahb_env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);

    // Create arrays of monitors
    master_monitor = new[NO_OF_MASTERS];
    slave_monitor = new[NO_OF_SLAVES];

    foreach (master_monitor[i]) begin
      master_monitor[i] =
        ahb_master_monitor::type_id::create($sformatf("master_monitor[%0d]", i), this);
    end

    foreach (slave_monitor[i]) begin
      slave_monitor[i] =
        ahb_slave_monitor::type_id::create($sformatf("slave_monitor[%0d]", i), this);
    end

    scoreboard = ahb_scoreboard::type_id::create("scoreboard", this);
    fifo_test = fifo_unit_test::type_id::create("fifo_test", this);
  endfunction

  // Key pattern: Connect monitor analysis ports to scoreboard FIFOs
  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);

    // Connect each master monitor to corresponding scoreboard FIFO
    foreach (master_monitor[i]) begin
      master_monitor[i].ahb_master_analysis_port.connect(
        scoreboard.ahb_master_analysis_fifo[i].analysis_export
      );
      `uvm_info(get_type_name(),
                $sformatf("Connected master_monitor[%0d] to scoreboard FIFO[%0d]", i, i),
                UVM_HIGH)
    end

    // Connect each slave monitor to corresponding scoreboard FIFO
    foreach (slave_monitor[i]) begin
      slave_monitor[i].ahb_slave_analysis_port.connect(
        scoreboard.ahb_slave_analysis_fifo[i].analysis_export
      );
      `uvm_info(get_type_name(),
                $sformatf("Connected slave_monitor[%0d] to scoreboard FIFO[%0d]", i, i),
                UVM_HIGH)
    end
  endfunction
endclass

// CHECK: moore.class.classdecl @ahb_env extends @"uvm_pkg::uvm_env"

//===----------------------------------------------------------------------===//
// Test demonstrating complete FIFO flow
//===----------------------------------------------------------------------===//

class ahb_fifo_test extends uvm_test;
  `uvm_component_utils(ahb_fifo_test)

  ahb_env env;

  function new(string name = "ahb_fifo_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = ahb_env::type_id::create("env", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);
    phase.raise_objection(this);
    #1000;  // Allow time for transactions to flow
    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), "TLM FIFO test complete", UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @ahb_fifo_test extends @"uvm_pkg::uvm_test"

//===----------------------------------------------------------------------===//
// Top module for test
//===----------------------------------------------------------------------===//

module tlm_fifo_test_top;
  initial begin
    $display("TLM FIFO pattern test - Iteration 108 Track D");
    run_test("ahb_fifo_test");
  end
endmodule

// CHECK: moore.module @tlm_fifo_test_top
