// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s
// REQUIRES: slang
// XFAIL: *

//===----------------------------------------------------------------------===//
// Scoreboard Pattern Test - Iteration 107 Track D
//===----------------------------------------------------------------------===//
//
// This test validates scoreboard patterns commonly found in production AVIPs
// (like mbit AXI4, APB, etc.). These patterns include:
//
// 1. uvm_tlm_analysis_fifo for transaction buffering
// 2. Semaphores for synchronization between comparison tasks
// 3. fork/join for parallel comparison tasks
// 4. forever loops with FIFO.get() blocking calls
// 5. Statistics counters for verification results
//
// Pattern source: mbit AXI4 axi4_scoreboard.sv
//
//===----------------------------------------------------------------------===//

`include "uvm_macros.svh"

import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Transaction class for testing
//===----------------------------------------------------------------------===//

class simple_txn extends uvm_sequence_item;
  `uvm_object_utils(simple_txn)

  bit [31:0] addr;
  bit [31:0] data;
  bit write;

  function new(string name = "simple_txn");
    super.new(name);
  endfunction
endclass

// CHECK: moore.class.classdecl @simple_txn

//===----------------------------------------------------------------------===//
// Scoreboard pattern from mbit AXI4 AVIP
//===----------------------------------------------------------------------===//

class axi4_style_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(axi4_style_scoreboard)

  // Pattern from axi4_scoreboard.sv: multiple analysis FIFOs
  uvm_tlm_analysis_fifo #(simple_txn) master_write_fifo;
  uvm_tlm_analysis_fifo #(simple_txn) slave_write_fifo;
  uvm_tlm_analysis_fifo #(simple_txn) master_read_fifo;
  uvm_tlm_analysis_fifo #(simple_txn) slave_read_fifo;

  // Semaphores for synchronization (pattern from axi4_scoreboard.sv)
  semaphore write_key;
  semaphore read_key;

  // Statistics counters
  int master_tx_count;
  int slave_tx_count;
  int verified_count;
  int failed_count;

  function new(string name = "axi4_style_scoreboard", uvm_component parent = null);
    super.new(name, parent);
    master_write_fifo = new("master_write_fifo", this);
    slave_write_fifo = new("slave_write_fifo", this);
    master_read_fifo = new("master_read_fifo", this);
    slave_read_fifo = new("slave_read_fifo", this);
    write_key = new(1);
    read_key = new(1);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
  endfunction

  virtual task run_phase(uvm_phase phase);
    super.run_phase(phase);

    // Pattern from axi4_scoreboard: fork parallel comparison tasks
    fork
      compare_write_transactions();
      compare_read_transactions();
    join
  endtask

  virtual task compare_write_transactions();
    simple_txn master_txn;
    simple_txn slave_txn;

    forever begin
      write_key.get(1);
      master_write_fifo.get(master_txn);
      `uvm_info(get_type_name(), $sformatf("Master write: addr=0x%08h data=0x%08h",
                master_txn.addr, master_txn.data), UVM_HIGH)
      slave_write_fifo.get(slave_txn);
      `uvm_info(get_type_name(), $sformatf("Slave write: addr=0x%08h data=0x%08h",
                slave_txn.addr, slave_txn.data), UVM_HIGH)

      // Compare
      if (master_txn.addr == slave_txn.addr && master_txn.data == slave_txn.data) begin
        verified_count++;
        `uvm_info(get_type_name(), "Write transaction MATCH", UVM_MEDIUM)
      end else begin
        failed_count++;
        `uvm_error(get_type_name(), "Write transaction MISMATCH")
      end

      master_tx_count++;
      slave_tx_count++;
      write_key.put(1);
    end
  endtask

  virtual task compare_read_transactions();
    simple_txn master_txn;
    simple_txn slave_txn;

    forever begin
      read_key.get(1);
      master_read_fifo.get(master_txn);
      slave_read_fifo.get(slave_txn);

      // Compare
      if (master_txn.addr == slave_txn.addr && master_txn.data == slave_txn.data) begin
        verified_count++;
      end else begin
        failed_count++;
      end

      read_key.put(1);
    end
  endtask

  function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    `uvm_info(get_type_name(), $sformatf("Verified: %0d, Failed: %0d", verified_count, failed_count), UVM_LOW)
  endfunction
endclass

// CHECK: moore.class.classdecl @axi4_style_scoreboard extends @"uvm_pkg::uvm_scoreboard"

//===----------------------------------------------------------------------===//
// Top module for test
//===----------------------------------------------------------------------===//

module scoreboard_test_top;
  initial begin
    $display("Scoreboard pattern test - Iteration 107");
  end
endmodule

// CHECK: moore.module @scoreboard_test_top
