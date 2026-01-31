// RUN: circt-verilog --ir-moore --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s 2>&1 | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Multi-Agent Virtual Sequence Test - Iteration 107 Track D
//===----------------------------------------------------------------------===//
//
// This test validates multi-agent interaction patterns commonly found in
// production AVIPs (like mbit AXI4, APB, SPI, etc.). These patterns are:
//
// 1. Virtual sequences with multiple agent sequencer handles
// 2. fork/join_none with forever loops (for slave responders)
// 3. fork/join for master transaction coordination
// 4. p_sequencer macro usage for accessing sub-sequencers
// 5. Repeat loops for transaction counts
// 6. Named fork blocks for debugging
//
// This represents common patterns from:
// - AXI4 AVIP: Master write/read + Slave responders running concurrently
// - APB AVIP: Master driver + Slave response sequences
// - SPI AVIP: Controller + Target sequences
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps
`include "uvm_macros.svh"

import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Simple transaction class for testing
//===----------------------------------------------------------------------===//

class simple_txn extends uvm_sequence_item;
  `uvm_object_utils(simple_txn)

  rand bit [31:0] addr;
  rand bit [31:0] data;
  rand bit        write;

  constraint c_addr_range {
    addr inside {[32'h0:32'hFF]};
  }

  function new(string name = "simple_txn");
    super.new(name);
  endfunction

  virtual function string convert2string();
    return $sformatf("TXN: %s addr=0x%08h data=0x%08h",
                     write ? "WR" : "RD", addr, data);
  endfunction
endclass

// CHECK: moore.class.classdecl @simple_txn

//===----------------------------------------------------------------------===//
// Sequencers for each agent type
//===----------------------------------------------------------------------===//

// Master sequencer (for driving transactions)
class master_sequencer extends uvm_sequencer #(simple_txn);
  `uvm_component_utils(master_sequencer)

  function new(string name = "master_sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

// CHECK: moore.class.classdecl @master_sequencer

// Slave sequencer (for responding to transactions)
class slave_sequencer extends uvm_sequencer #(simple_txn);
  `uvm_component_utils(slave_sequencer)

  function new(string name = "slave_sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

// CHECK: moore.class.classdecl @slave_sequencer

//===----------------------------------------------------------------------===//
// Virtual sequencer with multiple agent handles
// This pattern is used in mbit AVIPs (axi4_virtual_sequencer, etc.)
//===----------------------------------------------------------------------===//

class multi_agent_virtual_sequencer extends uvm_sequencer #(uvm_sequence_item);
  `uvm_component_utils(multi_agent_virtual_sequencer)

  // Multiple sequencer handles - pattern from mbit AXI4 AVIP
  master_sequencer master_write_seqr_h;
  master_sequencer master_read_seqr_h;
  slave_sequencer  slave_write_seqr_h;
  slave_sequencer  slave_read_seqr_h;

  function new(string name = "multi_agent_virtual_sequencer", uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

// CHECK: moore.class.classdecl @multi_agent_virtual_sequencer

//===----------------------------------------------------------------------===//
// Simple sequences for master and slave agents
//===----------------------------------------------------------------------===//

// Base sequence for master operations
class master_write_seq extends uvm_sequence #(simple_txn);
  `uvm_object_utils(master_write_seq)

  function new(string name = "master_write_seq");
    super.new(name);
  endfunction

  virtual task body();
    simple_txn item;
    item = simple_txn::type_id::create("item");
    start_item(item);
    if (!item.randomize() with { write == 1'b1; }) begin
      `uvm_error("MASTER_WRITE", "Randomization failed")
    end
    finish_item(item);
    `uvm_info("MASTER_WRITE", item.convert2string(), UVM_MEDIUM)
  endtask
endclass

// CHECK: moore.class.classdecl @master_write_seq

class master_read_seq extends uvm_sequence #(simple_txn);
  `uvm_object_utils(master_read_seq)

  function new(string name = "master_read_seq");
    super.new(name);
  endfunction

  virtual task body();
    simple_txn item;
    item = simple_txn::type_id::create("item");
    start_item(item);
    if (!item.randomize() with { write == 1'b0; }) begin
      `uvm_error("MASTER_READ", "Randomization failed")
    end
    finish_item(item);
    `uvm_info("MASTER_READ", item.convert2string(), UVM_MEDIUM)
  endtask
endclass

// CHECK: moore.class.classdecl @master_read_seq

// Slave response sequences (reactive)
class slave_write_response_seq extends uvm_sequence #(simple_txn);
  `uvm_object_utils(slave_write_response_seq)

  function new(string name = "slave_write_response_seq");
    super.new(name);
  endfunction

  virtual task body();
    simple_txn item;
    item = simple_txn::type_id::create("resp_item");
    start_item(item);
    // Slave just acknowledges
    item.write = 1'b1;
    item.data = 32'hACCE5500;
    finish_item(item);
    `uvm_info("SLAVE_WRITE_RESP", "Write response sent", UVM_HIGH)
  endtask
endclass

// CHECK: moore.class.classdecl @slave_write_response_seq

class slave_read_response_seq extends uvm_sequence #(simple_txn);
  `uvm_object_utils(slave_read_response_seq)

  function new(string name = "slave_read_response_seq");
    super.new(name);
  endfunction

  virtual task body();
    simple_txn item;
    item = simple_txn::type_id::create("resp_item");
    start_item(item);
    // Slave provides read data
    item.write = 1'b0;
    item.data = 32'hDEADBEEF;
    finish_item(item);
    `uvm_info("SLAVE_READ_RESP", "Read response sent", UVM_HIGH)
  endtask
endclass

// CHECK: moore.class.classdecl @slave_read_response_seq

//===----------------------------------------------------------------------===//
// Virtual base sequence - pattern from mbit axi4_virtual_base_seq
// Uses `uvm_declare_p_sequencer to get typed access to virtual sequencer
//===----------------------------------------------------------------------===//

class virtual_base_seq extends uvm_sequence #(uvm_sequence_item);
  `uvm_object_utils(virtual_base_seq)
  `uvm_declare_p_sequencer(multi_agent_virtual_sequencer)

  function new(string name = "virtual_base_seq");
    super.new(name);
  endfunction

  virtual task body();
    // Base implementation - just validate p_sequencer cast
    if (!$cast(p_sequencer, m_sequencer)) begin
      `uvm_error(get_full_name(), "Virtual sequencer pointer cast failed")
    end
  endtask
endclass

// CHECK: moore.class.classdecl @virtual_base_seq

//===----------------------------------------------------------------------===//
// Multi-Agent Virtual Sequence - Core Pattern from mbit AVIPs
// This is the key pattern: fork/join_none for slave responders,
// fork/join for master transactions
//===----------------------------------------------------------------------===//

class multi_agent_virtual_seq extends virtual_base_seq;
  `uvm_object_utils(multi_agent_virtual_seq)

  // Sequence handles - pattern from axi4_virtual_bk_write_read_seq
  master_write_seq         master_wr_seq_h;
  master_read_seq          master_rd_seq_h;
  slave_write_response_seq slave_wr_resp_seq_h;
  slave_read_response_seq  slave_rd_resp_seq_h;

  // Configuration
  int num_write_txns = 2;
  int num_read_txns = 3;

  function new(string name = "multi_agent_virtual_seq");
    super.new(name);
  endfunction

  virtual task body();
    // Call base class body first (validates p_sequencer)
    super.body();

    // Create all sequences
    master_wr_seq_h = master_write_seq::type_id::create("master_wr_seq_h");
    master_rd_seq_h = master_read_seq::type_id::create("master_rd_seq_h");
    slave_wr_resp_seq_h = slave_write_response_seq::type_id::create("slave_wr_resp_seq_h");
    slave_rd_resp_seq_h = slave_read_response_seq::type_id::create("slave_rd_resp_seq_h");

    `uvm_info(get_type_name(), "Starting multi-agent virtual sequence", UVM_NONE)

    //=========================================================================
    // Pattern 1: fork/join_none with forever loops for slave responders
    // This is the key AVIP pattern - slaves respond indefinitely until
    // the master transactions complete and the test ends
    //=========================================================================
    fork
      begin : SLAVE_WRITE_RESPONDER
        forever begin
          slave_wr_resp_seq_h.start(p_sequencer.slave_write_seqr_h);
        end
      end
      begin : SLAVE_READ_RESPONDER
        forever begin
          slave_rd_resp_seq_h.start(p_sequencer.slave_read_seqr_h);
        end
      end
    join_none

    //=========================================================================
    // Pattern 2: fork/join for master transactions
    // Master write and read sequences run in parallel but both must complete
    //=========================================================================
    fork
      begin : MASTER_WRITES
        repeat (num_write_txns) begin
          master_wr_seq_h.start(p_sequencer.master_write_seqr_h);
        end
        `uvm_info(get_type_name(),
                  $sformatf("Completed %0d write transactions", num_write_txns),
                  UVM_LOW)
      end
      begin : MASTER_READS
        repeat (num_read_txns) begin
          master_rd_seq_h.start(p_sequencer.master_read_seqr_h);
        end
        `uvm_info(get_type_name(),
                  $sformatf("Completed %0d read transactions", num_read_txns),
                  UVM_LOW)
      end
    join

    `uvm_info(get_type_name(), "Multi-agent virtual sequence complete", UVM_NONE)
  endtask
endclass

// CHECK: moore.class.classdecl @multi_agent_virtual_seq

//===----------------------------------------------------------------------===//
// Test class that runs the virtual sequence
//===----------------------------------------------------------------------===//

class multi_agent_test extends uvm_test;
  `uvm_component_utils(multi_agent_test)

  multi_agent_virtual_sequencer v_seqr;

  function new(string name = "multi_agent_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    v_seqr = multi_agent_virtual_sequencer::type_id::create("v_seqr", this);
    // In a real test, sub-sequencers would be created by agents
    // and connected in connect_phase
  endfunction

  virtual task run_phase(uvm_phase phase);
    multi_agent_virtual_seq vseq;

    phase.raise_objection(this);
    `uvm_info(get_type_name(), "Starting multi-agent test", UVM_LOW)

    vseq = multi_agent_virtual_seq::type_id::create("vseq");
    vseq.num_write_txns = 5;
    vseq.num_read_txns = 3;
    vseq.start(v_seqr);

    #100;
    `uvm_info(get_type_name(), "Multi-agent test complete", UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass

// CHECK: moore.class.classdecl @multi_agent_test

//===----------------------------------------------------------------------===//
// Simple top module for completeness
//===----------------------------------------------------------------------===//

module multi_agent_tb_top;
  initial begin
    $display("Multi-Agent Virtual Sequence Test - Iteration 107");
    run_test("multi_agent_test");
  end
endmodule

// CHECK: moore.module @multi_agent_tb_top
