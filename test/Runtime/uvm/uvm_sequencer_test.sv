//===----------------------------------------------------------------------===//
// UVM Sequencer Feature Test
//===----------------------------------------------------------------------===//
// Tests enhanced sequencer functionality including:
// - start_item / finish_item sequence execution
// - grab / ungrab for exclusive sequencer access
// - lock / unlock for sequencer arbitration
// - sequencer-driver communication
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package sequencer_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Test Transaction
  //==========================================================================
  class test_item extends uvm_sequence_item;
    `uvm_object_utils(test_item)

    rand bit [7:0] data;
    rand bit [3:0] cmd;
    int sequence_id;  // Track which sequence sent this

    function new(string name = "test_item");
      super.new(name);
      sequence_id = -1;
    endfunction

    virtual function string convert2string();
      return $sformatf("seq_id=%0d, cmd=%0h, data=%0h", sequence_id, cmd, data);
    endfunction

  endclass

  //==========================================================================
  // Test Driver - processes items from sequencer
  //==========================================================================
  class test_driver extends uvm_driver #(test_item);
    `uvm_component_utils(test_driver)

    int items_processed;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      items_processed = 0;
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        // Get next item from sequencer
        seq_item_port.get_next_item(req);

        // Process the item (simulate some work)
        `uvm_info("DRV", $sformatf("Processing: %s", req.convert2string()), UVM_MEDIUM)
        #10ns;
        items_processed++;

        // Signal completion
        seq_item_port.item_done();
      end
    endtask

  endclass

  //==========================================================================
  // Basic Sequence - uses start_item/finish_item
  //==========================================================================
  class basic_sequence extends uvm_sequence #(test_item);
    `uvm_object_utils(basic_sequence)

    int seq_id;
    int num_items = 3;

    function new(string name = "basic_sequence");
      super.new(name);
      seq_id = 0;
    endfunction

    virtual task body();
      test_item item;
      `uvm_info("SEQ", $sformatf("Starting basic_sequence (id=%0d)", seq_id), UVM_MEDIUM)

      repeat (num_items) begin
        item = test_item::type_id::create("item");

        // Standard sequence execution pattern
        start_item(item);
        if (!item.randomize())
          `uvm_error("SEQ", "Randomization failed")
        item.sequence_id = seq_id;
        finish_item(item);

        `uvm_info("SEQ", $sformatf("Sent item: %s", item.convert2string()), UVM_MEDIUM)
      end

      `uvm_info("SEQ", $sformatf("Finished basic_sequence (id=%0d)", seq_id), UVM_MEDIUM)
    endtask

  endclass

  //==========================================================================
  // Locking Sequence - uses lock/unlock for exclusive access
  //==========================================================================
  class locking_sequence extends uvm_sequence #(test_item);
    `uvm_object_utils(locking_sequence)

    int seq_id;

    function new(string name = "locking_sequence");
      super.new(name);
      seq_id = 100;
    endfunction

    virtual task body();
      test_item item;
      `uvm_info("SEQ", $sformatf("Locking sequence %0d acquiring lock", seq_id), UVM_MEDIUM)

      // Acquire lock for exclusive access
      lock();

      `uvm_info("SEQ", $sformatf("Locking sequence %0d has lock", seq_id), UVM_MEDIUM)

      // Verify we have the lock
      if (!has_lock())
        `uvm_error("SEQ", "Expected to have lock but don't")

      // Send items while holding lock
      repeat (2) begin
        item = test_item::type_id::create("item");
        start_item(item);
        if (!item.randomize())
          `uvm_error("SEQ", "Randomization failed")
        item.sequence_id = seq_id;
        item.cmd = 4'hF;  // Mark as locked item
        finish_item(item);
      end

      // Release lock
      unlock();
      `uvm_info("SEQ", $sformatf("Locking sequence %0d released lock", seq_id), UVM_MEDIUM)
    endtask

  endclass

  //==========================================================================
  // Grabbing Sequence - uses grab/ungrab for immediate exclusive access
  //==========================================================================
  class grabbing_sequence extends uvm_sequence #(test_item);
    `uvm_object_utils(grabbing_sequence)

    int seq_id;

    function new(string name = "grabbing_sequence");
      super.new(name);
      seq_id = 200;
    endfunction

    virtual task body();
      test_item item;
      `uvm_info("SEQ", $sformatf("Grabbing sequence %0d grabbing sequencer", seq_id), UVM_MEDIUM)

      // Grab for immediate exclusive access
      grab();

      `uvm_info("SEQ", $sformatf("Grabbing sequence %0d has grabbed", seq_id), UVM_MEDIUM)

      // Verify we have the lock (grab gives us a lock)
      if (!has_lock())
        `uvm_error("SEQ", "Expected to have lock after grab but don't")

      // Send high-priority items
      repeat (2) begin
        item = test_item::type_id::create("item");
        start_item(item);
        if (!item.randomize())
          `uvm_error("SEQ", "Randomization failed")
        item.sequence_id = seq_id;
        item.cmd = 4'hE;  // Mark as grabbed item
        finish_item(item);
      end

      // Release grab
      ungrab();
      `uvm_info("SEQ", $sformatf("Grabbing sequence %0d released grab", seq_id), UVM_MEDIUM)
    endtask

  endclass

  //==========================================================================
  // Test Agent
  //==========================================================================
  class test_agent extends uvm_agent;
    `uvm_component_utils(test_agent)

    test_driver drv;
    uvm_sequencer #(test_item) seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      drv = test_driver::type_id::create("drv", this);
      seqr = uvm_sequencer #(test_item)::type_id::create("seqr", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      drv.seq_item_port.connect(seqr.seq_item_export);
    endfunction

  endclass

  //==========================================================================
  // Test Environment
  //==========================================================================
  class test_env extends uvm_env;
    `uvm_component_utils(test_env)

    test_agent agent;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent = test_agent::type_id::create("agent", this);
    endfunction

  endclass

  //==========================================================================
  // Basic Sequence Test
  //==========================================================================
  class basic_seq_test extends uvm_test;
    `uvm_component_utils(basic_seq_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      basic_sequence seq;

      phase.raise_objection(this, "Running basic sequence test");

      `uvm_info("TEST", "Starting basic sequence test", UVM_NONE)

      seq = basic_sequence::type_id::create("seq");
      seq.seq_id = 1;
      seq.start(env.agent.seqr);

      `uvm_info("TEST", "Basic sequence test completed", UVM_NONE)

      phase.drop_objection(this, "Basic sequence test done");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", $sformatf("Driver processed %0d items",
                                  env.agent.drv.items_processed), UVM_NONE)
    endfunction

  endclass

  //==========================================================================
  // Lock/Unlock Test
  //==========================================================================
  class lock_test extends uvm_test;
    `uvm_component_utils(lock_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      locking_sequence lock_seq;
      basic_sequence basic_seq;

      phase.raise_objection(this, "Running lock test");

      `uvm_info("TEST", "Starting lock test", UVM_NONE)

      // Run locking sequence
      lock_seq = locking_sequence::type_id::create("lock_seq");
      lock_seq.start(env.agent.seqr);

      // Run basic sequence after lock released
      basic_seq = basic_sequence::type_id::create("basic_seq");
      basic_seq.seq_id = 2;
      basic_seq.num_items = 2;
      basic_seq.start(env.agent.seqr);

      `uvm_info("TEST", "Lock test completed", UVM_NONE)

      phase.drop_objection(this, "Lock test done");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", $sformatf("Driver processed %0d items",
                                  env.agent.drv.items_processed), UVM_NONE)
    endfunction

  endclass

  //==========================================================================
  // Grab/Ungrab Test
  //==========================================================================
  class grab_test extends uvm_test;
    `uvm_component_utils(grab_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      grabbing_sequence grab_seq;
      basic_sequence basic_seq;

      phase.raise_objection(this, "Running grab test");

      `uvm_info("TEST", "Starting grab test", UVM_NONE)

      // Run grabbing sequence
      grab_seq = grabbing_sequence::type_id::create("grab_seq");
      grab_seq.start(env.agent.seqr);

      // Run basic sequence after grab released
      basic_seq = basic_sequence::type_id::create("basic_seq");
      basic_seq.seq_id = 3;
      basic_seq.num_items = 2;
      basic_seq.start(env.agent.seqr);

      `uvm_info("TEST", "Grab test completed", UVM_NONE)

      phase.drop_objection(this, "Grab test done");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", $sformatf("Driver processed %0d items",
                                  env.agent.drv.items_processed), UVM_NONE)
    endfunction

  endclass

  //==========================================================================
  // Combined Sequencer Features Test
  //==========================================================================
  class sequencer_features_test extends uvm_test;
    `uvm_component_utils(sequencer_features_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      basic_sequence basic_seq1, basic_seq2;
      locking_sequence lock_seq;
      grabbing_sequence grab_seq;

      phase.raise_objection(this, "Running sequencer features test");

      `uvm_info("TEST", "=== Starting Sequencer Features Test ===", UVM_NONE)

      // Test 1: Basic sequence execution
      `uvm_info("TEST", "--- Test 1: Basic sequence execution ---", UVM_NONE)
      basic_seq1 = basic_sequence::type_id::create("basic_seq1");
      basic_seq1.seq_id = 1;
      basic_seq1.num_items = 2;
      basic_seq1.start(env.agent.seqr);

      // Test 2: Lock/unlock
      `uvm_info("TEST", "--- Test 2: Lock/unlock ---", UVM_NONE)
      lock_seq = locking_sequence::type_id::create("lock_seq");
      lock_seq.start(env.agent.seqr);

      // Test 3: Grab/ungrab
      `uvm_info("TEST", "--- Test 3: Grab/ungrab ---", UVM_NONE)
      grab_seq = grabbing_sequence::type_id::create("grab_seq");
      grab_seq.start(env.agent.seqr);

      // Test 4: Another basic sequence
      `uvm_info("TEST", "--- Test 4: Final basic sequence ---", UVM_NONE)
      basic_seq2 = basic_sequence::type_id::create("basic_seq2");
      basic_seq2.seq_id = 4;
      basic_seq2.num_items = 2;
      basic_seq2.start(env.agent.seqr);

      // Test 5: Verify arbitration mode can be set
      `uvm_info("TEST", "--- Test 5: Arbitration mode ---", UVM_NONE)
      env.agent.seqr.set_arbitration(UVM_SEQ_ARB_RANDOM);
      if (env.agent.seqr.get_arbitration() != UVM_SEQ_ARB_RANDOM)
        `uvm_error("TEST", "Arbitration mode not set correctly")
      else
        `uvm_info("TEST", "Arbitration mode set to RANDOM", UVM_MEDIUM)

      `uvm_info("TEST", "=== Sequencer Features Test Completed ===", UVM_NONE)

      phase.drop_objection(this, "Sequencer features test done");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", $sformatf("Total items processed by driver: %0d",
                                  env.agent.drv.items_processed), UVM_NONE)
      if (env.agent.drv.items_processed >= 10)
        `uvm_info("TEST", "TEST PASSED", UVM_NONE)
      else
        `uvm_error("TEST", $sformatf("TEST FAILED - expected at least 10 items, got %0d",
                                     env.agent.drv.items_processed))
    endfunction

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import sequencer_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting UVM Sequencer Test", UVM_NONE)
    run_test("sequencer_features_test");
  end

endmodule
