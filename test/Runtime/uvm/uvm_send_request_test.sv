//===----------------------------------------------------------------------===//
// UVM send_request Virtual Method Test
//===----------------------------------------------------------------------===//
// Tests that send_request is accessible via uvm_sequencer_base pointer.
// This is a regression test for the bug where uvm_sequence_base::send_request
// called m_sequencer.send_request() but m_sequencer was declared as
// uvm_sequencer_base which didn't have the send_request method.
//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package send_request_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Test Transaction
  //==========================================================================
  class simple_item extends uvm_sequence_item;
    `uvm_object_utils(simple_item)

    rand bit [7:0] data;

    function new(string name = "simple_item");
      super.new(name);
    endfunction

    virtual function string convert2string();
      return $sformatf("data=%0h", data);
    endfunction
  endclass

  //==========================================================================
  // Test Driver
  //==========================================================================
  class simple_driver extends uvm_driver #(simple_item);
    `uvm_component_utils(simple_driver)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        seq_item_port.get_next_item(req);
        `uvm_info("DRV", $sformatf("Got: %s", req.convert2string()), UVM_MEDIUM)
        #10ns;
        seq_item_port.item_done();
      end
    endtask
  endclass

  //==========================================================================
  // Test Sequence - exercises send_request indirectly via finish_item
  //==========================================================================
  class simple_sequence extends uvm_sequence #(simple_item);
    `uvm_object_utils(simple_sequence)

    function new(string name = "simple_sequence");
      super.new(name);
    endfunction

    virtual task body();
      simple_item item;

      `uvm_info("SEQ", "Starting simple sequence", UVM_MEDIUM)

      // Create and send an item using start_item/finish_item
      // finish_item internally calls send_request on m_sequencer
      item = simple_item::type_id::create("item");
      start_item(item);
      if (!item.randomize())
        `uvm_error("SEQ", "Randomization failed")
      finish_item(item);

      `uvm_info("SEQ", $sformatf("Sent item: %s", item.convert2string()), UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // Test sequence that explicitly calls send_request
  //==========================================================================
  class explicit_send_request_sequence extends uvm_sequence #(simple_item);
    `uvm_object_utils(explicit_send_request_sequence)

    function new(string name = "explicit_send_request_sequence");
      super.new(name);
    endfunction

    virtual task body();
      simple_item item;

      `uvm_info("SEQ", "Starting explicit send_request sequence", UVM_MEDIUM)

      // First wait for grant
      item = simple_item::type_id::create("item");
      if (!item.randomize())
        `uvm_error("SEQ", "Randomization failed")

      // Wait for grant from sequencer
      wait_for_grant();

      // Explicitly call send_request - this was the broken path
      // m_sequencer is uvm_sequencer_base, but we're calling send_request
      send_request(item);

      // Wait for completion
      wait_for_item_done();

      `uvm_info("SEQ", $sformatf("Explicitly sent item: %s", item.convert2string()), UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // Test Environment
  //==========================================================================
  class test_env extends uvm_env;
    `uvm_component_utils(test_env)

    simple_driver drv;
    uvm_sequencer #(simple_item) seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      drv = simple_driver::type_id::create("drv", this);
      seqr = uvm_sequencer #(simple_item)::type_id::create("seqr", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      drv.seq_item_port.connect(seqr.seq_item_export);
    endfunction
  endclass

  //==========================================================================
  // Test - verifies send_request works via both paths
  //==========================================================================
  class send_request_test extends uvm_test;
    `uvm_component_utils(send_request_test)

    test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = test_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      simple_sequence simple_seq;
      explicit_send_request_sequence explicit_seq;

      phase.raise_objection(this, "Running send_request test");

      // Test 1: Implicit send_request via finish_item
      `uvm_info("TEST", "Test 1: Testing implicit send_request via finish_item", UVM_NONE)
      simple_seq = simple_sequence::type_id::create("simple_seq");
      simple_seq.start(env.seqr);
      `uvm_info("TEST", "Test 1: PASSED", UVM_NONE)

      // Test 2: Explicit send_request call
      `uvm_info("TEST", "Test 2: Testing explicit send_request call", UVM_NONE)
      explicit_seq = explicit_send_request_sequence::type_id::create("explicit_seq");
      explicit_seq.start(env.seqr);
      `uvm_info("TEST", "Test 2: PASSED", UVM_NONE)

      `uvm_info("TEST", "All send_request tests PASSED", UVM_NONE)

      phase.drop_objection(this, "send_request test complete");
    endtask
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import send_request_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting UVM send_request Test", UVM_NONE)
    run_test("send_request_test");
  end

endmodule
