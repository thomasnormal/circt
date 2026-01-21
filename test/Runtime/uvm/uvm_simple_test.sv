//===----------------------------------------------------------------------===//
// Simple UVM Testbench to verify CIRCT UVM stubs
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package simple_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Simple Transaction
  //==========================================================================
  class simple_tx extends uvm_sequence_item;
    `uvm_object_utils(simple_tx)

    rand bit [7:0] data;
    rand bit [3:0] addr;

    function new(string name = "simple_tx");
      super.new(name);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      simple_tx rhs_tx;
      super.do_copy(rhs);
      if (!$cast(rhs_tx, rhs))
        `uvm_fatal("CAST", "Cast failed")
      data = rhs_tx.data;
      addr = rhs_tx.addr;
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      simple_tx rhs_tx;
      if (!$cast(rhs_tx, rhs))
        return 0;
      return (data == rhs_tx.data) && (addr == rhs_tx.addr);
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=%0h, data=%0h", addr, data);
    endfunction

  endclass

  //==========================================================================
  // Simple Driver
  //==========================================================================
  class simple_driver extends uvm_driver #(simple_tx);
    `uvm_component_utils(simple_driver)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      forever begin
        seq_item_port.get_next_item(req);
        `uvm_info("DRV", $sformatf("Driving: %s", req.convert2string()), UVM_MEDIUM)
        seq_item_port.item_done();
      end
    endtask

  endclass

  //==========================================================================
  // Simple Monitor
  //==========================================================================
  class simple_monitor extends uvm_monitor;
    `uvm_component_utils(simple_monitor)

    uvm_analysis_port #(simple_tx) ap;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
    endfunction

  endclass

  //==========================================================================
  // Simple Agent
  //==========================================================================
  class simple_agent extends uvm_agent;
    `uvm_component_utils(simple_agent)

    simple_driver drv;
    simple_monitor mon;
    uvm_sequencer #(simple_tx) seqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (is_active == UVM_ACTIVE) begin
        drv = simple_driver::type_id::create("drv", this);
        seqr = uvm_sequencer #(simple_tx)::type_id::create("seqr", this);
      end
      mon = simple_monitor::type_id::create("mon", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      if (is_active == UVM_ACTIVE) begin
        drv.seq_item_port.connect(seqr.seq_item_export);
      end
    endfunction

  endclass

  //==========================================================================
  // Simple Subscriber (Coverage)
  //==========================================================================
  class simple_subscriber extends uvm_subscriber #(simple_tx);
    `uvm_component_utils(simple_subscriber)

    int tx_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      tx_count = 0;
    endfunction

    virtual function void write(simple_tx t);
      tx_count++;
      `uvm_info("SUB", $sformatf("Received tx #%0d: %s", tx_count, t.convert2string()), UVM_MEDIUM)
    endfunction

  endclass

  //==========================================================================
  // Simple Scoreboard
  //==========================================================================
  class simple_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(simple_scoreboard)

    uvm_tlm_analysis_fifo #(simple_tx) fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("fifo", this);
    endfunction

  endclass

  //==========================================================================
  // Simple Environment
  //==========================================================================
  class simple_env extends uvm_env;
    `uvm_component_utils(simple_env)

    simple_agent agent;
    simple_scoreboard sb;
    simple_subscriber sub;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent = simple_agent::type_id::create("agent", this);
      sb = simple_scoreboard::type_id::create("sb", this);
      sub = simple_subscriber::type_id::create("sub", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      agent.mon.ap.connect(sb.fifo.analysis_export);
      agent.mon.ap.connect(sub.analysis_export);
    endfunction

  endclass

  //==========================================================================
  // Simple Sequence
  //==========================================================================
  class simple_sequence extends uvm_sequence #(simple_tx);
    `uvm_object_utils(simple_sequence)

    function new(string name = "simple_sequence");
      super.new(name);
    endfunction

    virtual task body();
      simple_tx tx;
      repeat (5) begin
        tx = simple_tx::type_id::create("tx");
        start_item(tx);
        if (!tx.randomize())
          `uvm_error("SEQ", "Randomization failed")
        finish_item(tx);
      end
    endtask

  endclass

  //==========================================================================
  // Simple Test
  //==========================================================================
  class simple_test extends uvm_test;
    `uvm_component_utils(simple_test)

    simple_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = simple_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      simple_sequence seq;
      phase.raise_objection(this, "Starting test");
      seq = simple_sequence::type_id::create("seq");
      seq.start(env.agent.seqr);
      phase.drop_objection(this, "Finished test");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "Test completed successfully", UVM_NONE)
    endfunction

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import simple_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting UVM test", UVM_NONE)
    run_test("simple_test");
  end

endmodule
