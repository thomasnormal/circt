// RUN: circt-verilog --parse-only --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s

// Test that UVM stub classes compile correctly and can be used
// in typical UVM testbench patterns

`timescale 1ns/1ps
`include "uvm_macros.svh"
import uvm_pkg::*;

// Test a basic sequence item
class my_seq_item extends uvm_sequence_item;
  `uvm_object_utils(my_seq_item)

  rand bit [31:0] data;
  rand bit [15:0] addr;

  function new(string name = "my_seq_item");
    super.new(name);
  endfunction

  virtual function void do_copy(uvm_object rhs);
    my_seq_item rhs_item;
    super.do_copy(rhs);
    if (!$cast(rhs_item, rhs)) return;
    data = rhs_item.data;
    addr = rhs_item.addr;
  endfunction

  virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
    my_seq_item rhs_item;
    if (!$cast(rhs_item, rhs)) return 0;
    return super.do_compare(rhs, comparer) &&
           data == rhs_item.data &&
           addr == rhs_item.addr;
  endfunction

  virtual function void do_print(uvm_printer printer);
    printer.print_field("data", data, 32, UVM_HEX);
    printer.print_field("addr", addr, 16, UVM_HEX);
  endfunction
endclass

// Test a basic sequence
class my_sequence extends uvm_sequence #(my_seq_item);
  `uvm_object_utils(my_sequence)

  function new(string name = "my_sequence");
    super.new(name);
  endfunction

  virtual task body();
    `uvm_info("SEQ", "Starting sequence", UVM_MEDIUM)
    req = my_seq_item::type_id::create("req");
    start_item(req);
    if (!req.randomize())
      `uvm_error("SEQ", "Randomization failed")
    finish_item(req);
  endtask
endclass

// Test a basic driver
class my_driver extends uvm_driver #(my_seq_item);
  `uvm_component_utils(my_driver)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    forever begin
      seq_item_port.get_next_item(req);
      `uvm_info("DRV", $sformatf("Driving item: data=0x%0h addr=0x%0h", req.data, req.addr), UVM_MEDIUM)
      #10;
      seq_item_port.item_done();
    end
  endtask
endclass

// Test a basic monitor
class my_monitor extends uvm_monitor;
  `uvm_component_utils(my_monitor)

  uvm_analysis_port #(my_seq_item) ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    my_seq_item item;
    item = my_seq_item::type_id::create("item");
    ap.write(item);
  endtask
endclass

// Test a subscriber for coverage
class my_coverage extends uvm_subscriber #(my_seq_item);
  `uvm_component_utils(my_coverage)

  covergroup cg with function sample(my_seq_item item);
    cp_data: coverpoint item.data[7:0];
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cg = new();
  endfunction

  virtual function void write(my_seq_item t);
    `uvm_info("COV", "Sampling coverage", UVM_HIGH)
    cg.sample(t);
  endfunction
endclass

// Test a basic agent
class my_agent extends uvm_agent;
  `uvm_component_utils(my_agent)

  my_driver drv;
  my_monitor mon;
  uvm_sequencer #(my_seq_item) seqr;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (is_active == UVM_ACTIVE) begin
      drv = my_driver::type_id::create("drv", this);
      seqr = uvm_sequencer #(my_seq_item)::type_id::create("seqr", this);
    end
    mon = my_monitor::type_id::create("mon", this);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    if (is_active == UVM_ACTIVE) begin
      drv.seq_item_port.connect(seqr.seq_item_export);
    end
  endfunction
endclass

// Test a basic environment
class my_env extends uvm_env;
  `uvm_component_utils(my_env)

  my_agent agent;
  my_coverage cov;
  uvm_tlm_analysis_fifo #(my_seq_item) analysis_fifo;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    agent = my_agent::type_id::create("agent", this);
    cov = my_coverage::type_id::create("cov", this);
    analysis_fifo = new("analysis_fifo", this);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    agent.mon.ap.connect(cov.analysis_export);
    agent.mon.ap.connect(analysis_fifo.analysis_export);
  endfunction
endclass

// Test a basic test
class my_test extends uvm_test;
  `uvm_component_utils(my_test)

  my_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = my_env::type_id::create("env", this);

    // Test config_db
    uvm_config_db #(uvm_active_passive_enum)::set(this, "env.agent", "is_active", UVM_ACTIVE);
  endfunction

  virtual function void end_of_elaboration_phase(uvm_phase phase);
    super.end_of_elaboration_phase(phase);
    uvm_top.print_topology();
  endfunction

  virtual task run_phase(uvm_phase phase);
    my_sequence seq;
    phase.raise_objection(this, "Starting test");

    seq = my_sequence::type_id::create("seq");
    seq.start(env.agent.seqr);

    phase.drop_objection(this, "Test done");
  endtask
endclass

// Top-level module for testing
module uvm_stub_test_top;
  initial begin
    run_test("my_test");
  end
endmodule
