// RUN: circt-verilog --ir-moore --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s 2>&1 | FileCheck %s
// REQUIRES: slang

// This test verifies that the UVM stub package can be successfully imported
// and that basic UVM patterns compile correctly.

`timescale 1ns/1ps
`include "uvm_macros.svh"
import uvm_pkg::*;

//===----------------------------------------------------------------------===//
// Basic UVM object and component test
//===----------------------------------------------------------------------===//

// CHECK: moore.class.classdecl @my_transaction extends @"uvm_pkg::uvm_sequence_item"
class my_transaction extends uvm_sequence_item;
  `uvm_object_utils(my_transaction)

  rand bit [7:0] data;
  rand bit [3:0] addr;

  function new(string name = "my_transaction");
    super.new(name);
  endfunction

  virtual function string convert2string();
    return $sformatf("addr=%0h data=%0h", addr, data);
  endfunction
endclass

// CHECK: moore.class.classdecl @my_driver extends @"uvm_pkg::uvm_driver"
class my_driver extends uvm_driver #(my_transaction);
  `uvm_component_utils(my_driver)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    forever begin
      my_transaction tr;
      seq_item_port.get_next_item(tr);
      `uvm_info("DRV", $sformatf("Driving: %s", tr.convert2string()), UVM_MEDIUM)
      #10;
      seq_item_port.item_done();
    end
  endtask
endclass

// CHECK: moore.class.classdecl @my_monitor extends @"uvm_pkg::uvm_monitor"
class my_monitor extends uvm_monitor;
  `uvm_component_utils(my_monitor)

  uvm_analysis_port #(my_transaction) ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    my_transaction tr;
    forever begin
      tr = new("mon_tr");
      #20;
      ap.write(tr);
    end
  endtask
endclass

// CHECK: moore.class.classdecl @my_agent extends @"uvm_pkg::uvm_agent"
class my_agent extends uvm_agent;
  `uvm_component_utils(my_agent)

  my_driver drv;
  my_monitor mon;
  uvm_sequencer #(my_transaction) seqr;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (is_active == UVM_ACTIVE) begin
      drv = my_driver::type_id::create("drv", this);
      seqr = uvm_sequencer #(my_transaction)::type_id::create("seqr", this);
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

// CHECK: moore.class.classdecl @my_scoreboard extends @"uvm_pkg::uvm_scoreboard"
class my_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(my_scoreboard)

  uvm_analysis_imp #(my_transaction, my_scoreboard) analysis_export;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    analysis_export = new("analysis_export", this);
  endfunction

  virtual function void write(my_transaction tr);
    `uvm_info("SCB", $sformatf("Received: %s", tr.convert2string()), UVM_HIGH)
  endfunction
endclass

// CHECK: moore.class.classdecl @my_env extends @"uvm_pkg::uvm_env"
class my_env extends uvm_env;
  `uvm_component_utils(my_env)

  my_agent agent;
  my_scoreboard scb;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    agent = my_agent::type_id::create("agent", this);
    scb = my_scoreboard::type_id::create("scb", this);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    agent.mon.ap.connect(scb.analysis_export);
  endfunction
endclass

// CHECK: moore.class.classdecl @my_test extends @"uvm_pkg::uvm_test"
class my_test extends uvm_test;
  `uvm_component_utils(my_test)

  my_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = my_env::type_id::create("env", this);
  endfunction

  virtual task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    `uvm_info("TEST", "Test starting", UVM_LOW)
    #100;
    `uvm_info("TEST", "Test complete", UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass

//===----------------------------------------------------------------------===//
// Test sequence
//===----------------------------------------------------------------------===//

// CHECK: moore.class.classdecl @my_sequence extends @"uvm_pkg::uvm_sequence_{{[0-9]+}}"
class my_sequence extends uvm_sequence #(my_transaction);
  `uvm_object_utils(my_sequence)

  function new(string name = "my_sequence");
    super.new(name);
  endfunction

  virtual task body();
    my_transaction tr;
    repeat(5) begin
      `uvm_create(tr)
      void'(tr.randomize());
      `uvm_send(tr)
    end
  endtask
endclass

//===----------------------------------------------------------------------===//
// Test config_db
//===----------------------------------------------------------------------===//

// CHECK: moore.class.classdecl @my_config extends @"uvm_pkg::uvm_object"
class my_config extends uvm_object;
  `uvm_object_utils(my_config)

  bit [7:0] max_addr = 8'hFF;
  int unsigned timeout = 1000;

  function new(string name = "my_config");
    super.new(name);
  endfunction
endclass

// CHECK: moore.module @uvm_stub_test() {
module uvm_stub_test;
  initial begin
    my_config cfg = new("cfg");
    uvm_config_db #(my_config)::set(null, "*", "config", cfg);
    run_test("my_test");
  end
endmodule
