// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_recorder and uvm_text_tr_database.
// Verifies recording APIs don't crash.

// CHECK: [TEST] database created: PASS
// CHECK: [TEST] stream opened: PASS
// CHECK: [TEST] recorder works: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class recorder_test extends uvm_test;
    `uvm_component_utils(recorder_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_text_tr_database db;
      uvm_tr_stream stream;
      uvm_recorder recorder;
      phase.raise_objection(this);

      // Test 1: create database
      db = new("db");
      if (db != null)
        `uvm_info("TEST", "database created: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "database created: FAIL")

      db.open_db();

      // Test 2: open stream
      stream = db.open_stream("test_stream");
      if (stream != null)
        `uvm_info("TEST", "stream opened: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "stream opened: FAIL")

      // Test 3: record fields
      recorder = stream.open_recorder("test_rec");
      if (recorder != null) begin
        recorder.record_field("my_field", 42, 32);
        recorder.record_string("my_string", "hello");
        recorder.close();
        `uvm_info("TEST", "recorder works: PASS", UVM_LOW)
      end else begin
        `uvm_error("TEST", "recorder works: FAIL - null recorder")
      end

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("recorder_test");
endmodule
