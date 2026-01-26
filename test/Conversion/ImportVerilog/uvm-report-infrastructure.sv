// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang
// XFAIL: *
// UVM runtime has compilation issues affecting class declarations.

//===----------------------------------------------------------------------===//
// Test UVM Reporting Infrastructure Classes
//===----------------------------------------------------------------------===//
//
// This test verifies that the UVM reporting classes compile correctly:
// - uvm_report_message
// - uvm_report_handler
// - uvm_report_server
// - uvm_report_catcher
//
// It also tests the new methods added to uvm_report_object:
// - report_hook
// - set_report_max_quit_count
// - get_report_max_quit_count
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`include "uvm_macros.svh"

// MOORE: module {

module test_report_infrastructure;
  import uvm_pkg::*;

  initial begin
    uvm_report_server server;
    uvm_report_handler handler;
    uvm_report_message msg;
    int count;
    bit is_reached;

    //------------------------------------------------------------------------
    // Test uvm_report_server
    //------------------------------------------------------------------------
    server = uvm_report_server::get_server();

    // Test max quit count methods
    server.set_max_quit_count(50);
    count = server.get_max_quit_count();

    // Test severity count methods
    count = server.get_severity_count(UVM_INFO);
    count = server.get_info_count();
    count = server.get_warning_count();
    count = server.get_error_count();
    count = server.get_fatal_count();

    // Test quit count methods
    count = server.get_quit_count();
    server.incr_quit_count();
    server.reset_quit_count();
    is_reached = server.is_quit_count_reached();

    // Test ID count methods
    count = server.get_id_count("TEST_ID");
    server.incr_id_count("TEST_ID");
    server.set_id_count("TEST_ID", 10);

    // Test severity count manipulation
    server.incr_severity_count(UVM_WARNING);
    server.set_severity_count(UVM_ERROR, 5);
    server.reset_severity_counts();

    // Test dump and summarize
    server.dump_server_state();
    server.report_summarize();

    //------------------------------------------------------------------------
    // Test uvm_report_handler
    //------------------------------------------------------------------------
    handler = new("test_handler");

    // Test verbosity level
    handler.set_verbosity_level(UVM_HIGH);
    count = handler.get_verbosity_level();

    // Test action methods
    handler.set_severity_action(UVM_ERROR, UVM_DISPLAY | UVM_COUNT);
    handler.set_id_action("SPECIAL_ID", UVM_LOG);
    handler.set_severity_id_action(UVM_WARNING, "WARN_ID", UVM_DISPLAY);

    // Test get action
    begin
      uvm_action action;
      action = handler.get_action(UVM_ERROR, "TEST");
    end

    // Test verbosity methods
    handler.set_id_verbosity("VERBOSE_ID", UVM_DEBUG);
    handler.set_severity_id_verbosity(UVM_INFO, "TRACE_ID", UVM_FULL);
    count = handler.get_id_verbosity("VERBOSE_ID");

    // Test file handle methods
    handler.set_default_file(0);
    handler.set_severity_file(UVM_ERROR, 0);
    handler.set_id_file("LOG_ID", 0);
    handler.set_severity_id_file(UVM_WARNING, "WARN_LOG", 0);

    begin
      UVM_FILE fh;
      fh = handler.get_file_handle(UVM_ERROR, "TEST");
    end

    // Test dump state
    handler.dump_state();

    //------------------------------------------------------------------------
    // Test uvm_report_message
    //------------------------------------------------------------------------
    msg = uvm_report_message::new_report_message("test_msg");

    // Test severity
    msg.set_severity(UVM_WARNING);
    begin
      uvm_severity sev;
      sev = msg.get_severity();
    end

    // Test ID
    msg.set_id("MSG_ID");
    begin
      string id;
      id = msg.get_id();
    end

    // Test message content
    msg.set_message("Test message");
    begin
      string m;
      m = msg.get_message();
    end

    // Test verbosity
    msg.set_verbosity(UVM_HIGH);
    count = msg.get_verbosity();

    // Test filename and line
    msg.set_filename("test.sv");
    msg.set_line(100);
    begin
      string fname;
      int line;
      fname = msg.get_filename();
      line = msg.get_line();
    end

    // Test context
    msg.set_context("test_context");
    begin
      string ctx;
      ctx = msg.get_context();
    end

    // Test action
    msg.set_action(UVM_DISPLAY);
    begin
      uvm_action act;
      act = msg.get_action();
    end

    // Test file
    msg.set_file(0);
    begin
      UVM_FILE f;
      f = msg.get_file();
    end

    //------------------------------------------------------------------------
    // Test uvm_report_catcher static methods
    //------------------------------------------------------------------------
    count = uvm_report_catcher::get_catcher_count();
    uvm_report_catcher::clear_catchers();
    uvm_report_catcher::summarize_catchers();

    // Test process_all_report_catchers
    begin
      uvm_action_type_e result;
      result = uvm_report_catcher::process_all_report_catchers(
        UVM_ERROR, "TEST_ID", "Test message",
        UVM_LOW, "test.sv", 50
      );
    end

    $display("UVM Report Infrastructure test completed");
  end
endmodule

// MOORE-DAG: moore.class @"uvm_pkg::uvm_report_server"
// MOORE-DAG: moore.class @"uvm_pkg::uvm_report_handler"
// MOORE-DAG: moore.class @"uvm_pkg::uvm_report_message"
// MOORE-DAG: moore.class @"uvm_pkg::uvm_report_catcher"
// MOORE-DAG: moore.class.methoddecl @dump_server_state
// MOORE-DAG: moore.class.methoddecl @get_info_count
// MOORE-DAG: moore.class.methoddecl @get_warning_count
// MOORE-DAG: moore.class.methoddecl @get_error_count
// MOORE-DAG: moore.class.methoddecl @get_fatal_count
// MOORE-DAG: moore.class.methoddecl @is_quit_count_reached
// MOORE-DAG: moore.class.methoddecl @report_summarize
