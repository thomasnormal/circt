//===----------------------------------------------------------------------===//
// UVM Reporting Test - Tests for enhanced UVM reporting functionality
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
// REQUIRES: slang
//
// This test verifies the UVM reporting infrastructure including:
// - uvm_report_handler
// - uvm_report_server (including dump_server_state, severity counts)
// - uvm_report_message
// - uvm_report_catcher
// - report_hook functionality
// - set_report_max_quit_count
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`include "uvm_macros.svh"

package reporting_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Custom Report Catcher - Example implementation
  //==========================================================================
  class error_demote_catcher extends uvm_report_catcher;
    string target_id;
    int catch_count = 0;

    function new(string name = "error_demote_catcher");
      super.new(name);
      target_id = "";
    endfunction

    // Set the ID to catch
    function void set_target_id(string id);
      target_id = id;
    endfunction

    // Override catch_action to demote specific errors to warnings
    virtual function uvm_action_type_e catch_action();
      if (get_id() == target_id && get_severity() == UVM_ERROR) begin
        // Demote error to warning by catching it
        catch_count++;
        set_severity(UVM_WARNING);
        return CAUGHT;  // Suppress the original message
      end
      return THROW;  // Let other messages through
    endfunction

  endclass

  //==========================================================================
  // Component with custom report_hook
  //==========================================================================
  class hooked_component extends uvm_component;
    `uvm_component_utils(hooked_component)

    int hook_call_count = 0;
    bit suppress_test_id = 0;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    // Override report_hook to customize report handling
    virtual function bit report_hook(
      string id,
      string message,
      int verbosity,
      string filename,
      int line
    );
      hook_call_count++;
      // Suppress messages with "SUPPRESS" id if enabled
      if (suppress_test_id && id == "SUPPRESS")
        return 0;  // Suppress
      return 1;  // Allow
    endfunction

  endclass

  //==========================================================================
  // Test for report handler
  //==========================================================================
  class report_handler_test extends uvm_test;
    `uvm_component_utils(report_handler_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_report_handler handler;

      phase.raise_objection(this, "Starting report_handler_test");

      `uvm_info("HANDLER_TEST", "Testing uvm_report_handler", UVM_LOW)

      // Get or create a report handler
      handler = get_report_handler();

      // Test verbosity level methods
      handler.set_verbosity_level(UVM_HIGH);
      if (handler.get_verbosity_level() != UVM_HIGH)
        `uvm_error("HANDLER_TEST", "get_verbosity_level mismatch")

      // Test action configuration
      handler.set_severity_action(UVM_WARNING, UVM_DISPLAY | UVM_COUNT);
      handler.set_id_action("SPECIAL", UVM_LOG);
      handler.set_severity_id_action(UVM_ERROR, "CRITICAL", UVM_DISPLAY | UVM_EXIT);

      // Test verbosity configuration for IDs
      handler.set_id_verbosity("VERBOSE_ID", UVM_DEBUG);
      if (handler.get_id_verbosity("VERBOSE_ID") != UVM_DEBUG)
        `uvm_error("HANDLER_TEST", "get_id_verbosity mismatch")

      // Test file handle configuration (just exercise the API)
      handler.set_default_file(0);
      handler.set_severity_file(UVM_ERROR, 0);
      handler.set_id_file("LOG_ID", 0);
      handler.set_severity_id_file(UVM_INFO, "LOG_ID", 0);

      // Dump handler state for debugging
      handler.dump_state();

      `uvm_info("HANDLER_TEST", "uvm_report_handler tests passed", UVM_LOW)

      phase.drop_objection(this, "Finished report_handler_test");
    endtask

  endclass

  //==========================================================================
  // Test for report server
  //==========================================================================
  class report_server_test extends uvm_test;
    `uvm_component_utils(report_server_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_report_server server;
      int count;

      phase.raise_objection(this, "Starting report_server_test");

      `uvm_info("SERVER_TEST", "Testing uvm_report_server", UVM_LOW)

      // Get the report server singleton
      server = uvm_report_server::get_server();

      // Test max quit count
      server.set_max_quit_count(20);
      if (server.get_max_quit_count() != 20)
        `uvm_error("SERVER_TEST", "set_max_quit_count failed")

      // Also test via component method
      set_report_max_quit_count(15);
      if (get_report_max_quit_count() != 15)
        `uvm_error("SERVER_TEST", "component max_quit_count methods failed")

      // Reset counts for clean testing
      server.reset_severity_counts();
      server.reset_quit_count();

      // Generate some messages to test counting
      `uvm_info("SERVER_TEST", "Test info message 1", UVM_LOW)
      `uvm_info("SERVER_TEST", "Test info message 2", UVM_LOW)
      `uvm_warning("SERVER_TEST", "Test warning message")

      // Check severity counts
      if (server.get_info_count() < 2)
        `uvm_error("SERVER_TEST", $sformatf("Expected at least 2 info, got %0d", server.get_info_count()))

      if (server.get_warning_count() < 1)
        `uvm_error("SERVER_TEST", $sformatf("Expected at least 1 warning, got %0d", server.get_warning_count()))

      // Test ID count
      count = server.get_id_count("SERVER_TEST");
      if (count < 3)
        `uvm_error("SERVER_TEST", $sformatf("Expected at least 3 id counts, got %0d", count))

      // Test is_quit_count_reached
      server.reset_quit_count();
      server.set_max_quit_count(2);
      if (server.is_quit_count_reached())
        `uvm_error("SERVER_TEST", "is_quit_count_reached should be false")

      server.incr_quit_count();
      server.incr_quit_count();
      if (!server.is_quit_count_reached())
        `uvm_error("SERVER_TEST", "is_quit_count_reached should be true")

      // Dump server state
      server.dump_server_state();

      // Test report summarize
      server.report_summarize();

      `uvm_info("SERVER_TEST", "uvm_report_server tests passed", UVM_LOW)

      phase.drop_objection(this, "Finished report_server_test");
    endtask

  endclass

  //==========================================================================
  // Test for report message
  //==========================================================================
  class report_message_test extends uvm_test;
    `uvm_component_utils(report_message_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_report_message msg;

      phase.raise_objection(this, "Starting report_message_test");

      `uvm_info("MSG_TEST", "Testing uvm_report_message", UVM_LOW)

      // Create using factory method
      msg = uvm_report_message::new_report_message("test_msg");

      // Test severity
      msg.set_severity(UVM_WARNING);
      if (msg.get_severity() != UVM_WARNING)
        `uvm_error("MSG_TEST", "severity mismatch")

      // Test ID
      msg.set_id("TEST_ID");
      if (msg.get_id() != "TEST_ID")
        `uvm_error("MSG_TEST", "id mismatch")

      // Test message
      msg.set_message("Test message content");
      if (msg.get_message() != "Test message content")
        `uvm_error("MSG_TEST", "message mismatch")

      // Test verbosity
      msg.set_verbosity(UVM_HIGH);
      if (msg.get_verbosity() != UVM_HIGH)
        `uvm_error("MSG_TEST", "verbosity mismatch")

      // Test filename and line
      msg.set_filename("test.sv");
      msg.set_line(100);
      if (msg.get_filename() != "test.sv")
        `uvm_error("MSG_TEST", "filename mismatch")
      if (msg.get_line() != 100)
        `uvm_error("MSG_TEST", "line mismatch")

      // Test context
      msg.set_context("test_context");
      if (msg.get_context() != "test_context")
        `uvm_error("MSG_TEST", "context mismatch")

      // Test action
      msg.set_action(UVM_DISPLAY | UVM_LOG);
      if (msg.get_action() != (UVM_DISPLAY | UVM_LOG))
        `uvm_error("MSG_TEST", "action mismatch")

      // Test file handle
      msg.set_file(0);
      if (msg.get_file() != 0)
        `uvm_error("MSG_TEST", "file mismatch")

      `uvm_info("MSG_TEST", "uvm_report_message tests passed", UVM_LOW)

      phase.drop_objection(this, "Finished report_message_test");
    endtask

  endclass

  //==========================================================================
  // Test for report catcher
  //==========================================================================
  class report_catcher_test extends uvm_test;
    `uvm_component_utils(report_catcher_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      error_demote_catcher catcher1;
      error_demote_catcher catcher2;
      int count;

      phase.raise_objection(this, "Starting report_catcher_test");

      `uvm_info("CATCHER_TEST", "Testing uvm_report_catcher", UVM_LOW)

      // Clear any existing catchers
      uvm_report_catcher::clear_catchers();

      // Verify initial state
      if (uvm_report_catcher::get_catcher_count() != 0)
        `uvm_error("CATCHER_TEST", "Expected 0 catchers initially")

      // Create and add catchers
      catcher1 = new("catcher1");
      catcher1.set_target_id("DEMOTE_ME");
      uvm_report_catcher::add(catcher1);

      catcher2 = new("catcher2");
      catcher2.set_target_id("ALSO_DEMOTE");
      uvm_report_catcher::add(catcher2);

      // Verify catcher count
      if (uvm_report_catcher::get_catcher_count() != 2)
        `uvm_error("CATCHER_TEST", "Expected 2 catchers")

      // Summarize catchers
      uvm_report_catcher::summarize_catchers();

      // Test catcher processing (this would normally be called internally)
      // Process a message that should be caught
      begin
        uvm_action_type_e result;
        result = uvm_report_catcher::process_all_report_catchers(
          UVM_ERROR, "DEMOTE_ME", "Test error to demote",
          UVM_LOW, "test.sv", 50
        );
        if (result != CAUGHT)
          `uvm_error("CATCHER_TEST", "Expected message to be caught")
        if (catcher1.catch_count != 1)
          `uvm_error("CATCHER_TEST", "Expected catcher1 to catch 1 message")
      end

      // Process a message that should NOT be caught
      begin
        uvm_action_type_e result;
        result = uvm_report_catcher::process_all_report_catchers(
          UVM_ERROR, "OTHER_ID", "Test error to pass through",
          UVM_LOW, "test.sv", 60
        );
        if (result != THROW)
          `uvm_error("CATCHER_TEST", "Expected message to be thrown")
      end

      // Test remove
      uvm_report_catcher::remove(catcher1);
      if (uvm_report_catcher::get_catcher_count() != 1)
        `uvm_error("CATCHER_TEST", "Expected 1 catcher after remove")

      // Test clear
      uvm_report_catcher::clear_catchers();
      if (uvm_report_catcher::get_catcher_count() != 0)
        `uvm_error("CATCHER_TEST", "Expected 0 catchers after clear")

      `uvm_info("CATCHER_TEST", "uvm_report_catcher tests passed", UVM_LOW)

      phase.drop_objection(this, "Finished report_catcher_test");
    endtask

  endclass

  //==========================================================================
  // Test for report_hook
  //==========================================================================
  class report_hook_test extends uvm_test;
    `uvm_component_utils(report_hook_test)

    hooked_component hooked_comp;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      hooked_comp = hooked_component::type_id::create("hooked_comp", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Starting report_hook_test");

      `uvm_info("HOOK_TEST", "Testing report_hook functionality", UVM_LOW)

      // Reset hook counter
      hooked_comp.hook_call_count = 0;

      // Generate some messages through the hooked component
      hooked_comp.uvm_report_info("NORMAL", "This message should pass", UVM_LOW);
      hooked_comp.uvm_report_info("NORMAL", "Another normal message", UVM_LOW);

      // Verify hook was called
      if (hooked_comp.hook_call_count < 2)
        `uvm_error("HOOK_TEST", $sformatf("Expected at least 2 hook calls, got %0d", hooked_comp.hook_call_count))

      // Enable suppression and test
      hooked_comp.suppress_test_id = 1;
      hooked_comp.hook_call_count = 0;

      hooked_comp.uvm_report_info("SUPPRESS", "This should be suppressed", UVM_LOW);
      hooked_comp.uvm_report_info("NORMAL", "This should pass", UVM_LOW);

      // Hook should be called for both, but one suppressed
      if (hooked_comp.hook_call_count < 2)
        `uvm_error("HOOK_TEST", "Hook should be called even for suppressed messages")

      `uvm_info("HOOK_TEST", "report_hook tests passed", UVM_LOW)

      phase.drop_objection(this, "Finished report_hook_test");
    endtask

  endclass

  //==========================================================================
  // Comprehensive reporting test
  //==========================================================================
  class comprehensive_reporting_test extends uvm_test;
    `uvm_component_utils(comprehensive_reporting_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_report_server server;
      uvm_report_handler handler;

      phase.raise_objection(this, "Starting comprehensive_reporting_test");

      `uvm_info("COMP_TEST", "=== Comprehensive UVM Reporting Test ===", UVM_NONE)

      // Get server and handler
      server = uvm_report_server::get_server();
      handler = get_report_handler();

      // Configure reporting
      set_report_verbosity_level(UVM_HIGH);
      set_report_max_quit_count(100);

      // Exercise action configuration
      set_report_severity_action(UVM_INFO, UVM_DISPLAY);
      set_report_id_action("VERBOSE", UVM_DISPLAY | UVM_LOG);
      set_report_severity_id_action(UVM_WARNING, "SPECIAL", UVM_DISPLAY);

      // Exercise verbosity configuration
      set_report_id_verbosity("DEBUG_ID", UVM_DEBUG);
      set_report_severity_id_verbosity(UVM_INFO, "TRACE", UVM_FULL);

      // Exercise file configuration
      set_report_default_file(0);
      set_report_severity_file(UVM_ERROR, 0);
      set_report_id_file("FILE_ID", 0);
      set_report_severity_id_file(UVM_WARNING, "FILE_ID", 0);

      // Generate various messages
      `uvm_info("COMP_TEST", "Info message at UVM_LOW", UVM_LOW)
      `uvm_info("COMP_TEST", "Info message at UVM_MEDIUM", UVM_MEDIUM)
      `uvm_info("COMP_TEST", "Info message at UVM_HIGH", UVM_HIGH)
      `uvm_warning("COMP_TEST", "Warning message")

      // Dump states
      handler.dump_state();
      server.dump_server_state();

      // Final summary
      server.report_summarize();

      `uvm_info("COMP_TEST", "=== Comprehensive test completed ===", UVM_NONE)

      phase.drop_objection(this, "Finished comprehensive_reporting_test");
    endtask

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import reporting_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting UVM Reporting Tests", UVM_NONE)
    // Run the comprehensive test by default
    run_test("comprehensive_reporting_test");
  end

endmodule
