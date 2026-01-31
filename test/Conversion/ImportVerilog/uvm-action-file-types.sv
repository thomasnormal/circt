// RUN: circt-verilog --parse-only --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s

// Test that uvm_action and UVM_FILE types are properly defined in the UVM stubs.
// These types are used throughout UVM reporting infrastructure for action flags
// and file handles.

`timescale 1ns/1ps
`include "uvm_macros.svh"
import uvm_pkg::*;

// Test class using uvm_action type for report action configuration
class action_test_component extends uvm_component;
  `uvm_component_utils(action_test_component)

  // Test using uvm_action type directly
  uvm_action my_action;

  // Test using UVM_FILE type for file handles
  UVM_FILE my_file_handle;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    // Initialize with action type constant
    my_action = UVM_DISPLAY | UVM_LOG;
    my_file_handle = 0;
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    // Test combining action flags (uvm_action is an int bitfield)
    my_action = UVM_NO_ACTION;
    my_action = my_action | UVM_DISPLAY;
    my_action = my_action | UVM_COUNT;
  endfunction

  virtual function void configure_report_actions();
    // Test set_report_severity_action which uses uvm_action
    set_report_severity_action(UVM_INFO, UVM_DISPLAY);
    set_report_severity_action(UVM_WARNING, UVM_DISPLAY | UVM_LOG);
    set_report_severity_action(UVM_ERROR, UVM_DISPLAY | UVM_COUNT);
    set_report_severity_action(UVM_FATAL, UVM_DISPLAY | UVM_EXIT);

    // Test set_report_id_action which uses uvm_action
    set_report_id_action("TEST_ID", UVM_DISPLAY);

    // Test set_report_severity_id_action which uses uvm_action
    set_report_severity_id_action(UVM_INFO, "SPECIAL_ID", UVM_LOG);
  endfunction

  virtual function void configure_report_files();
    // Test set_report_default_file which uses UVM_FILE
    set_report_default_file(my_file_handle);

    // Test set_report_severity_file which uses UVM_FILE
    set_report_severity_file(UVM_ERROR, my_file_handle);

    // Test set_report_id_file which uses UVM_FILE
    set_report_id_file("LOG_ID", my_file_handle);

    // Test set_report_severity_id_file which uses UVM_FILE
    set_report_severity_id_file(UVM_WARNING, "WARN_LOG", my_file_handle);
  endfunction
endclass

// Test using uvm_action in a custom report handler
class custom_report_handler;
  uvm_action default_action;
  UVM_FILE default_file;

  // Associative array with uvm_action values
  uvm_action severity_actions[uvm_severity];
  UVM_FILE severity_files[uvm_severity];

  function new();
    default_action = UVM_DISPLAY;
    default_file = 0;

    // Initialize severity action mapping
    severity_actions[UVM_INFO] = UVM_DISPLAY;
    severity_actions[UVM_WARNING] = UVM_DISPLAY | UVM_LOG;
    severity_actions[UVM_ERROR] = UVM_DISPLAY | UVM_COUNT;
    severity_actions[UVM_FATAL] = UVM_DISPLAY | UVM_EXIT;

    // Initialize severity file mapping
    severity_files[UVM_INFO] = 0;
    severity_files[UVM_WARNING] = 0;
    severity_files[UVM_ERROR] = 0;
    severity_files[UVM_FATAL] = 0;
  endfunction

  function uvm_action get_action(uvm_severity severity);
    if (severity_actions.exists(severity))
      return severity_actions[severity];
    return default_action;
  endfunction

  function void set_action(uvm_severity severity, uvm_action action);
    severity_actions[severity] = action;
  endfunction

  function UVM_FILE get_file(uvm_severity severity);
    if (severity_files.exists(severity))
      return severity_files[severity];
    return default_file;
  endfunction

  function void set_file(uvm_severity severity, UVM_FILE file);
    severity_files[severity] = file;
  endfunction
endclass

// Simple module to instantiate and test
module uvm_action_file_test;
  initial begin
    action_test_component comp;
    custom_report_handler handler;
    uvm_action test_action;
    UVM_FILE test_file;

    // Test direct type usage
    test_action = UVM_DISPLAY | UVM_LOG | UVM_COUNT;
    test_file = 0;

    // Verify action flag combinations work
    if ((test_action & UVM_DISPLAY) != 0) begin
      $display("UVM_DISPLAY flag is set");
    end

    if ((test_action & UVM_EXIT) == 0) begin
      $display("UVM_EXIT flag is not set");
    end

    handler = new();
    handler.set_action(UVM_ERROR, UVM_DISPLAY | UVM_STOP);
    test_action = handler.get_action(UVM_ERROR);

    $display("Test completed successfully");
  end
endmodule
