// RUN: circt-verilog --parse-only --no-uvm-auto-include -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv %s
// REQUIRES: slang

// Test UVM utility classes: cmdline_processor, report_server, report_catcher

`timescale 1ns/1ps

`include "uvm_macros.svh"
import uvm_pkg::*;

// Test uvm_cmdline_processor usage
class cmdline_test extends uvm_test;
  `uvm_component_utils(cmdline_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    uvm_cmdline_processor clp;
    string test_name;
    string args[$];
    int found;

    super.build_phase(phase);

    // Get the command line processor singleton
    clp = uvm_cmdline_processor::get_inst();

    // Test various cmdline methods
    found = clp.get_arg_value("+UVM_TESTNAME=", test_name);
    `uvm_info("CMDLINE", $sformatf("Test name found=%0d, value=%s", found, test_name), UVM_LOW)

    clp.get_args(args);
    `uvm_info("CMDLINE", $sformatf("Number of args: %0d", args.size()), UVM_LOW)
  endfunction
endclass

// Test uvm_report_server usage
class report_server_test extends uvm_test;
  `uvm_component_utils(report_server_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    uvm_report_server srv;
    int error_count;
    int warning_count;

    super.build_phase(phase);

    // Get the report server singleton
    srv = uvm_report_server::get_server();

    // Test various report server methods
    srv.set_max_quit_count(20);
    `uvm_info("REPORT", $sformatf("Max quit count: %0d", srv.get_max_quit_count()), UVM_LOW)

    error_count = srv.get_severity_count(UVM_ERROR);
    warning_count = srv.get_severity_count(UVM_WARNING);
    `uvm_info("REPORT", $sformatf("Errors: %0d, Warnings: %0d", error_count, warning_count), UVM_LOW)
  endfunction

  virtual function void report_phase(uvm_phase phase);
    uvm_report_server srv;
    super.report_phase(phase);

    srv = uvm_report_server::get_server();
    `uvm_info("REPORT", $sformatf("Final error count: %0d", srv.get_severity_count(UVM_ERROR)), UVM_LOW)
    `uvm_info("REPORT", $sformatf("Final warning count: %0d", srv.get_severity_count(UVM_WARNING)), UVM_LOW)
  endfunction
endclass

// Test uvm_report_catcher usage - implements pure virtual catch() method
class my_error_catcher extends uvm_report_catcher;
  int caught_errors = 0;

  function new(string name = "my_error_catcher");
    super.new(name);
  endfunction

  // Implement the pure virtual catch() method required by uvm-core
  virtual function action_e catch();
    // Catch all errors and demote to warnings
    if (get_severity() == UVM_ERROR) begin
      caught_errors++;
      set_severity(UVM_WARNING);
      set_message({get_message(), " [DEMOTED FROM ERROR]"});
      `uvm_info("CATCHER", $sformatf("Caught error #%0d, demoting to warning", caught_errors), UVM_LOW)
    end
    return THROW;
  endfunction
endclass

class report_catcher_test extends uvm_test;
  `uvm_component_utils(report_catcher_test)

  my_error_catcher catcher;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    catcher = new("catcher");
    // Use uvm_report_cb::add to register the catcher (uvm-core API)
    uvm_report_cb::add(null, catcher);
  endfunction

  virtual task run_phase(uvm_phase phase);
    phase.raise_objection(this);

    // These would normally be errors but our catcher demotes them
    `uvm_info("TEST", "Starting catcher test", UVM_LOW)
    #10;
    `uvm_info("TEST", "Catcher test complete", UVM_LOW)

    phase.drop_objection(this);
  endtask

  virtual function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    uvm_report_cb::delete(null, catcher);
    `uvm_info("TEST", $sformatf("Catcher caught %0d errors", catcher.caught_errors), UVM_LOW)
  endfunction
endclass

// Top-level module
module uvm_utilities_test_top;
  import uvm_pkg::*;
  initial begin
    run_test("cmdline_test");
  end
endmodule
