// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
// REQUIRES: slang
// XFAIL: *

// Test UVM objection mechanism support
`include "uvm_macros.svh"
import uvm_pkg::*;

// Test basic objection raise/drop pattern
class basic_objection_test extends uvm_test;
  `uvm_component_utils(basic_objection_test)

  function new(string name = "basic_objection_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this, "Starting test");
    #100ns;
    phase.drop_objection(this, "Test complete");
  endtask
endclass

// Test objection counting
class objection_count_test extends uvm_test;
  `uvm_component_utils(objection_count_test)

  function new(string name = "objection_count_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection objection;
    int count;

    phase.raise_objection(this, "First", 1);
    phase.raise_objection(this, "Second", 1);

    // Get the objection and check count
    objection = phase.get_objection();
    if (objection != null) begin
      count = objection.get_objection_count(this);
      `uvm_info("OBJECTION", $sformatf("Objection count: %0d", count), UVM_LOW)
    end

    #50ns;
    phase.drop_objection(this, "First done", 1);
    #50ns;
    phase.drop_objection(this, "Second done", 1);
  endtask
endclass

// Test uvm_objection class directly
class direct_objection_test extends uvm_test;
  `uvm_component_utils(direct_objection_test)

  uvm_objection my_objection;

  function new(string name = "direct_objection_test", uvm_component parent = null);
    super.new(name, parent);
    my_objection = new("my_objection");
  endfunction

  task run_phase(uvm_phase phase);
    int count;

    phase.raise_objection(this);

    // Use direct objection
    my_objection.raise_objection(this, "Direct raise", 2);
    count = my_objection.get_objection_count();
    `uvm_info("DIRECT_OBJ", $sformatf("Direct objection count: %0d", count), UVM_LOW)

    my_objection.drop_objection(this, "Direct drop", 1);
    count = my_objection.get_objection_count();
    `uvm_info("DIRECT_OBJ", $sformatf("After drop count: %0d", count), UVM_LOW)

    my_objection.drop_objection(this, "Final drop", 1);
    count = my_objection.get_objection_count();
    `uvm_info("DIRECT_OBJ", $sformatf("Final count: %0d", count), UVM_LOW)

    #100ns;
    phase.drop_objection(this);
  endtask
endclass

// Test uvm_test_done global objection
class test_done_objection_test extends uvm_test;
  `uvm_component_utils(test_done_objection_test)

  function new(string name = "test_done_objection_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    int count;

    phase.raise_objection(this);

    // Use global test_done objection
    uvm_test_done.raise_objection(this, "Test active");
    count = uvm_test_done.get_objection_count();
    `uvm_info("TEST_DONE", $sformatf("Test done objection count: %0d", count), UVM_LOW)

    #100ns;
    uvm_test_done.drop_objection(this, "Test complete");
    count = uvm_test_done.get_objection_count();
    `uvm_info("TEST_DONE", $sformatf("Test done after drop: %0d", count), UVM_LOW)

    phase.drop_objection(this);
  endtask
endclass

// Test multiple objections with count parameter
class multi_count_objection_test extends uvm_test;
  `uvm_component_utils(multi_count_objection_test)

  function new(string name = "multi_count_objection_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection obj;
    int count;

    // Raise with count > 1
    phase.raise_objection(this, "Multiple", 5);

    obj = phase.get_objection();
    if (obj != null) begin
      count = obj.get_objection_count();
      `uvm_info("MULTI", $sformatf("After raise(5): %0d", count), UVM_LOW)
    end

    // Drop with count > 1
    phase.drop_objection(this, "Drop some", 3);

    if (obj != null) begin
      count = obj.get_objection_count();
      `uvm_info("MULTI", $sformatf("After drop(3): %0d", count), UVM_LOW)
    end

    #100ns;
    phase.drop_objection(this, "Drop rest", 2);
  endtask
endclass

// Test get_objection_total and clear methods
class objection_total_test extends uvm_test;
  `uvm_component_utils(objection_total_test)

  function new(string name = "objection_total_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection obj;
    int count, total;

    phase.raise_objection(this);

    obj = new("total_test_objection");

    // Raise multiple times
    obj.raise_objection(this, "First", 1);
    obj.raise_objection(this, "Second", 2);
    obj.raise_objection(this, "Third", 3);

    count = obj.get_objection_count();
    total = obj.get_objection_total();
    `uvm_info("TOTAL", $sformatf("Count: %0d, Total: %0d", count, total), UVM_LOW)

    // Drop some
    obj.drop_objection(this, "Drop", 4);
    count = obj.get_objection_count();
    total = obj.get_objection_total();
    `uvm_info("TOTAL", $sformatf("After drop - Count: %0d, Total: %0d", count, total), UVM_LOW)

    // Clear all
    obj.clear();
    count = obj.get_objection_count();
    `uvm_info("TOTAL", $sformatf("After clear - Count: %0d", count), UVM_LOW)

    #100ns;
    phase.drop_objection(this);
  endtask
endclass

// Test raised() method
class objection_raised_test extends uvm_test;
  `uvm_component_utils(objection_raised_test)

  function new(string name = "objection_raised_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection obj;
    bit is_raised;

    phase.raise_objection(this);

    obj = new("raised_test_objection");

    // Check initially not raised
    is_raised = obj.raised();
    `uvm_info("RAISED", $sformatf("Initially raised: %0b", is_raised), UVM_LOW)

    // Raise and check
    obj.raise_objection(this);
    is_raised = obj.raised();
    `uvm_info("RAISED", $sformatf("After raise - raised: %0b", is_raised), UVM_LOW)

    // Drop and check
    obj.drop_objection(this);
    is_raised = obj.raised();
    `uvm_info("RAISED", $sformatf("After drop - raised: %0b", is_raised), UVM_LOW)

    #100ns;
    phase.drop_objection(this);
  endtask
endclass

// Test drain time methods
class drain_time_test extends uvm_test;
  `uvm_component_utils(drain_time_test)

  function new(string name = "drain_time_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection obj;
    time drain;

    phase.raise_objection(this);

    obj = new("drain_test_objection");

    // Set drain time
    obj.set_drain_time(this, 50ns);
    drain = obj.get_drain_time();
    `uvm_info("DRAIN", $sformatf("Drain time: %0t", drain), UVM_LOW)

    #100ns;
    phase.drop_objection(this);
  endtask
endclass

// Test phase get_objection_count method
class phase_objection_count_test extends uvm_test;
  `uvm_component_utils(phase_objection_count_test)

  function new(string name = "phase_objection_count_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    int count;

    // Initially no objections
    count = phase.get_objection_count();
    `uvm_info("PHASE_COUNT", $sformatf("Initial phase objection count: %0d", count), UVM_LOW)

    // Raise and check via phase
    phase.raise_objection(this, "Test", 3);
    count = phase.get_objection_count();
    `uvm_info("PHASE_COUNT", $sformatf("After raise(3): %0d", count), UVM_LOW)

    // Drop and check
    phase.drop_objection(this, "Done", 2);
    count = phase.get_objection_count();
    `uvm_info("PHASE_COUNT", $sformatf("After drop(2): %0d", count), UVM_LOW)

    #100ns;
    phase.drop_objection(this, "Final", 1);
  endtask
endclass

// Test display_objections method
class display_objections_test extends uvm_test;
  `uvm_component_utils(display_objections_test)

  function new(string name = "display_objections_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    uvm_objection obj;

    phase.raise_objection(this);

    obj = new("display_test_objection");
    obj.raise_objection(this, "First", 2);
    obj.raise_objection(this, "Second", 3);

    // Display objection state
    obj.display_objections();

    obj.drop_objection(this, "Drop", 5);
    obj.display_objections(null, 0);

    #100ns;
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial begin
    run_test();
  end
endmodule
