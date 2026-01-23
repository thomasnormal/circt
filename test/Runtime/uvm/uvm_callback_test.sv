//===----------------------------------------------------------------------===//
// UVM Callback Test
// Tests for uvm_callback, uvm_callbacks class, add/delete methods, and
// callback iteration patterns
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package callback_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Forward declarations
  //==========================================================================
  typedef class my_driver;
  typedef class my_driver_callback;

  //==========================================================================
  // Custom Callback Base Class
  //==========================================================================
  class my_driver_callback extends uvm_callback;
    `uvm_object_utils(my_driver_callback)

    string cb_name;
    int call_count;

    function new(string name = "my_driver_callback");
      super.new(name);
      cb_name = name;
      call_count = 0;
    endfunction

    // Virtual callback methods to be overridden
    virtual function void pre_drive(my_driver drv, ref bit [7:0] data);
      // Default implementation - do nothing
    endfunction

    virtual function void post_drive(my_driver drv, bit [7:0] data);
      // Default implementation - do nothing
    endfunction

    virtual function bit check_data(my_driver drv, bit [7:0] data);
      // Default implementation - return success
      return 1;
    endfunction
  endclass

  //==========================================================================
  // Logging Callback - Logs all driver operations
  //==========================================================================
  class logging_callback extends my_driver_callback;
    `uvm_object_utils(logging_callback)

    bit [7:0] logged_data[$];

    function new(string name = "logging_callback");
      super.new(name);
    endfunction

    virtual function void pre_drive(my_driver drv, ref bit [7:0] data);
      call_count++;
      `uvm_info("LOG_CB", $sformatf("[%s] pre_drive called with data=0x%0h", cb_name, data), UVM_MEDIUM)
    endfunction

    virtual function void post_drive(my_driver drv, bit [7:0] data);
      call_count++;
      logged_data.push_back(data);
      `uvm_info("LOG_CB", $sformatf("[%s] post_drive called with data=0x%0h (total logged: %0d)",
                                    cb_name, data, logged_data.size()), UVM_MEDIUM)
    endfunction
  endclass

  //==========================================================================
  // Data Modifier Callback - Modifies data before driving
  //==========================================================================
  class data_modifier_callback extends my_driver_callback;
    `uvm_object_utils(data_modifier_callback)

    bit [7:0] xor_mask;

    function new(string name = "data_modifier_callback");
      super.new(name);
      xor_mask = 8'hFF;  // Default: invert all bits
    endfunction

    virtual function void pre_drive(my_driver drv, ref bit [7:0] data);
      bit [7:0] original_data = data;
      call_count++;
      data = data ^ xor_mask;
      `uvm_info("MOD_CB", $sformatf("[%s] pre_drive: modified data 0x%0h -> 0x%0h (xor mask=0x%0h)",
                                    cb_name, original_data, data, xor_mask), UVM_MEDIUM)
    endfunction
  endclass

  //==========================================================================
  // Validation Callback - Validates data meets criteria
  //==========================================================================
  class validation_callback extends my_driver_callback;
    `uvm_object_utils(validation_callback)

    bit [7:0] min_value;
    bit [7:0] max_value;
    int validation_failures;

    function new(string name = "validation_callback");
      super.new(name);
      min_value = 8'h00;
      max_value = 8'hFF;
      validation_failures = 0;
    endfunction

    virtual function bit check_data(my_driver drv, bit [7:0] data);
      call_count++;
      if (data < min_value || data > max_value) begin
        validation_failures++;
        `uvm_warning("VAL_CB", $sformatf("[%s] Validation failed: data=0x%0h not in range [0x%0h, 0x%0h]",
                                          cb_name, data, min_value, max_value))
        return 0;
      end
      `uvm_info("VAL_CB", $sformatf("[%s] Validation passed: data=0x%0h", cb_name, data), UVM_HIGH)
      return 1;
    endfunction
  endclass

  //==========================================================================
  // Counter Callback - Counts callback invocations
  //==========================================================================
  class counter_callback extends my_driver_callback;
    `uvm_object_utils(counter_callback)

    int pre_drive_count;
    int post_drive_count;
    int check_count;

    function new(string name = "counter_callback");
      super.new(name);
      pre_drive_count = 0;
      post_drive_count = 0;
      check_count = 0;
    endfunction

    virtual function void pre_drive(my_driver drv, ref bit [7:0] data);
      pre_drive_count++;
      call_count++;
    endfunction

    virtual function void post_drive(my_driver drv, bit [7:0] data);
      post_drive_count++;
      call_count++;
    endfunction

    virtual function bit check_data(my_driver drv, bit [7:0] data);
      check_count++;
      call_count++;
      return 1;
    endfunction

    function void report_counts();
      `uvm_info("CNT_CB", $sformatf("[%s] Callback counts - pre_drive: %0d, post_drive: %0d, check: %0d",
                                    cb_name, pre_drive_count, post_drive_count, check_count), UVM_LOW)
    endfunction
  endclass

  //==========================================================================
  // Custom Driver with Callback Support
  //==========================================================================
  class my_driver extends uvm_driver;
    `uvm_component_utils(my_driver)

    // Callback pool for this driver type
    my_driver_callback cb_queue[$];

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    // Manual callback iteration - pre_drive
    function void do_pre_drive(ref bit [7:0] data);
      foreach (cb_queue[i]) begin
        cb_queue[i].pre_drive(this, data);
      end
    endfunction

    // Manual callback iteration - post_drive
    function void do_post_drive(bit [7:0] data);
      foreach (cb_queue[i]) begin
        cb_queue[i].post_drive(this, data);
      end
    endfunction

    // Manual callback iteration - check_data (exit on failure)
    function bit do_check_data(bit [7:0] data);
      foreach (cb_queue[i]) begin
        if (!cb_queue[i].check_data(this, data)) begin
          return 0;  // Exit on first failure
        end
      end
      return 1;
    endfunction

    // Add callback to queue
    function void add_callback(my_driver_callback cb, uvm_apprepend ordering = UVM_APPEND);
      if (ordering == UVM_APPEND)
        cb_queue.push_back(cb);
      else
        cb_queue.push_front(cb);
      `uvm_info("DRV", $sformatf("Added callback '%s' (ordering=%s, total=%0d)",
                                 cb.cb_name, ordering.name(), cb_queue.size()), UVM_MEDIUM)
    endfunction

    // Delete callback from queue
    function bit delete_callback(my_driver_callback cb);
      foreach (cb_queue[i]) begin
        if (cb_queue[i] == cb) begin
          cb_queue.delete(i);
          `uvm_info("DRV", $sformatf("Deleted callback '%s' (remaining=%0d)",
                                     cb.cb_name, cb_queue.size()), UVM_MEDIUM)
          return 1;
        end
      end
      `uvm_warning("DRV", $sformatf("Callback '%s' not found for deletion", cb.cb_name))
      return 0;
    endfunction

    // Get number of callbacks
    function int get_callback_count();
      return cb_queue.size();
    endfunction

    // Simulated drive operation
    task drive_data(bit [7:0] data);
      `uvm_info("DRV", $sformatf("Starting drive operation with data=0x%0h", data), UVM_MEDIUM)

      // Pre-drive callbacks (can modify data)
      do_pre_drive(data);

      // Check data validity
      if (!do_check_data(data)) begin
        `uvm_warning("DRV", "Data check failed, skipping drive")
        return;
      end

      // Simulate driving
      `uvm_info("DRV", $sformatf("Driving data=0x%0h", data), UVM_HIGH)
      #10ns;

      // Post-drive callbacks
      do_post_drive(data);

      `uvm_info("DRV", "Drive operation complete", UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // Test Environment
  //==========================================================================
  class callback_env extends uvm_env;
    `uvm_component_utils(callback_env)

    my_driver drv;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      drv = my_driver::type_id::create("drv", this);
    endfunction
  endclass

  //==========================================================================
  // Test: Basic Callback Add/Delete
  //==========================================================================
  class test_basic_add_delete extends uvm_test;
    `uvm_component_utils(test_basic_add_delete)

    callback_env env;
    logging_callback log_cb;
    counter_callback cnt_cb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Testing callback add/delete");

      // Create callbacks
      log_cb = logging_callback::type_id::create("log_cb");
      cnt_cb = counter_callback::type_id::create("cnt_cb");

      // Test 1: Add callbacks
      `uvm_info("TEST", "=== Test 1: Adding callbacks ===", UVM_NONE)
      env.drv.add_callback(log_cb, UVM_APPEND);
      env.drv.add_callback(cnt_cb, UVM_APPEND);

      if (env.drv.get_callback_count() != 2)
        `uvm_error("TEST", $sformatf("Expected 2 callbacks, got %0d", env.drv.get_callback_count()))
      else
        `uvm_info("TEST", "Callback count correct after adding", UVM_LOW)

      // Test 2: Drive with callbacks
      `uvm_info("TEST", "=== Test 2: Driving with callbacks ===", UVM_NONE)
      env.drv.drive_data(8'hAB);

      // Verify callbacks were invoked
      if (cnt_cb.pre_drive_count != 1 || cnt_cb.post_drive_count != 1)
        `uvm_error("TEST", "Counter callback not invoked correctly")
      else
        `uvm_info("TEST", "Callbacks invoked correctly during drive", UVM_LOW)

      // Test 3: Delete callback
      `uvm_info("TEST", "=== Test 3: Deleting callback ===", UVM_NONE)
      if (!env.drv.delete_callback(log_cb))
        `uvm_error("TEST", "Failed to delete log_cb")

      if (env.drv.get_callback_count() != 1)
        `uvm_error("TEST", $sformatf("Expected 1 callback after delete, got %0d",
                                     env.drv.get_callback_count()))
      else
        `uvm_info("TEST", "Callback count correct after delete", UVM_LOW)

      // Test 4: Drive after delete
      `uvm_info("TEST", "=== Test 4: Driving after callback deletion ===", UVM_NONE)
      env.drv.drive_data(8'hCD);

      cnt_cb.report_counts();

      if (cnt_cb.pre_drive_count != 2 || cnt_cb.post_drive_count != 2)
        `uvm_error("TEST", "Counter callback counts incorrect after second drive")
      else
        `uvm_info("TEST", "Remaining callback works correctly", UVM_LOW)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_basic_add_delete completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Callback Ordering (APPEND vs PREPEND)
  //==========================================================================
  class test_callback_ordering extends uvm_test;
    `uvm_component_utils(test_callback_ordering)

    callback_env env;
    data_modifier_callback mod_cb1;
    data_modifier_callback mod_cb2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      bit [7:0] test_data;

      phase.raise_objection(this, "Testing callback ordering");

      // Create modifier callbacks
      mod_cb1 = data_modifier_callback::type_id::create("mod_cb1");
      mod_cb1.xor_mask = 8'h0F;  // Modify lower nibble

      mod_cb2 = data_modifier_callback::type_id::create("mod_cb2");
      mod_cb2.xor_mask = 8'hF0;  // Modify upper nibble

      // Test 1: APPEND order (cb1 first, then cb2)
      `uvm_info("TEST", "=== Test 1: APPEND order (cb1, cb2) ===", UVM_NONE)
      env.drv.add_callback(mod_cb1, UVM_APPEND);
      env.drv.add_callback(mod_cb2, UVM_APPEND);

      test_data = 8'hAA;
      env.drv.do_pre_drive(test_data);
      // Expected: 0xAA ^ 0x0F = 0xA5, then 0xA5 ^ 0xF0 = 0x55
      if (test_data != 8'h55)
        `uvm_error("TEST", $sformatf("APPEND order: Expected 0x55, got 0x%0h", test_data))
      else
        `uvm_info("TEST", $sformatf("APPEND order correct: 0xAA -> 0x%0h", test_data), UVM_LOW)

      // Clear callbacks
      env.drv.cb_queue.delete();

      // Test 2: PREPEND order (cb2 first, then cb1)
      `uvm_info("TEST", "=== Test 2: PREPEND order (cb2 prepended before cb1) ===", UVM_NONE)
      env.drv.add_callback(mod_cb1, UVM_APPEND);
      env.drv.add_callback(mod_cb2, UVM_PREPEND);  // cb2 goes first

      test_data = 8'hAA;
      env.drv.do_pre_drive(test_data);
      // Expected: 0xAA ^ 0xF0 = 0x5A, then 0x5A ^ 0x0F = 0x55
      if (test_data != 8'h55)
        `uvm_error("TEST", $sformatf("PREPEND order: Expected 0x55, got 0x%0h", test_data))
      else
        `uvm_info("TEST", $sformatf("PREPEND order correct: 0xAA -> 0x%0h", test_data), UVM_LOW)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_callback_ordering completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Exit on Callback Failure
  //==========================================================================
  class test_exit_on_failure extends uvm_test;
    `uvm_component_utils(test_exit_on_failure)

    callback_env env;
    validation_callback val_cb1;
    validation_callback val_cb2;
    counter_callback cnt_cb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      bit result;

      phase.raise_objection(this, "Testing exit on callback failure");

      // Create validation callbacks
      val_cb1 = validation_callback::type_id::create("val_cb1");
      val_cb1.min_value = 8'h10;
      val_cb1.max_value = 8'hEF;

      val_cb2 = validation_callback::type_id::create("val_cb2");
      val_cb2.min_value = 8'h20;
      val_cb2.max_value = 8'hDF;

      cnt_cb = counter_callback::type_id::create("cnt_cb");

      env.drv.add_callback(val_cb1, UVM_APPEND);
      env.drv.add_callback(val_cb2, UVM_APPEND);
      env.drv.add_callback(cnt_cb, UVM_APPEND);

      // Test 1: Data passes all validators
      `uvm_info("TEST", "=== Test 1: Valid data (0x50) ===", UVM_NONE)
      result = env.drv.do_check_data(8'h50);
      if (!result)
        `uvm_error("TEST", "Expected validation to pass for 0x50")
      else
        `uvm_info("TEST", "Validation passed as expected", UVM_LOW)

      // Test 2: Data fails first validator
      `uvm_info("TEST", "=== Test 2: Data fails first validator (0x05) ===", UVM_NONE)
      result = env.drv.do_check_data(8'h05);
      if (result)
        `uvm_error("TEST", "Expected validation to fail for 0x05")
      else
        `uvm_info("TEST", "Validation failed as expected (exit on first failure)", UVM_LOW)

      // Verify second validator was not called (exit on failure)
      if (val_cb2.call_count > 1)
        `uvm_warning("TEST", "Second validator was called - exit on failure may not be working")

      // Test 3: Data passes first, fails second
      `uvm_info("TEST", "=== Test 3: Data fails second validator (0x15) ===", UVM_NONE)
      result = env.drv.do_check_data(8'h15);  // passes val_cb1, fails val_cb2
      if (result)
        `uvm_error("TEST", "Expected validation to fail for 0x15")
      else
        `uvm_info("TEST", "Validation correctly failed at second callback", UVM_LOW)

      `uvm_info("TEST", $sformatf("Validation failures: cb1=%0d, cb2=%0d",
                                  val_cb1.validation_failures, val_cb2.validation_failures), UVM_LOW)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_exit_on_failure completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: uvm_callbacks Static Class Usage
  //==========================================================================
  class test_uvm_callbacks_class extends uvm_test;
    `uvm_component_utils(test_uvm_callbacks_class)

    callback_env env;
    my_driver_callback cb1;
    my_driver_callback cb2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Testing uvm_callbacks class");

      // Create callbacks
      cb1 = my_driver_callback::type_id::create("cb1");
      cb2 = my_driver_callback::type_id::create("cb2");

      // Test uvm_callbacks::add static method
      `uvm_info("TEST", "=== Testing uvm_callbacks::add ===", UVM_NONE)
      uvm_callbacks#(my_driver, my_driver_callback)::add(env.drv, cb1, UVM_APPEND);
      `uvm_info("TEST", "Called uvm_callbacks::add with UVM_APPEND", UVM_LOW)

      uvm_callbacks#(my_driver, my_driver_callback)::add(env.drv, cb2, UVM_PREPEND);
      `uvm_info("TEST", "Called uvm_callbacks::add with UVM_PREPEND", UVM_LOW)

      // Test uvm_callbacks::delete static method
      `uvm_info("TEST", "=== Testing uvm_callbacks::delete ===", UVM_NONE)
      uvm_callbacks#(my_driver, my_driver_callback)::delete(env.drv, cb1);
      `uvm_info("TEST", "Called uvm_callbacks::delete", UVM_LOW)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_uvm_callbacks_class completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Callback Macros
  //==========================================================================
  class test_callback_macros extends uvm_test;
    `uvm_component_utils(test_callback_macros)

    callback_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Testing callback macros");

      // Test uvm_register_cb macro (empty in stub)
      `uvm_info("TEST", "=== Testing `uvm_register_cb macro ===", UVM_NONE)
      `uvm_register_cb(my_driver, my_driver_callback)
      `uvm_info("TEST", "uvm_register_cb macro executed (stub)", UVM_LOW)

      // Test uvm_do_callbacks macro (empty in stub)
      `uvm_info("TEST", "=== Testing `uvm_do_callbacks macro ===", UVM_NONE)
      `uvm_do_callbacks(my_driver, my_driver_callback, pre_drive(env.drv, _data))
      `uvm_info("TEST", "uvm_do_callbacks macro executed (stub)", UVM_LOW)

      // Test uvm_do_callbacks_exit_on macro (empty in stub)
      `uvm_info("TEST", "=== Testing `uvm_do_callbacks_exit_on macro ===", UVM_NONE)
      `uvm_do_callbacks_exit_on(my_driver, my_driver_callback, check_data(env.drv, 8'h00), 0)
      `uvm_info("TEST", "uvm_do_callbacks_exit_on macro executed (stub)", UVM_LOW)

      // Test uvm_do_obj_callbacks macro (empty in stub)
      `uvm_info("TEST", "=== Testing `uvm_do_obj_callbacks macro ===", UVM_NONE)
      `uvm_do_obj_callbacks(my_driver, my_driver_callback, env.drv, post_drive(env.drv, 8'h00))
      `uvm_info("TEST", "uvm_do_obj_callbacks macro executed (stub)", UVM_LOW)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_callback_macros completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Multiple Callback Types
  //==========================================================================
  class test_multiple_callback_types extends uvm_test;
    `uvm_component_utils(test_multiple_callback_types)

    callback_env env;
    logging_callback log_cb;
    data_modifier_callback mod_cb;
    validation_callback val_cb;
    counter_callback cnt_cb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Testing multiple callback types");

      // Create different callback types
      log_cb = logging_callback::type_id::create("log_cb");
      mod_cb = data_modifier_callback::type_id::create("mod_cb");
      mod_cb.xor_mask = 8'h55;
      val_cb = validation_callback::type_id::create("val_cb");
      val_cb.min_value = 8'h00;
      val_cb.max_value = 8'hFF;
      cnt_cb = counter_callback::type_id::create("cnt_cb");

      // Add callbacks in order: log -> mod -> val -> count
      env.drv.add_callback(log_cb, UVM_APPEND);
      env.drv.add_callback(mod_cb, UVM_APPEND);
      env.drv.add_callback(val_cb, UVM_APPEND);
      env.drv.add_callback(cnt_cb, UVM_APPEND);

      `uvm_info("TEST", $sformatf("Total callbacks: %0d", env.drv.get_callback_count()), UVM_LOW)

      // Drive multiple data values
      `uvm_info("TEST", "=== Driving with multiple callback types ===", UVM_NONE)
      env.drv.drive_data(8'hAA);
      env.drv.drive_data(8'h55);
      env.drv.drive_data(8'h00);

      // Report statistics
      `uvm_info("TEST", "=== Callback Statistics ===", UVM_NONE)
      `uvm_info("TEST", $sformatf("Logging callback: %0d entries logged", log_cb.logged_data.size()), UVM_LOW)
      `uvm_info("TEST", $sformatf("Modifier callback: %0d calls", mod_cb.call_count), UVM_LOW)
      `uvm_info("TEST", $sformatf("Validation callback: %0d calls, %0d failures",
                                  val_cb.call_count, val_cb.validation_failures), UVM_LOW)
      cnt_cb.report_counts();

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_multiple_callback_types completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Dynamic Callback Management
  //==========================================================================
  class test_dynamic_callback_mgmt extends uvm_test;
    `uvm_component_utils(test_dynamic_callback_mgmt)

    callback_env env;
    counter_callback cb_array[5];

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = callback_env::type_id::create("env", this);
    endfunction

    virtual task run_phase(uvm_phase phase);
      int i;

      phase.raise_objection(this, "Testing dynamic callback management");

      // Create array of callbacks
      for (i = 0; i < 5; i++) begin
        cb_array[i] = counter_callback::type_id::create($sformatf("cb_%0d", i));
      end

      // Test 1: Add all callbacks
      `uvm_info("TEST", "=== Test 1: Adding 5 callbacks ===", UVM_NONE)
      for (i = 0; i < 5; i++) begin
        env.drv.add_callback(cb_array[i], UVM_APPEND);
      end
      `uvm_info("TEST", $sformatf("Callback count: %0d", env.drv.get_callback_count()), UVM_LOW)

      // Drive once
      env.drv.drive_data(8'h42);

      // Verify all callbacks were called
      for (i = 0; i < 5; i++) begin
        if (cb_array[i].call_count == 0)
          `uvm_error("TEST", $sformatf("cb_%0d was not called", i))
      end
      `uvm_info("TEST", "All callbacks were invoked", UVM_LOW)

      // Test 2: Delete every other callback
      `uvm_info("TEST", "=== Test 2: Deleting callbacks 0, 2, 4 ===", UVM_NONE)
      for (i = 0; i < 5; i += 2) begin
        env.drv.delete_callback(cb_array[i]);
      end
      `uvm_info("TEST", $sformatf("Callback count after delete: %0d", env.drv.get_callback_count()), UVM_LOW)

      if (env.drv.get_callback_count() != 2)
        `uvm_error("TEST", "Expected 2 callbacks remaining")

      // Test 3: Re-add deleted callbacks at front
      `uvm_info("TEST", "=== Test 3: Re-adding callbacks with PREPEND ===", UVM_NONE)
      for (i = 0; i < 5; i += 2) begin
        env.drv.add_callback(cb_array[i], UVM_PREPEND);
      end
      `uvm_info("TEST", $sformatf("Callback count after re-add: %0d", env.drv.get_callback_count()), UVM_LOW)

      if (env.drv.get_callback_count() != 5)
        `uvm_error("TEST", "Expected 5 callbacks after re-add")

      // Drive again and check new order
      env.drv.drive_data(8'hBB);

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_dynamic_callback_mgmt completed", UVM_NONE)
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import callback_test_pkg::*;

  initial begin
    `uvm_info("TB", "UVM Callback Test Starting", UVM_NONE)
    // Run the basic add/delete test by default
    run_test("test_basic_add_delete");
  end

endmodule
