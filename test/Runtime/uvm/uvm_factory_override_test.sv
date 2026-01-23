//===----------------------------------------------------------------------===//
// UVM Factory Override Test
// Tests for set_type_override_by_type(), set_inst_override_by_type(),
// set_type_override_by_name(), and set_inst_override_by_name() methods
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package factory_override_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Base Transaction Class
  //==========================================================================
  class base_tx extends uvm_sequence_item;
    `uvm_object_utils(base_tx)

    rand bit [7:0] data;

    function new(string name = "base_tx");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "base_tx";
    endfunction

    virtual function string convert2string();
      return $sformatf("base_tx: data=%0h", data);
    endfunction
  endclass

  //==========================================================================
  // Extended Transaction Class (for type override)
  //==========================================================================
  class extended_tx extends base_tx;
    `uvm_object_utils(extended_tx)

    rand bit [7:0] extra_data;

    function new(string name = "extended_tx");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "extended_tx";
    endfunction

    virtual function string convert2string();
      return $sformatf("extended_tx: data=%0h, extra_data=%0h", data, extra_data);
    endfunction
  endclass

  //==========================================================================
  // Base Driver Class
  //==========================================================================
  class base_driver extends uvm_driver #(base_tx);
    `uvm_component_utils(base_driver)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_driver_type();
      return "base_driver";
    endfunction

    virtual task run_phase(uvm_phase phase);
      `uvm_info("DRV", $sformatf("Running %s", get_driver_type()), UVM_MEDIUM)
    endtask
  endclass

  //==========================================================================
  // Extended Driver Class (for component type override)
  //==========================================================================
  class extended_driver extends base_driver;
    `uvm_component_utils(extended_driver)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_driver_type();
      return "extended_driver";
    endfunction
  endclass

  //==========================================================================
  // Instance-specific Driver Class (for instance override)
  //==========================================================================
  class inst_specific_driver extends base_driver;
    `uvm_component_utils(inst_specific_driver)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_driver_type();
      return "inst_specific_driver";
    endfunction
  endclass

  //==========================================================================
  // Test Environment
  //==========================================================================
  class factory_test_env extends uvm_env;
    `uvm_component_utils(factory_test_env)

    base_driver drv1;
    base_driver drv2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Create drivers using factory - overrides should apply
      drv1 = base_driver::type_id::create("drv1", this);
      drv2 = base_driver::type_id::create("drv2", this);
    endfunction
  endclass

  //==========================================================================
  // Test: Type Override by Name
  //==========================================================================
  class test_type_override_by_name extends uvm_test;
    `uvm_component_utils(test_type_override_by_name)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set type override by name - all base_driver instances become extended_driver
      f.set_type_override_by_name("base_driver", "extended_driver");

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // Verify both drivers are extended_driver type
      if (env.drv1.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected extended_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly overridden to extended_driver", UVM_MEDIUM)

      if (env.drv2.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv2 type mismatch: expected extended_driver, got %s",
                                     env.drv2.get_driver_type()))
      else
        `uvm_info("TEST", "drv2 correctly overridden to extended_driver", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_type_override_by_name completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Type Override by Type
  //==========================================================================
  class test_type_override_by_type extends uvm_test;
    `uvm_component_utils(test_type_override_by_type)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set type override by type
      f.set_type_override_by_type(base_driver::get_type(),
                                  extended_driver::get_type());

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // Verify both drivers are extended_driver type
      if (env.drv1.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected extended_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly overridden to extended_driver via by_type", UVM_MEDIUM)

      if (env.drv2.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv2 type mismatch: expected extended_driver, got %s",
                                     env.drv2.get_driver_type()))
      else
        `uvm_info("TEST", "drv2 correctly overridden to extended_driver via by_type", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_type_override_by_type completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Instance Override by Name
  //==========================================================================
  class test_inst_override_by_name extends uvm_test;
    `uvm_component_utils(test_inst_override_by_name)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set instance override by name - only drv1 should be overridden
      f.set_inst_override_by_name("base_driver", "inst_specific_driver",
                                  "uvm_test_top.env.drv1");

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // drv1 should be inst_specific_driver
      if (env.drv1.get_driver_type() != "inst_specific_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected inst_specific_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly overridden to inst_specific_driver", UVM_MEDIUM)

      // drv2 should remain base_driver (no override)
      if (env.drv2.get_driver_type() != "base_driver")
        `uvm_error("TEST", $sformatf("drv2 type mismatch: expected base_driver, got %s",
                                     env.drv2.get_driver_type()))
      else
        `uvm_info("TEST", "drv2 correctly remains base_driver (no override)", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_inst_override_by_name completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Instance Override by Type
  //==========================================================================
  class test_inst_override_by_type extends uvm_test;
    `uvm_component_utils(test_inst_override_by_type)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set instance override by type - only drv2 should be overridden
      f.set_inst_override_by_type(base_driver::get_type(),
                                  inst_specific_driver::get_type(),
                                  "uvm_test_top.env.drv2");

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // drv1 should remain base_driver (no override)
      if (env.drv1.get_driver_type() != "base_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected base_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly remains base_driver (no override)", UVM_MEDIUM)

      // drv2 should be inst_specific_driver
      if (env.drv2.get_driver_type() != "inst_specific_driver")
        `uvm_error("TEST", $sformatf("drv2 type mismatch: expected inst_specific_driver, got %s",
                                     env.drv2.get_driver_type()))
      else
        `uvm_info("TEST", "drv2 correctly overridden to inst_specific_driver via by_type", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_inst_override_by_type completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Instance Override Priority over Type Override
  //==========================================================================
  class test_override_priority extends uvm_test;
    `uvm_component_utils(test_override_priority)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set type override for all base_driver
      f.set_type_override_by_name("base_driver", "extended_driver");

      // Set instance override for drv1 - should have higher priority
      f.set_inst_override_by_name("base_driver", "inst_specific_driver",
                                  "uvm_test_top.env.drv1");

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // drv1 should be inst_specific_driver (instance override takes priority)
      if (env.drv1.get_driver_type() != "inst_specific_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected inst_specific_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly uses instance override over type override", UVM_MEDIUM)

      // drv2 should be extended_driver (type override applies)
      if (env.drv2.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv2 type mismatch: expected extended_driver, got %s",
                                     env.drv2.get_driver_type()))
      else
        `uvm_info("TEST", "drv2 correctly uses type override", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_override_priority completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Object Type Override
  //==========================================================================
  class test_object_type_override extends uvm_test;
    `uvm_component_utils(test_object_type_override)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set type override for objects
      f.set_type_override_by_type(base_tx::get_type(),
                                  extended_tx::get_type());
    endfunction

    virtual task run_phase(uvm_phase phase);
      base_tx tx;
      phase.raise_objection(this, "Testing object override");

      // Create object using factory
      tx = base_tx::type_id::create("tx");

      if (tx.get_type_name() != "extended_tx")
        `uvm_error("TEST", $sformatf("tx type mismatch: expected extended_tx, got %s",
                                     tx.get_type_name()))
      else
        `uvm_info("TEST", "Object correctly overridden to extended_tx", UVM_MEDIUM)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_object_type_override completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // Test: Replace Parameter (replace=0 should not override existing)
  //==========================================================================
  class test_replace_parameter extends uvm_test;
    `uvm_component_utils(test_replace_parameter)

    factory_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // First override - should work
      f.set_type_override_by_name("base_driver", "extended_driver", 1);

      // Second override with replace=0 - should NOT override the first
      f.set_type_override_by_name("base_driver", "inst_specific_driver", 0);

      env = factory_test_env::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      // Both drivers should be extended_driver (first override preserved)
      if (env.drv1.get_driver_type() != "extended_driver")
        `uvm_error("TEST", $sformatf("drv1 type mismatch: expected extended_driver, got %s",
                                     env.drv1.get_driver_type()))
      else
        `uvm_info("TEST", "drv1 correctly uses first override (replace=0 respected)", UVM_MEDIUM)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_replace_parameter completed", UVM_NONE)
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import factory_override_test_pkg::*;

  initial begin
    `uvm_info("TB", "Factory Override Test Starting", UVM_NONE)
    // Run any of the tests defined above
    // Default: test_type_override_by_name
    run_test("test_type_override_by_name");
  end

endmodule
