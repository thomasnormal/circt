//===----------------------------------------------------------------------===//
// UVM Factory Patterns Test
// Comprehensive tests for UVM factory patterns including:
// 1. type_id::create() pattern
// 2. Factory overrides (set_type_override_by_type, set_inst_override_by_type)
// 3. Factory registration macros equivalents
// 4. get_type() and get_type_name()
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package uvm_factory_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // SECTION 1: Basic Factory Registration with Macros
  //==========================================================================

  // Base object class with `uvm_object_utils macro
  class base_object extends uvm_object;
    `uvm_object_utils(base_object)

    rand bit [31:0] value;

    function new(string name = "base_object");
      super.new(name);
    endfunction

    // Explicit get_type_name for verification
    virtual function string get_type_name();
      return "base_object";
    endfunction

    virtual function string convert2string();
      return $sformatf("base_object: value=0x%0h", value);
    endfunction
  endclass

  // Extended object class for type override testing
  class extended_object extends base_object;
    `uvm_object_utils(extended_object)

    rand bit [15:0] extra_field;

    function new(string name = "extended_object");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "extended_object";
    endfunction

    virtual function string convert2string();
      return $sformatf("extended_object: value=0x%0h, extra_field=0x%0h", value, extra_field);
    endfunction
  endclass

  // Alternative extended object for testing multiple overrides
  class alt_extended_object extends base_object;
    `uvm_object_utils(alt_extended_object)

    rand bit [7:0] alt_field;

    function new(string name = "alt_extended_object");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "alt_extended_object";
    endfunction

    virtual function string convert2string();
      return $sformatf("alt_extended_object: value=0x%0h, alt_field=0x%0h", value, alt_field);
    endfunction
  endclass

  //==========================================================================
  // SECTION 2: Component Factory Registration
  //==========================================================================

  // Base component class with `uvm_component_utils macro
  class base_component extends uvm_component;
    `uvm_component_utils(base_component)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    // Explicit get_type_name for verification
    virtual function string get_type_name();
      return "base_component";
    endfunction

    virtual function string get_component_type();
      return "base_component";
    endfunction
  endclass

  // Extended component for type override
  class extended_component extends base_component;
    `uvm_component_utils(extended_component)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "extended_component";
    endfunction

    virtual function string get_component_type();
      return "extended_component";
    endfunction
  endclass

  // Instance-specific component for instance override
  class instance_component extends base_component;
    `uvm_component_utils(instance_component)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "instance_component";
    endfunction

    virtual function string get_component_type();
      return "instance_component";
    endfunction
  endclass

  //==========================================================================
  // SECTION 3: Sequence Item and Sequence Classes
  //==========================================================================

  class test_seq_item extends uvm_sequence_item;
    `uvm_object_utils(test_seq_item)

    rand bit [7:0] addr;
    rand bit [31:0] data;
    rand bit write;

    function new(string name = "test_seq_item");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "test_seq_item";
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=0x%0h data=0x%0h write=%0b", addr, data, write);
    endfunction
  endclass

  class ext_seq_item extends test_seq_item;
    `uvm_object_utils(ext_seq_item)

    rand bit [3:0] burst_len;

    function new(string name = "ext_seq_item");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "ext_seq_item";
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=0x%0h data=0x%0h write=%0b burst_len=%0d",
                       addr, data, write, burst_len);
    endfunction
  endclass

  //==========================================================================
  // TEST 1: type_id::create() Pattern Test
  //==========================================================================
  class test_type_id_create extends uvm_test;
    `uvm_component_utils(test_type_id_create)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      base_object obj1, obj2;
      base_component comp;
      test_seq_item item;

      phase.raise_objection(this, "Testing type_id::create()");

      // Test 1.1: Object creation via type_id::create()
      `uvm_info("TEST", "Testing object creation via type_id::create()", UVM_NONE)
      obj1 = base_object::type_id::create("obj1");
      if (obj1 == null)
        `uvm_error("TEST", "Failed to create obj1 via type_id::create()")
      else begin
        `uvm_info("TEST", $sformatf("Created obj1: %s", obj1.get_type_name()), UVM_MEDIUM)
        if (obj1.get_type_name() != "base_object")
          `uvm_error("TEST", $sformatf("obj1 type mismatch: expected base_object, got %s",
                                       obj1.get_type_name()))
        else
          `uvm_info("TEST", "PASS: obj1 created with correct type", UVM_NONE)
      end

      // Test 1.2: Multiple object creations
      obj2 = base_object::type_id::create("obj2");
      if (obj2 == null)
        `uvm_error("TEST", "Failed to create obj2")
      else begin
        // Verify they are different instances
        if (obj1 == obj2)
          `uvm_error("TEST", "obj1 and obj2 should be different instances")
        else
          `uvm_info("TEST", "PASS: Multiple objects created as separate instances", UVM_NONE)
      end

      // Test 1.3: Component creation via type_id::create()
      `uvm_info("TEST", "Testing component creation via type_id::create()", UVM_NONE)
      comp = base_component::type_id::create("comp", this);
      if (comp == null)
        `uvm_error("TEST", "Failed to create component via type_id::create()")
      else begin
        `uvm_info("TEST", $sformatf("Created comp: %s", comp.get_type_name()), UVM_MEDIUM)
        if (comp.get_type_name() != "base_component")
          `uvm_error("TEST", $sformatf("comp type mismatch: expected base_component, got %s",
                                       comp.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Component created with correct type", UVM_NONE)
      end

      // Test 1.4: Sequence item creation
      `uvm_info("TEST", "Testing sequence item creation via type_id::create()", UVM_NONE)
      item = test_seq_item::type_id::create("item");
      if (item == null)
        `uvm_error("TEST", "Failed to create sequence item")
      else begin
        if (item.get_type_name() != "test_seq_item")
          `uvm_error("TEST", $sformatf("item type mismatch: expected test_seq_item, got %s",
                                       item.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Sequence item created with correct type", UVM_NONE)
      end

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_type_id_create completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 2: Factory Overrides Test
  //==========================================================================
  class test_factory_overrides extends uvm_test;
    `uvm_component_utils(test_factory_overrides)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Set type override: all base_object should become extended_object
      `uvm_info("TEST", "Setting type override: base_object -> extended_object", UVM_NONE)
      f.set_type_override_by_type(base_object::get_type(),
                                  extended_object::get_type());
    endfunction

    virtual task run_phase(uvm_phase phase);
      base_object obj;

      phase.raise_objection(this, "Testing factory overrides");

      // Test 2.1: Object should be created as overridden type
      `uvm_info("TEST", "Testing type override for objects", UVM_NONE)
      obj = base_object::type_id::create("obj");
      if (obj == null)
        `uvm_error("TEST", "Failed to create object")
      else begin
        if (obj.get_type_name() != "extended_object")
          `uvm_error("TEST", $sformatf("Type override failed: expected extended_object, got %s",
                                       obj.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Type override applied correctly", UVM_NONE)
      end

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_factory_overrides completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 3: set_type_override_by_type Test
  //==========================================================================
  class env_for_type_override extends uvm_env;
    `uvm_component_utils(env_for_type_override)

    base_component comp1;
    base_component comp2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp1 = base_component::type_id::create("comp1", this);
      comp2 = base_component::type_id::create("comp2", this);
    endfunction
  endclass

  class test_set_type_override_by_type extends uvm_test;
    `uvm_component_utils(test_set_type_override_by_type)

    env_for_type_override env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Apply type override BEFORE creating environment
      `uvm_info("TEST", "Applying set_type_override_by_type", UVM_NONE)
      f.set_type_override_by_type(base_component::get_type(),
                                  extended_component::get_type());

      env = env_for_type_override::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);

      // Both components should be extended_component
      `uvm_info("TEST", "Checking type override results", UVM_NONE)

      if (env.comp1.get_component_type() != "extended_component")
        `uvm_error("TEST", $sformatf("comp1 override failed: expected extended_component, got %s",
                                     env.comp1.get_component_type()))
      else
        `uvm_info("TEST", "PASS: comp1 correctly overridden to extended_component", UVM_NONE)

      if (env.comp2.get_component_type() != "extended_component")
        `uvm_error("TEST", $sformatf("comp2 override failed: expected extended_component, got %s",
                                     env.comp2.get_component_type()))
      else
        `uvm_info("TEST", "PASS: comp2 correctly overridden to extended_component", UVM_NONE)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_set_type_override_by_type completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 4: set_inst_override_by_type Test
  //==========================================================================
  class env_for_inst_override extends uvm_env;
    `uvm_component_utils(env_for_inst_override)

    base_component comp1;
    base_component comp2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp1 = base_component::type_id::create("comp1", this);
      comp2 = base_component::type_id::create("comp2", this);
    endfunction
  endclass

  class test_set_inst_override_by_type extends uvm_test;
    `uvm_component_utils(test_set_inst_override_by_type)

    env_for_inst_override env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Apply instance override ONLY for comp1
      `uvm_info("TEST", "Applying set_inst_override_by_type for comp1 only", UVM_NONE)
      f.set_inst_override_by_type(base_component::get_type(),
                                  instance_component::get_type(),
                                  "uvm_test_top.env.comp1");

      env = env_for_inst_override::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);

      // comp1 should be instance_component, comp2 should remain base_component
      `uvm_info("TEST", "Checking instance override results", UVM_NONE)

      if (env.comp1.get_component_type() != "instance_component")
        `uvm_error("TEST", $sformatf("comp1 override failed: expected instance_component, got %s",
                                     env.comp1.get_component_type()))
      else
        `uvm_info("TEST", "PASS: comp1 correctly overridden to instance_component", UVM_NONE)

      if (env.comp2.get_component_type() != "base_component")
        `uvm_error("TEST", $sformatf("comp2 should be base_component, got %s",
                                     env.comp2.get_component_type()))
      else
        `uvm_info("TEST", "PASS: comp2 correctly remains base_component", UVM_NONE)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_set_inst_override_by_type completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 5: get_type() and get_type_name() Tests
  //==========================================================================
  class test_get_type_and_name extends uvm_test;
    `uvm_component_utils(test_get_type_and_name)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_object_wrapper obj_type;
      uvm_object_wrapper comp_type;
      uvm_object_wrapper item_type;
      base_object obj;
      base_component comp;
      test_seq_item item;

      phase.raise_objection(this, "Testing get_type() and get_type_name()");

      // Test 5.1: get_type() returns valid wrapper
      `uvm_info("TEST", "Testing get_type() for objects", UVM_NONE)
      obj_type = base_object::get_type();
      if (obj_type == null)
        `uvm_error("TEST", "base_object::get_type() returned null")
      else begin
        `uvm_info("TEST", $sformatf("base_object::get_type() returned: %s",
                                    obj_type.get_type_name()), UVM_MEDIUM)
        if (obj_type.get_type_name() != "base_object")
          `uvm_error("TEST", $sformatf("get_type_name() mismatch: expected base_object, got %s",
                                       obj_type.get_type_name()))
        else
          `uvm_info("TEST", "PASS: get_type() returns correct wrapper", UVM_NONE)
      end

      // Test 5.2: get_type() for components
      `uvm_info("TEST", "Testing get_type() for components", UVM_NONE)
      comp_type = base_component::get_type();
      if (comp_type == null)
        `uvm_error("TEST", "base_component::get_type() returned null")
      else begin
        if (comp_type.get_type_name() != "base_component")
          `uvm_error("TEST", $sformatf("Component get_type_name() mismatch: expected base_component, got %s",
                                       comp_type.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Component get_type() returns correct wrapper", UVM_NONE)
      end

      // Test 5.3: get_type() for sequence items
      `uvm_info("TEST", "Testing get_type() for sequence items", UVM_NONE)
      item_type = test_seq_item::get_type();
      if (item_type == null)
        `uvm_error("TEST", "test_seq_item::get_type() returned null")
      else begin
        if (item_type.get_type_name() != "test_seq_item")
          `uvm_error("TEST", $sformatf("Seq item get_type_name() mismatch: expected test_seq_item, got %s",
                                       item_type.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Sequence item get_type() returns correct wrapper", UVM_NONE)
      end

      // Test 5.4: get_type_name() on instances
      `uvm_info("TEST", "Testing get_type_name() on instances", UVM_NONE)
      obj = base_object::type_id::create("obj");
      comp = base_component::type_id::create("comp", this);
      item = test_seq_item::type_id::create("item");

      if (obj != null && obj.get_type_name() != "base_object")
        `uvm_error("TEST", "Instance get_type_name() failed for object")
      else
        `uvm_info("TEST", "PASS: Object instance get_type_name() correct", UVM_NONE)

      if (comp != null && comp.get_type_name() != "base_component")
        `uvm_error("TEST", "Instance get_type_name() failed for component")
      else
        `uvm_info("TEST", "PASS: Component instance get_type_name() correct", UVM_NONE)

      if (item != null && item.get_type_name() != "test_seq_item")
        `uvm_error("TEST", "Instance get_type_name() failed for sequence item")
      else
        `uvm_info("TEST", "PASS: Sequence item instance get_type_name() correct", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_get_type_and_name completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 6: Factory Registration Macros Equivalents
  //==========================================================================

  // Test class without macros - manual factory registration
  class manual_reg_object extends uvm_object;
    // Manual equivalent of `uvm_object_utils(manual_reg_object)
    typedef uvm_object_registry #(manual_reg_object, "manual_reg_object") type_id;

    static function type_id get_type();
      return type_id::get();
    endfunction

    virtual function uvm_object create(string name = "");
      manual_reg_object tmp = new(name);
      return tmp;
    endfunction

    function new(string name = "manual_reg_object");
      super.new(name);
    endfunction

    virtual function string get_type_name();
      return "manual_reg_object";
    endfunction
  endclass

  // Test component without macros - manual factory registration
  class manual_reg_component extends uvm_component;
    // Manual equivalent of `uvm_component_utils(manual_reg_component)
    typedef uvm_component_registry #(manual_reg_component, "manual_reg_component") type_id;

    static function type_id get_type();
      return type_id::get();
    endfunction

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function string get_type_name();
      return "manual_reg_component";
    endfunction
  endclass

  class test_factory_registration_manual extends uvm_test;
    `uvm_component_utils(test_factory_registration_manual)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      manual_reg_object obj;
      manual_reg_component comp;
      uvm_object_wrapper wrapper;

      phase.raise_objection(this, "Testing manual factory registration");

      // Test 6.1: Manual object registration works with type_id::create()
      `uvm_info("TEST", "Testing manual object registration", UVM_NONE)
      obj = manual_reg_object::type_id::create("obj");
      if (obj == null)
        `uvm_error("TEST", "Failed to create manually registered object")
      else begin
        if (obj.get_type_name() != "manual_reg_object")
          `uvm_error("TEST", $sformatf("Manual obj type mismatch: expected manual_reg_object, got %s",
                                       obj.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Manual object registration works", UVM_NONE)
      end

      // Test 6.2: Manual component registration works with type_id::create()
      `uvm_info("TEST", "Testing manual component registration", UVM_NONE)
      comp = manual_reg_component::type_id::create("comp", this);
      if (comp == null)
        `uvm_error("TEST", "Failed to create manually registered component")
      else begin
        if (comp.get_type_name() != "manual_reg_component")
          `uvm_error("TEST", $sformatf("Manual comp type mismatch: expected manual_reg_component, got %s",
                                       comp.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Manual component registration works", UVM_NONE)
      end

      // Test 6.3: get_type() works with manual registration
      `uvm_info("TEST", "Testing get_type() with manual registration", UVM_NONE)
      wrapper = manual_reg_object::get_type();
      if (wrapper == null)
        `uvm_error("TEST", "manual_reg_object::get_type() returned null")
      else begin
        if (wrapper.get_type_name() != "manual_reg_object")
          `uvm_error("TEST", "Manual registration get_type() failed")
        else
          `uvm_info("TEST", "PASS: get_type() works with manual registration", UVM_NONE)
      end

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_factory_registration_manual completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 7: Override Priority (Instance > Type)
  //==========================================================================
  class env_for_override_priority extends uvm_env;
    `uvm_component_utils(env_for_override_priority)

    base_component comp1;
    base_component comp2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp1 = base_component::type_id::create("comp1", this);
      comp2 = base_component::type_id::create("comp2", this);
    endfunction
  endclass

  class test_override_priority extends uvm_test;
    `uvm_component_utils(test_override_priority)

    env_for_override_priority env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // First: Set type override for ALL base_component
      `uvm_info("TEST", "Setting type override: base_component -> extended_component", UVM_NONE)
      f.set_type_override_by_type(base_component::get_type(),
                                  extended_component::get_type());

      // Then: Set instance override for comp1 (should have higher priority)
      `uvm_info("TEST", "Setting instance override: comp1 -> instance_component", UVM_NONE)
      f.set_inst_override_by_type(base_component::get_type(),
                                  instance_component::get_type(),
                                  "uvm_test_top.env.comp1");

      env = env_for_override_priority::type_id::create("env", this);
    endfunction

    virtual function void check_phase(uvm_phase phase);
      super.check_phase(phase);

      // comp1 should use instance override (instance_component)
      // comp2 should use type override (extended_component)
      `uvm_info("TEST", "Checking override priority", UVM_NONE)

      if (env.comp1.get_component_type() != "instance_component")
        `uvm_error("TEST", $sformatf("comp1 should be instance_component (instance override), got %s",
                                     env.comp1.get_component_type()))
      else
        `uvm_info("TEST", "PASS: Instance override takes priority over type override", UVM_NONE)

      if (env.comp2.get_component_type() != "extended_component")
        `uvm_error("TEST", $sformatf("comp2 should be extended_component (type override), got %s",
                                     env.comp2.get_component_type()))
      else
        `uvm_info("TEST", "PASS: Type override applied to comp2", UVM_NONE)
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_override_priority completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 8: Factory print() and Debug
  //==========================================================================
  class test_factory_print extends uvm_test;
    `uvm_component_utils(test_factory_print)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_factory f;
      base_object obj;
      extended_object ext_obj;
      base_component comp;

      phase.raise_objection(this, "Testing factory print");

      // Create some objects to ensure they're registered
      obj = base_object::type_id::create("obj");
      ext_obj = extended_object::type_id::create("ext_obj");
      comp = base_component::type_id::create("comp", this);

      // Get factory and print
      f = uvm_factory::get();
      `uvm_info("TEST", "Calling factory.print()", UVM_NONE)
      f.print();
      `uvm_info("TEST", "PASS: factory.print() executed", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_factory_print completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 9: Sequence Item Override Test
  //==========================================================================
  class test_seq_item_override extends uvm_test;
    `uvm_component_utils(test_seq_item_override)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      super.build_phase(phase);

      // Override test_seq_item with ext_seq_item
      `uvm_info("TEST", "Setting override: test_seq_item -> ext_seq_item", UVM_NONE)
      f.set_type_override_by_type(test_seq_item::get_type(),
                                  ext_seq_item::get_type());
    endfunction

    virtual task run_phase(uvm_phase phase);
      test_seq_item item;

      phase.raise_objection(this, "Testing sequence item override");

      item = test_seq_item::type_id::create("item");
      if (item == null)
        `uvm_error("TEST", "Failed to create sequence item")
      else begin
        if (item.get_type_name() != "ext_seq_item")
          `uvm_error("TEST", $sformatf("Seq item override failed: expected ext_seq_item, got %s",
                                       item.get_type_name()))
        else
          `uvm_info("TEST", "PASS: Sequence item correctly overridden to ext_seq_item", UVM_NONE)
      end

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_seq_item_override completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 10: Replace Parameter Test
  //==========================================================================
  class test_replace_param extends uvm_test;
    `uvm_component_utils(test_replace_param)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_factory f = uvm_factory::get();
      base_object obj;

      phase.raise_objection(this, "Testing replace parameter");

      // First override with replace=1 (default)
      `uvm_info("TEST", "Setting first override: base_object -> extended_object", UVM_NONE)
      f.set_type_override_by_name("base_object", "extended_object", 1);

      // Second override with replace=0 (should NOT replace)
      `uvm_info("TEST", "Setting second override (replace=0): base_object -> alt_extended_object", UVM_NONE)
      f.set_type_override_by_name("base_object", "alt_extended_object", 0);

      // Create object - should still be extended_object (first override)
      obj = base_object::type_id::create("obj");
      if (obj == null)
        `uvm_error("TEST", "Failed to create object")
      else begin
        if (obj.get_type_name() != "extended_object")
          `uvm_error("TEST", $sformatf("Replace param failed: expected extended_object, got %s",
                                       obj.get_type_name()))
        else
          `uvm_info("TEST", "PASS: replace=0 correctly preserved first override", UVM_NONE)
      end

      // Third override with replace=1 (should replace)
      `uvm_info("TEST", "Setting third override (replace=1): base_object -> alt_extended_object", UVM_NONE)
      f.set_type_override_by_name("base_object", "alt_extended_object", 1);

      obj = base_object::type_id::create("obj2");
      if (obj == null)
        `uvm_error("TEST", "Failed to create obj2")
      else begin
        if (obj.get_type_name() != "alt_extended_object")
          `uvm_error("TEST", $sformatf("Replace param failed: expected alt_extended_object, got %s",
                                       obj.get_type_name()))
        else
          `uvm_info("TEST", "PASS: replace=1 correctly replaced override", UVM_NONE)
      end

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_replace_param completed", UVM_NONE)
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import uvm_factory_test_pkg::*;

  initial begin
    `uvm_info("TB", "UVM Factory Test Starting", UVM_NONE)
    // Run the type_id::create test by default
    // Other tests can be selected via +UVM_TESTNAME
    run_test("test_type_id_create");
  end

endmodule
