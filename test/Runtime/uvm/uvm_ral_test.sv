//===----------------------------------------------------------------------===//
// UVM Register Abstraction Layer (RAL) Test
// Comprehensive tests for UVM RAL patterns including:
// 1. uvm_reg_field
// 2. uvm_reg
// 3. uvm_reg_block
// 4. uvm_reg_map
// 5. Basic read/write methods
// 6. get/set methods
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package uvm_ral_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // SECTION 1: uvm_reg_field Tests
  //==========================================================================

  // Custom register field for testing
  class test_reg_field extends uvm_reg_field;
    `uvm_object_utils(test_reg_field)

    function new(string name = "test_reg_field");
      super.new(name);
    endfunction
  endclass

  //==========================================================================
  // SECTION 2: Custom Register Definitions
  //==========================================================================

  // Control register with multiple fields
  class ctrl_reg extends uvm_reg;
    `uvm_object_utils(ctrl_reg)

    rand uvm_reg_field enable;
    rand uvm_reg_field mode;
    rand uvm_reg_field status;
    rand uvm_reg_field reserved;

    function new(string name = "ctrl_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      // Create fields using new() - uvm_reg_field doesn't have factory registration
      enable = new("enable");
      mode = new("mode");
      status = new("status");
      reserved = new("reserved");

      // Configure fields: configure(parent, size, lsb_pos, access, volatile, reset, has_reset, is_rand, individually_accessible)
      enable.configure(this, 1, 0, "RW", 0, 1'b0, 1, 1, 0);
      mode.configure(this, 4, 1, "RW", 0, 4'h0, 1, 1, 0);
      status.configure(this, 3, 5, "RO", 1, 3'h0, 1, 0, 0);
      reserved.configure(this, 24, 8, "RO", 0, 24'h0, 1, 0, 0);
    endfunction
  endclass

  // Data register - simple 32-bit read/write
  class data_reg extends uvm_reg;
    `uvm_object_utils(data_reg)

    rand uvm_reg_field data;

    function new(string name = "data_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data = new("data");
      data.configure(this, 32, 0, "RW", 0, 32'h0, 1, 1, 0);
    endfunction
  endclass

  // Status register - read-only with clear-on-read fields
  class status_reg extends uvm_reg;
    `uvm_object_utils(status_reg)

    rand uvm_reg_field busy;
    rand uvm_reg_field error;
    rand uvm_reg_field done;
    rand uvm_reg_field irq_pending;

    function new(string name = "status_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      busy = new("busy");
      error = new("error");
      done = new("done");
      irq_pending = new("irq_pending");

      busy.configure(this, 1, 0, "RO", 1, 1'b0, 1, 0, 0);
      error.configure(this, 1, 1, "RO", 1, 1'b0, 1, 0, 0);
      done.configure(this, 1, 2, "RO", 1, 1'b0, 1, 0, 0);
      irq_pending.configure(this, 1, 3, "W1C", 1, 1'b0, 1, 0, 0);
    endfunction
  endclass

  // Interrupt enable register
  class int_en_reg extends uvm_reg;
    `uvm_object_utils(int_en_reg)

    rand uvm_reg_field global_en;
    rand uvm_reg_field int_mask;

    function new(string name = "int_en_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      global_en = new("global_en");
      int_mask = new("int_mask");

      global_en.configure(this, 1, 0, "RW", 0, 1'b0, 1, 1, 0);
      int_mask.configure(this, 8, 8, "RW", 0, 8'h0, 1, 1, 0);
    endfunction
  endclass

  //==========================================================================
  // SECTION 3: Register Block Definition
  //==========================================================================

  class test_reg_block extends uvm_reg_block;
    `uvm_object_utils(test_reg_block)

    rand ctrl_reg ctrl;
    rand data_reg data;
    rand status_reg status;
    rand int_en_reg int_en;

    uvm_reg_map default_map;

    function new(string name = "test_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      // Create registers
      ctrl = ctrl_reg::type_id::create("ctrl");
      ctrl.configure(this, null, "");
      ctrl.build();

      data = data_reg::type_id::create("data");
      data.configure(this, null, "");
      data.build();

      status = status_reg::type_id::create("status");
      status.configure(this, null, "");
      status.build();

      int_en = int_en_reg::type_id::create("int_en");
      int_en.configure(this, null, "");
      int_en.build();

      // Create address map
      default_map = create_map("default_map", 'h0, 4, UVM_LITTLE_ENDIAN);

      // Add registers to map with offsets
      default_map.add_reg(ctrl, 'h00, "RW");
      default_map.add_reg(data, 'h04, "RW");
      default_map.add_reg(status, 'h08, "RO");
      default_map.add_reg(int_en, 'h0C, "RW");

      // Lock the model
      lock_model();
    endfunction
  endclass

  //==========================================================================
  // SECTION 4: Hierarchical Register Block (nested blocks)
  //==========================================================================

  class child_reg_block extends uvm_reg_block;
    `uvm_object_utils(child_reg_block)

    rand data_reg data;
    uvm_reg_map default_map;

    function new(string name = "child_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data = data_reg::type_id::create("data");
      data.configure(this, null, "");
      data.build();

      default_map = create_map("default_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(data, 'h00, "RW");

      lock_model();
    endfunction
  endclass

  class parent_reg_block extends uvm_reg_block;
    `uvm_object_utils(parent_reg_block)

    rand ctrl_reg ctrl;
    rand child_reg_block child0;
    rand child_reg_block child1;

    uvm_reg_map default_map;

    function new(string name = "parent_reg_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      // Create parent register
      ctrl = ctrl_reg::type_id::create("ctrl");
      ctrl.configure(this, null, "");
      ctrl.build();

      // Create child blocks
      child0 = child_reg_block::type_id::create("child0");
      child0.configure(this, "");
      child0.build();

      child1 = child_reg_block::type_id::create("child1");
      child1.configure(this, "");
      child1.build();

      // Create address map
      default_map = create_map("default_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(ctrl, 'h00, "RW");
      default_map.add_submap(child0.default_map, 'h100);
      default_map.add_submap(child1.default_map, 'h200);

      lock_model();
    endfunction
  endclass

  //==========================================================================
  // TEST 1: Basic uvm_reg_field Operations
  //==========================================================================

  class test_reg_field_ops extends uvm_test;
    `uvm_component_utils(test_reg_field_ops)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_data_t value;

      phase.raise_objection(this, "Testing uvm_reg_field operations");

      // Test 1.1: Field get/set operations
      `uvm_info("TEST", "=== Test 1.1: Field get/set operations ===", UVM_NONE)

      // Set enable field
      reg_model.ctrl.enable.set(1);
      value = reg_model.ctrl.enable.get();
      if (value != 1)
        `uvm_error("TEST", $sformatf("Enable field set/get failed: expected 1, got %0d", value))
      else
        `uvm_info("TEST", "PASS: Enable field set/get works correctly", UVM_NONE)

      // Set mode field
      reg_model.ctrl.mode.set(4'hA);
      value = reg_model.ctrl.mode.get();
      if (value != 4'hA)
        `uvm_error("TEST", $sformatf("Mode field set/get failed: expected 0xA, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Mode field set/get works correctly", UVM_NONE)

      // Test 1.2: Field configuration queries
      `uvm_info("TEST", "=== Test 1.2: Field configuration queries ===", UVM_NONE)

      if (reg_model.ctrl.enable.get_n_bits() != 1)
        `uvm_error("TEST", "Enable field size mismatch")
      else
        `uvm_info("TEST", "PASS: Enable field size is 1 bit", UVM_NONE)

      if (reg_model.ctrl.mode.get_n_bits() != 4)
        `uvm_error("TEST", "Mode field size mismatch")
      else
        `uvm_info("TEST", "PASS: Mode field size is 4 bits", UVM_NONE)

      if (reg_model.ctrl.enable.get_lsb_pos() != 0)
        `uvm_error("TEST", "Enable field LSB position mismatch")
      else
        `uvm_info("TEST", "PASS: Enable field LSB position is 0", UVM_NONE)

      if (reg_model.ctrl.mode.get_lsb_pos() != 1)
        `uvm_error("TEST", "Mode field LSB position mismatch")
      else
        `uvm_info("TEST", "PASS: Mode field LSB position is 1", UVM_NONE)

      // Test 1.3: Field access type
      `uvm_info("TEST", "=== Test 1.3: Field access type ===", UVM_NONE)

      if (reg_model.ctrl.enable.get_access() != "RW")
        `uvm_error("TEST", $sformatf("Enable access mismatch: expected RW, got %s",
                                     reg_model.ctrl.enable.get_access()))
      else
        `uvm_info("TEST", "PASS: Enable field access is RW", UVM_NONE)

      if (reg_model.ctrl.status.get_access() != "RO")
        `uvm_error("TEST", $sformatf("Status access mismatch: expected RO, got %s",
                                     reg_model.ctrl.status.get_access()))
      else
        `uvm_info("TEST", "PASS: Status field access is RO", UVM_NONE)

      // Test 1.4: Field reset value
      `uvm_info("TEST", "=== Test 1.4: Field reset value ===", UVM_NONE)

      value = reg_model.ctrl.enable.get_reset();
      if (value != 0)
        `uvm_error("TEST", $sformatf("Enable reset value mismatch: expected 0, got %0d", value))
      else
        `uvm_info("TEST", "PASS: Enable field reset value is 0", UVM_NONE)

      // Test 1.5: Field parent access
      `uvm_info("TEST", "=== Test 1.5: Field parent access ===", UVM_NONE)

      if (reg_model.ctrl.enable.get_parent() != reg_model.ctrl)
        `uvm_error("TEST", "Enable field parent mismatch")
      else
        `uvm_info("TEST", "PASS: Enable field parent is ctrl register", UVM_NONE)

      // Test 1.6: Mirrored value operations
      `uvm_info("TEST", "=== Test 1.6: Mirrored value operations ===", UVM_NONE)

      reg_model.ctrl.enable.set_mirrored_value(1);
      value = reg_model.ctrl.enable.get_mirrored_value();
      if (value != 1)
        `uvm_error("TEST", $sformatf("Mirrored value mismatch: expected 1, got %0d", value))
      else
        `uvm_info("TEST", "PASS: Mirrored value set/get works correctly", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_field_ops completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 2: Basic uvm_reg Operations
  //==========================================================================

  class test_reg_ops extends uvm_test;
    `uvm_component_utils(test_reg_ops)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_data_t value;
      uvm_reg_field fields[$];

      phase.raise_objection(this, "Testing uvm_reg operations");

      // Test 2.1: Register get/set operations (aggregate across fields)
      `uvm_info("TEST", "=== Test 2.1: Register get/set operations ===", UVM_NONE)

      // Set entire register value
      reg_model.ctrl.set(32'h0000_001F);  // enable=1, mode=0xF, status=0
      value = reg_model.ctrl.get();
      // Note: status is RO, so only enable and mode should be settable
      if ((value & 32'h1F) != 32'h1F)
        `uvm_error("TEST", $sformatf("Ctrl register set/get failed: got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Ctrl register set/get works correctly", UVM_NONE)

      // Test 2.2: Register size
      `uvm_info("TEST", "=== Test 2.2: Register size ===", UVM_NONE)

      if (reg_model.ctrl.get_n_bits() != 32)
        `uvm_error("TEST", $sformatf("Ctrl register size mismatch: expected 32, got %0d",
                                     reg_model.ctrl.get_n_bits()))
      else
        `uvm_info("TEST", "PASS: Ctrl register size is 32 bits", UVM_NONE)

      // Test 2.3: Get fields from register
      `uvm_info("TEST", "=== Test 2.3: Get fields from register ===", UVM_NONE)

      reg_model.ctrl.get_fields(fields);
      if (fields.size() != 4)
        `uvm_error("TEST", $sformatf("Ctrl register field count mismatch: expected 4, got %0d",
                                     fields.size()))
      else
        `uvm_info("TEST", "PASS: Ctrl register has 4 fields", UVM_NONE)

      // Test 2.4: Get field by name
      `uvm_info("TEST", "=== Test 2.4: Get field by name ===", UVM_NONE)

      begin
        uvm_reg_field enable_field = reg_model.ctrl.get_field_by_name("enable");
        if (enable_field == null)
          `uvm_error("TEST", "Failed to get enable field by name")
        else if (enable_field != reg_model.ctrl.enable)
          `uvm_error("TEST", "get_field_by_name returned wrong field")
        else
          `uvm_info("TEST", "PASS: get_field_by_name works correctly", UVM_NONE)
      end

      // Test 2.5: Register parent/block access
      `uvm_info("TEST", "=== Test 2.5: Register parent/block access ===", UVM_NONE)

      if (reg_model.ctrl.get_parent() != reg_model)
        `uvm_error("TEST", "Ctrl register parent mismatch")
      else
        `uvm_info("TEST", "PASS: Ctrl register parent is correct", UVM_NONE)

      if (reg_model.ctrl.get_block() != reg_model)
        `uvm_error("TEST", "Ctrl register block mismatch")
      else
        `uvm_info("TEST", "PASS: Ctrl register block is correct", UVM_NONE)

      // Test 2.6: Register offset
      `uvm_info("TEST", "=== Test 2.6: Register offset ===", UVM_NONE)

      if (reg_model.ctrl.get_offset(reg_model.default_map) != 'h00)
        `uvm_error("TEST", "Ctrl register offset mismatch")
      else
        `uvm_info("TEST", "PASS: Ctrl register offset is 0x00", UVM_NONE)

      if (reg_model.data.get_offset(reg_model.default_map) != 'h04)
        `uvm_error("TEST", "Data register offset mismatch")
      else
        `uvm_info("TEST", "PASS: Data register offset is 0x04", UVM_NONE)

      // Test 2.7: Register reset value
      `uvm_info("TEST", "=== Test 2.7: Register reset value ===", UVM_NONE)

      value = reg_model.ctrl.get_reset();
      if (value != 0)
        `uvm_error("TEST", $sformatf("Ctrl register reset value mismatch: expected 0, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Ctrl register reset value is 0", UVM_NONE)

      // Test 2.8: Register mirrored value
      `uvm_info("TEST", "=== Test 2.8: Register mirrored value ===", UVM_NONE)

      value = reg_model.ctrl.get_mirrored_value();
      `uvm_info("TEST", $sformatf("Ctrl register mirrored value: 0x%0h", value), UVM_MEDIUM)
      `uvm_info("TEST", "PASS: get_mirrored_value works", UVM_NONE)

      // Test 2.9: Register access type
      `uvm_info("TEST", "=== Test 2.9: Register access type ===", UVM_NONE)

      `uvm_info("TEST", $sformatf("Ctrl register access: %s", reg_model.ctrl.get_access()), UVM_MEDIUM)
      `uvm_info("TEST", "PASS: get_access works", UVM_NONE)

      // Test 2.10: Check if register is in map
      `uvm_info("TEST", "=== Test 2.10: Register in map check ===", UVM_NONE)

      if (!reg_model.ctrl.is_in_map(reg_model.default_map))
        `uvm_error("TEST", "Ctrl register should be in default_map")
      else
        `uvm_info("TEST", "PASS: Ctrl register is in default_map", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_ops completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 3: Basic uvm_reg_block Operations
  //==========================================================================

  class test_reg_block_ops extends uvm_test;
    `uvm_component_utils(test_reg_block_ops)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg regs[$];
      uvm_reg_map maps[$];

      phase.raise_objection(this, "Testing uvm_reg_block operations");

      // Test 3.1: Get all registers from block
      `uvm_info("TEST", "=== Test 3.1: Get all registers from block ===", UVM_NONE)

      reg_model.get_registers(regs, UVM_REG_NO_HIER);
      if (regs.size() != 4)
        `uvm_error("TEST", $sformatf("Register count mismatch: expected 4, got %0d", regs.size()))
      else
        `uvm_info("TEST", "PASS: Block has 4 registers", UVM_NONE)

      // Test 3.2: Get register by name
      `uvm_info("TEST", "=== Test 3.2: Get register by name ===", UVM_NONE)

      begin
        uvm_reg found_reg = reg_model.get_reg_by_name("ctrl");
        if (found_reg == null)
          `uvm_error("TEST", "Failed to get ctrl register by name")
        else if (found_reg != reg_model.ctrl)
          `uvm_error("TEST", "get_reg_by_name returned wrong register")
        else
          `uvm_info("TEST", "PASS: get_reg_by_name works correctly", UVM_NONE)
      end

      // Test 3.3: Get maps from block
      `uvm_info("TEST", "=== Test 3.3: Get maps from block ===", UVM_NONE)

      reg_model.get_maps(maps);
      if (maps.size() != 1)
        `uvm_error("TEST", $sformatf("Map count mismatch: expected 1, got %0d", maps.size()))
      else
        `uvm_info("TEST", "PASS: Block has 1 map", UVM_NONE)

      // Test 3.4: Get default map
      `uvm_info("TEST", "=== Test 3.4: Get default map ===", UVM_NONE)

      if (reg_model.get_default_map() == null)
        `uvm_error("TEST", "Default map is null")
      else
        `uvm_info("TEST", "PASS: Default map exists", UVM_NONE)

      // Test 3.5: Get map by name
      `uvm_info("TEST", "=== Test 3.5: Get map by name ===", UVM_NONE)

      begin
        uvm_reg_map found_map = reg_model.get_map_by_name("default_map");
        if (found_map == null)
          `uvm_error("TEST", "Failed to get default_map by name")
        else
          `uvm_info("TEST", "PASS: get_map_by_name works correctly", UVM_NONE)
      end

      // Test 3.6: Block name
      `uvm_info("TEST", "=== Test 3.6: Block name ===", UVM_NONE)

      if (reg_model.get_name() != "reg_model")
        `uvm_error("TEST", $sformatf("Block name mismatch: expected reg_model, got %s",
                                     reg_model.get_name()))
      else
        `uvm_info("TEST", "PASS: Block name is correct", UVM_NONE)

      // Test 3.7: Block full name
      `uvm_info("TEST", "=== Test 3.7: Block full name ===", UVM_NONE)

      `uvm_info("TEST", $sformatf("Block full name: %s", reg_model.get_full_name()), UVM_MEDIUM)
      `uvm_info("TEST", "PASS: get_full_name works", UVM_NONE)

      // Test 3.8: Block is locked
      `uvm_info("TEST", "=== Test 3.8: Block is locked ===", UVM_NONE)

      if (!reg_model.is_locked())
        `uvm_error("TEST", "Block should be locked after build")
      else
        `uvm_info("TEST", "PASS: Block is locked", UVM_NONE)

      // Test 3.9: Reset block
      `uvm_info("TEST", "=== Test 3.9: Reset block ===", UVM_NONE)

      // First set some values
      reg_model.ctrl.set(32'h1234);
      reg_model.data.set(32'hABCD);

      // Reset the block
      reg_model.reset();

      // Check values are reset
      if (reg_model.ctrl.get() != 0 || reg_model.data.get() != 0)
        `uvm_error("TEST", "Block reset failed")
      else
        `uvm_info("TEST", "PASS: Block reset works correctly", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_block_ops completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 4: Basic uvm_reg_map Operations
  //==========================================================================

  class test_reg_map_ops extends uvm_test;
    `uvm_component_utils(test_reg_map_ops)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_map map;
      uvm_reg regs[$];
      uvm_reg_addr_t addr;

      phase.raise_objection(this, "Testing uvm_reg_map operations");

      map = reg_model.default_map;

      // Test 4.1: Map base address
      `uvm_info("TEST", "=== Test 4.1: Map base address ===", UVM_NONE)

      addr = map.get_base_addr();
      if (addr != 'h0)
        `uvm_error("TEST", $sformatf("Map base address mismatch: expected 0, got 0x%0h", addr))
      else
        `uvm_info("TEST", "PASS: Map base address is 0", UVM_NONE)

      // Test 4.2: Map n_bytes (bus width)
      `uvm_info("TEST", "=== Test 4.2: Map bus width ===", UVM_NONE)

      if (map.get_n_bytes() != 4)
        `uvm_error("TEST", $sformatf("Map n_bytes mismatch: expected 4, got %0d", map.get_n_bytes()))
      else
        `uvm_info("TEST", "PASS: Map bus width is 4 bytes", UVM_NONE)

      // Test 4.3: Map endianness
      `uvm_info("TEST", "=== Test 4.3: Map endianness ===", UVM_NONE)

      if (map.get_endian() != UVM_LITTLE_ENDIAN)
        `uvm_error("TEST", "Map endianness mismatch")
      else
        `uvm_info("TEST", "PASS: Map is little endian", UVM_NONE)

      // Test 4.4: Get register by offset
      `uvm_info("TEST", "=== Test 4.4: Get register by offset ===", UVM_NONE)

      begin
        uvm_reg found_reg = map.get_reg_by_offset('h00);
        if (found_reg == null)
          `uvm_error("TEST", "Failed to get register at offset 0x00")
        else if (found_reg != reg_model.ctrl)
          `uvm_error("TEST", "get_reg_by_offset returned wrong register")
        else
          `uvm_info("TEST", "PASS: get_reg_by_offset works correctly", UVM_NONE)
      end

      begin
        uvm_reg found_reg = map.get_reg_by_offset('h04);
        if (found_reg == null)
          `uvm_error("TEST", "Failed to get register at offset 0x04")
        else if (found_reg != reg_model.data)
          `uvm_error("TEST", "get_reg_by_offset returned wrong register for 0x04")
        else
          `uvm_info("TEST", "PASS: get_reg_by_offset works for data register", UVM_NONE)
      end

      // Test 4.5: Get all registers from map
      `uvm_info("TEST", "=== Test 4.5: Get all registers from map ===", UVM_NONE)

      map.get_registers(regs, UVM_REG_NO_HIER);
      if (regs.size() != 4)
        `uvm_error("TEST", $sformatf("Map register count mismatch: expected 4, got %0d", regs.size()))
      else
        `uvm_info("TEST", "PASS: Map has 4 registers", UVM_NONE)

      // Test 4.6: Map parent block
      `uvm_info("TEST", "=== Test 4.6: Map parent block ===", UVM_NONE)

      if (map.get_parent() != reg_model)
        `uvm_error("TEST", "Map parent mismatch")
      else
        `uvm_info("TEST", "PASS: Map parent is correct", UVM_NONE)

      // Test 4.7: Map name
      `uvm_info("TEST", "=== Test 4.7: Map name ===", UVM_NONE)

      if (map.get_name() != "default_map")
        `uvm_error("TEST", $sformatf("Map name mismatch: expected default_map, got %s", map.get_name()))
      else
        `uvm_info("TEST", "PASS: Map name is correct", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_map_ops completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 5: Hierarchical Register Block Test
  //==========================================================================

  class test_hierarchical_reg_block extends uvm_test;
    `uvm_component_utils(test_hierarchical_reg_block)

    parent_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = parent_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_block blocks[$];
      uvm_reg_map submaps[$];

      phase.raise_objection(this, "Testing hierarchical register blocks");

      // Test 5.1: Get child blocks
      `uvm_info("TEST", "=== Test 5.1: Get child blocks ===", UVM_NONE)

      reg_model.get_blocks(blocks, UVM_REG_NO_HIER);
      if (blocks.size() != 2)
        `uvm_error("TEST", $sformatf("Child block count mismatch: expected 2, got %0d", blocks.size()))
      else
        `uvm_info("TEST", "PASS: Parent has 2 child blocks", UVM_NONE)

      // Test 5.2: Get block by name
      `uvm_info("TEST", "=== Test 5.2: Get block by name ===", UVM_NONE)

      begin
        uvm_reg_block found_block = reg_model.get_block_by_name("child0");
        if (found_block == null)
          `uvm_error("TEST", "Failed to get child0 block by name")
        else if (found_block != reg_model.child0)
          `uvm_error("TEST", "get_block_by_name returned wrong block")
        else
          `uvm_info("TEST", "PASS: get_block_by_name works correctly", UVM_NONE)
      end

      // Test 5.3: Submap access
      `uvm_info("TEST", "=== Test 5.3: Submap access ===", UVM_NONE)

      reg_model.default_map.get_submaps(submaps, UVM_REG_NO_HIER);
      if (submaps.size() != 2)
        `uvm_error("TEST", $sformatf("Submap count mismatch: expected 2, got %0d", submaps.size()))
      else
        `uvm_info("TEST", "PASS: Parent map has 2 submaps", UVM_NONE)

      // Test 5.4: Submap offset
      `uvm_info("TEST", "=== Test 5.4: Submap offset ===", UVM_NONE)

      begin
        uvm_reg_addr_t offset = reg_model.default_map.get_submap_offset(reg_model.child0.default_map);
        if (offset != 'h100)
          `uvm_error("TEST", $sformatf("child0 submap offset mismatch: expected 0x100, got 0x%0h", offset))
        else
          `uvm_info("TEST", "PASS: child0 submap offset is 0x100", UVM_NONE)
      end

      begin
        uvm_reg_addr_t offset = reg_model.default_map.get_submap_offset(reg_model.child1.default_map);
        if (offset != 'h200)
          `uvm_error("TEST", $sformatf("child1 submap offset mismatch: expected 0x200, got 0x%0h", offset))
        else
          `uvm_info("TEST", "PASS: child1 submap offset is 0x200", UVM_NONE)
      end

      // Test 5.5: Child block parent
      `uvm_info("TEST", "=== Test 5.5: Child block parent ===", UVM_NONE)

      if (reg_model.child0.get_parent() != reg_model)
        `uvm_error("TEST", "child0 parent mismatch")
      else
        `uvm_info("TEST", "PASS: child0 parent is correct", UVM_NONE)

      // Test 5.6: Hierarchical register access
      `uvm_info("TEST", "=== Test 5.6: Hierarchical register access ===", UVM_NONE)

      // Set values in child registers
      reg_model.child0.data.set(32'hDEAD);
      reg_model.child1.data.set(32'hBEEF);

      if (reg_model.child0.data.get() != 32'hDEAD)
        `uvm_error("TEST", "child0 data value mismatch")
      else
        `uvm_info("TEST", "PASS: child0 data access works", UVM_NONE)

      if (reg_model.child1.data.get() != 32'hBEEF)
        `uvm_error("TEST", "child1 data value mismatch")
      else
        `uvm_info("TEST", "PASS: child1 data access works", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_hierarchical_reg_block completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 6: Register Predict Operations
  //==========================================================================

  class test_reg_predict extends uvm_test;
    `uvm_component_utils(test_reg_predict)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_data_t value;
      bit success;

      phase.raise_objection(this, "Testing register predict operations");

      // Test 6.1: Field predict
      `uvm_info("TEST", "=== Test 6.1: Field predict ===", UVM_NONE)

      success = reg_model.ctrl.enable.predict(1);
      value = reg_model.ctrl.enable.get_mirrored_value();
      if (value != 1)
        `uvm_error("TEST", $sformatf("Field predict failed: expected mirrored value 1, got %0d", value))
      else
        `uvm_info("TEST", "PASS: Field predict works correctly", UVM_NONE)

      // Test 6.2: Register predict
      `uvm_info("TEST", "=== Test 6.2: Register predict ===", UVM_NONE)

      success = reg_model.data.predict(32'hCAFE_BABE);
      value = reg_model.data.get_mirrored_value();
      if (value != 32'hCAFE_BABE)
        `uvm_error("TEST", $sformatf("Register predict failed: expected 0xCAFEBABE, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Register predict works correctly", UVM_NONE)

      // Test 6.3: Predict with UVM_PREDICT_WRITE
      `uvm_info("TEST", "=== Test 6.3: Predict with UVM_PREDICT_WRITE ===", UVM_NONE)

      success = reg_model.ctrl.mode.predict(4'h5, -1, UVM_PREDICT_WRITE);
      value = reg_model.ctrl.mode.get_mirrored_value();
      if (value != 4'h5)
        `uvm_error("TEST", $sformatf("Predict WRITE failed: expected 0x5, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Predict with UVM_PREDICT_WRITE works", UVM_NONE)

      // Test 6.4: Predict with UVM_PREDICT_READ
      `uvm_info("TEST", "=== Test 6.4: Predict with UVM_PREDICT_READ ===", UVM_NONE)

      success = reg_model.ctrl.status.predict(3'h7, -1, UVM_PREDICT_READ);
      value = reg_model.ctrl.status.get_mirrored_value();
      if (value != 3'h7)
        `uvm_error("TEST", $sformatf("Predict READ failed: expected 0x7, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Predict with UVM_PREDICT_READ works", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_predict completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 7: Register Adapter Test
  //==========================================================================

  // Simple bus transaction
  class simple_bus_txn extends uvm_sequence_item;
    `uvm_object_utils(simple_bus_txn)

    rand bit [31:0] addr;
    rand bit [31:0] data;
    rand bit write;

    function new(string name = "simple_bus_txn");
      super.new(name);
    endfunction
  endclass

  // Simple adapter for testing
  class simple_reg_adapter extends uvm_reg_adapter;
    `uvm_object_utils(simple_reg_adapter)

    function new(string name = "simple_reg_adapter");
      super.new(name);
      supports_byte_enable = 0;
      provides_responses = 0;
    endfunction

    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      simple_bus_txn txn = simple_bus_txn::type_id::create("txn");
      txn.addr = rw.addr;
      txn.data = rw.data;
      txn.write = (rw.kind == UVM_WRITE);
      return txn;
    endfunction

    virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
      simple_bus_txn txn;
      if (!$cast(txn, bus_item)) begin
        `uvm_fatal("CAST", "bus2reg: cast failed")
        return;
      end
      rw.addr = txn.addr;
      rw.data = txn.data;
      rw.kind = txn.write ? UVM_WRITE : UVM_READ;
      rw.status = UVM_IS_OK;
    endfunction
  endclass

  class test_reg_adapter extends uvm_test;
    `uvm_component_utils(test_reg_adapter)

    test_reg_block reg_model;
    simple_reg_adapter adapter;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
      adapter = simple_reg_adapter::type_id::create("adapter");
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_bus_op rw;
      uvm_sequence_item bus_item;

      phase.raise_objection(this, "Testing register adapter");

      // Test 7.1: reg2bus conversion
      `uvm_info("TEST", "=== Test 7.1: reg2bus conversion ===", UVM_NONE)

      rw.addr = 32'h100;
      rw.data = 32'hDEADBEEF;
      rw.kind = UVM_WRITE;
      rw.byte_en = 4'hF;

      bus_item = adapter.reg2bus(rw);
      if (bus_item == null)
        `uvm_error("TEST", "reg2bus returned null")
      else begin
        simple_bus_txn txn;
        if (!$cast(txn, bus_item))
          `uvm_error("TEST", "reg2bus cast failed")
        else begin
          if (txn.addr != 32'h100 || txn.data != 32'hDEADBEEF || txn.write != 1)
            `uvm_error("TEST", "reg2bus conversion mismatch")
          else
            `uvm_info("TEST", "PASS: reg2bus conversion works correctly", UVM_NONE)
        end
      end

      // Test 7.2: bus2reg conversion
      `uvm_info("TEST", "=== Test 7.2: bus2reg conversion ===", UVM_NONE)

      begin
        simple_bus_txn txn = simple_bus_txn::type_id::create("txn");
        uvm_reg_bus_op rw2;

        txn.addr = 32'h200;
        txn.data = 32'hCAFEBABE;
        txn.write = 0;

        adapter.bus2reg(txn, rw2);
        if (rw2.addr != 32'h200 || rw2.data != 32'hCAFEBABE || rw2.kind != UVM_READ)
          `uvm_error("TEST", "bus2reg conversion mismatch")
        else
          `uvm_info("TEST", "PASS: bus2reg conversion works correctly", UVM_NONE)
      end

      // Test 7.3: Adapter properties
      `uvm_info("TEST", "=== Test 7.3: Adapter properties ===", UVM_NONE)

      if (adapter.supports_byte_enable != 0)
        `uvm_error("TEST", "supports_byte_enable should be 0")
      else
        `uvm_info("TEST", "PASS: supports_byte_enable is correct", UVM_NONE)

      if (adapter.provides_responses != 0)
        `uvm_error("TEST", "provides_responses should be 0")
      else
        `uvm_info("TEST", "PASS: provides_responses is correct", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_adapter completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 8: uvm_reg_item Test
  //==========================================================================

  class test_reg_item extends uvm_test;
    `uvm_component_utils(test_reg_item)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_item item;

      phase.raise_objection(this, "Testing uvm_reg_item");

      // Test 8.1: Create reg item
      `uvm_info("TEST", "=== Test 8.1: Create reg item ===", UVM_NONE)

      item = new("test_item");
      if (item == null)
        `uvm_error("TEST", "Failed to create uvm_reg_item")
      else
        `uvm_info("TEST", "PASS: uvm_reg_item created successfully", UVM_NONE)

      // Test 8.2: Set item properties
      `uvm_info("TEST", "=== Test 8.2: Set item properties ===", UVM_NONE)

      item.element = reg_model.ctrl;
      // Note: element_kind is a uvm_object, not an enum in the CIRCT UVM implementation
      item.kind = UVM_WRITE;
      item.value.push_back(32'h12345678);
      item.offset = 'h00;
      item.map = reg_model.default_map;

      if (item.element != reg_model.ctrl)
        `uvm_error("TEST", "Item element mismatch")
      else
        `uvm_info("TEST", "PASS: Item element set correctly", UVM_NONE)

      if (item.kind != UVM_WRITE)
        `uvm_error("TEST", "Item kind mismatch")
      else
        `uvm_info("TEST", "PASS: Item kind set correctly", UVM_NONE)

      if (item.value[0] != 32'h12345678)
        `uvm_error("TEST", "Item value mismatch")
      else
        `uvm_info("TEST", "PASS: Item value set correctly", UVM_NONE)

      // Test 8.3: Item type name
      `uvm_info("TEST", "=== Test 8.3: Item type name ===", UVM_NONE)

      if (item.get_type_name() != "uvm_reg_item")
        `uvm_error("TEST", $sformatf("Item type name mismatch: expected uvm_reg_item, got %s",
                                     item.get_type_name()))
      else
        `uvm_info("TEST", "PASS: Item type name is correct", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_reg_item completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 9: Field Access Types Test
  //==========================================================================

  class access_type_reg extends uvm_reg;
    `uvm_object_utils(access_type_reg)

    rand uvm_reg_field rw_field;
    rand uvm_reg_field ro_field;
    rand uvm_reg_field wo_field;
    rand uvm_reg_field w1c_field;
    rand uvm_reg_field rc_field;
    rand uvm_reg_field rs_field;
    rand uvm_reg_field wrc_field;
    rand uvm_reg_field wrs_field;

    function new(string name = "access_type_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      rw_field = new("rw_field");
      ro_field = new("ro_field");
      wo_field = new("wo_field");
      w1c_field = new("w1c_field");
      rc_field = new("rc_field");
      rs_field = new("rs_field");
      wrc_field = new("wrc_field");
      wrs_field = new("wrs_field");

      rw_field.configure(this, 4, 0, "RW", 0, 4'h0, 1, 1, 0);
      ro_field.configure(this, 4, 4, "RO", 0, 4'h0, 1, 0, 0);
      wo_field.configure(this, 4, 8, "WO", 0, 4'h0, 1, 1, 0);
      w1c_field.configure(this, 4, 12, "W1C", 0, 4'h0, 1, 1, 0);
      rc_field.configure(this, 4, 16, "RC", 0, 4'h0, 1, 0, 0);
      rs_field.configure(this, 4, 20, "RS", 0, 4'h0, 1, 0, 0);
      wrc_field.configure(this, 4, 24, "WRC", 0, 4'h0, 1, 1, 0);
      wrs_field.configure(this, 4, 28, "WRS", 0, 4'h0, 1, 1, 0);
    endfunction
  endclass

  class test_field_access_types extends uvm_test;
    `uvm_component_utils(test_field_access_types)

    access_type_reg reg_inst;
    uvm_reg_block reg_block;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      uvm_reg_map map;
      super.build_phase(phase);

      // uvm_reg_block doesn't have type_id, use new() directly
      reg_block = new("reg_block");
      reg_block.configure();

      reg_inst = access_type_reg::type_id::create("access_reg");
      reg_inst.configure(reg_block, null, "");
      reg_inst.build();

      map = reg_block.create_map("default_map", 'h0, 4, UVM_LITTLE_ENDIAN);
      map.add_reg(reg_inst, 'h00, "RW");
      reg_block.lock_model();
    endfunction

    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Testing field access types");

      // Test 9.1: RW access
      `uvm_info("TEST", "=== Test 9.1: RW access type ===", UVM_NONE)
      if (reg_inst.rw_field.get_access() != "RW")
        `uvm_error("TEST", $sformatf("RW access mismatch: got %s", reg_inst.rw_field.get_access()))
      else
        `uvm_info("TEST", "PASS: RW access type correct", UVM_NONE)

      // Test 9.2: RO access
      `uvm_info("TEST", "=== Test 9.2: RO access type ===", UVM_NONE)
      if (reg_inst.ro_field.get_access() != "RO")
        `uvm_error("TEST", $sformatf("RO access mismatch: got %s", reg_inst.ro_field.get_access()))
      else
        `uvm_info("TEST", "PASS: RO access type correct", UVM_NONE)

      // Test 9.3: WO access
      `uvm_info("TEST", "=== Test 9.3: WO access type ===", UVM_NONE)
      if (reg_inst.wo_field.get_access() != "WO")
        `uvm_error("TEST", $sformatf("WO access mismatch: got %s", reg_inst.wo_field.get_access()))
      else
        `uvm_info("TEST", "PASS: WO access type correct", UVM_NONE)

      // Test 9.4: W1C access
      `uvm_info("TEST", "=== Test 9.4: W1C access type ===", UVM_NONE)
      if (reg_inst.w1c_field.get_access() != "W1C")
        `uvm_error("TEST", $sformatf("W1C access mismatch: got %s", reg_inst.w1c_field.get_access()))
      else
        `uvm_info("TEST", "PASS: W1C access type correct", UVM_NONE)

      // Test 9.5: RC access
      `uvm_info("TEST", "=== Test 9.5: RC access type ===", UVM_NONE)
      if (reg_inst.rc_field.get_access() != "RC")
        `uvm_error("TEST", $sformatf("RC access mismatch: got %s", reg_inst.rc_field.get_access()))
      else
        `uvm_info("TEST", "PASS: RC access type correct", UVM_NONE)

      // Test 9.6: RS access
      `uvm_info("TEST", "=== Test 9.6: RS access type ===", UVM_NONE)
      if (reg_inst.rs_field.get_access() != "RS")
        `uvm_error("TEST", $sformatf("RS access mismatch: got %s", reg_inst.rs_field.get_access()))
      else
        `uvm_info("TEST", "PASS: RS access type correct", UVM_NONE)

      // Test 9.7: WRC access
      `uvm_info("TEST", "=== Test 9.7: WRC access type ===", UVM_NONE)
      if (reg_inst.wrc_field.get_access() != "WRC")
        `uvm_error("TEST", $sformatf("WRC access mismatch: got %s", reg_inst.wrc_field.get_access()))
      else
        `uvm_info("TEST", "PASS: WRC access type correct", UVM_NONE)

      // Test 9.8: WRS access
      `uvm_info("TEST", "=== Test 9.8: WRS access type ===", UVM_NONE)
      if (reg_inst.wrs_field.get_access() != "WRS")
        `uvm_error("TEST", $sformatf("WRS access mismatch: got %s", reg_inst.wrs_field.get_access()))
      else
        `uvm_info("TEST", "PASS: WRS access type correct", UVM_NONE)

      // Test 9.9: is_known_access
      `uvm_info("TEST", "=== Test 9.9: is_known_access ===", UVM_NONE)
      if (!reg_inst.rw_field.is_known_access())
        `uvm_error("TEST", "RW should be known access")
      else
        `uvm_info("TEST", "PASS: is_known_access works correctly", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_field_access_types completed", UVM_NONE)
    endfunction
  endclass

  //==========================================================================
  // TEST 10: Comprehensive RAL Integration Test
  //==========================================================================

  class test_ral_integration extends uvm_test;
    `uvm_component_utils(test_ral_integration)

    test_reg_block reg_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_model = test_reg_block::type_id::create("reg_model");
      reg_model.build();
    endfunction

    virtual task run_phase(uvm_phase phase);
      uvm_reg_data_t value;
      uvm_reg regs[$];
      uvm_reg_field fields[$];

      phase.raise_objection(this, "Testing RAL integration");

      // Integration test: Full workflow
      `uvm_info("TEST", "=== RAL Integration Test ===", UVM_NONE)

      // Step 1: Reset all registers
      `uvm_info("TEST", "Step 1: Reset all registers", UVM_NONE)
      reg_model.reset();

      // Step 2: Verify reset values
      `uvm_info("TEST", "Step 2: Verify reset values", UVM_NONE)
      value = reg_model.ctrl.get();
      if (value != 0)
        `uvm_error("TEST", $sformatf("Ctrl not reset: 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Ctrl is reset to 0", UVM_NONE)

      // Step 3: Program registers using field access
      `uvm_info("TEST", "Step 3: Program registers via fields", UVM_NONE)
      reg_model.ctrl.enable.set(1);
      reg_model.ctrl.mode.set(4'h5);
      reg_model.data.data.set(32'hDEAD_BEEF);
      reg_model.int_en.global_en.set(1);
      reg_model.int_en.int_mask.set(8'hFF);

      // Step 4: Verify programmed values
      `uvm_info("TEST", "Step 4: Verify programmed values", UVM_NONE)

      value = reg_model.ctrl.get();
      // enable=1 at bit 0, mode=5 at bits 1-4 = 0b01011 = 0xB
      if ((value & 32'h1F) != 32'h0B)
        `uvm_error("TEST", $sformatf("Ctrl value mismatch: expected 0x0B, got 0x%0h", value & 32'h1F))
      else
        `uvm_info("TEST", "PASS: Ctrl value is correct", UVM_NONE)

      value = reg_model.data.get();
      if (value != 32'hDEAD_BEEF)
        `uvm_error("TEST", $sformatf("Data value mismatch: expected 0xDEADBEEF, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Data value is correct", UVM_NONE)

      value = reg_model.int_en.get();
      // global_en=1 at bit 0, int_mask=0xFF at bits 8-15 = 0xFF01
      if (value != 32'h0000_FF01)
        `uvm_error("TEST", $sformatf("Int_en value mismatch: expected 0xFF01, got 0x%0h", value))
      else
        `uvm_info("TEST", "PASS: Int_en value is correct", UVM_NONE)

      // Step 5: Iterate through all registers
      `uvm_info("TEST", "Step 5: Iterate through all registers", UVM_NONE)
      reg_model.get_registers(regs);
      foreach (regs[i]) begin
        `uvm_info("TEST", $sformatf("  Register: %s, offset=0x%0h, value=0x%0h",
                                    regs[i].get_name(),
                                    regs[i].get_offset(reg_model.default_map),
                                    regs[i].get()), UVM_MEDIUM)
      end
      `uvm_info("TEST", $sformatf("PASS: Iterated through %0d registers", regs.size()), UVM_NONE)

      // Step 6: Iterate through fields of ctrl register
      `uvm_info("TEST", "Step 6: Iterate through ctrl fields", UVM_NONE)
      reg_model.ctrl.get_fields(fields);
      foreach (fields[i]) begin
        `uvm_info("TEST", $sformatf("  Field: %s, size=%0d, lsb=%0d, access=%s, value=0x%0h",
                                    fields[i].get_name(),
                                    fields[i].get_n_bits(),
                                    fields[i].get_lsb_pos(),
                                    fields[i].get_access(),
                                    fields[i].get()), UVM_MEDIUM)
      end
      `uvm_info("TEST", $sformatf("PASS: Iterated through %0d fields", fields.size()), UVM_NONE)

      // Step 7: Predict and verify mirrored values
      `uvm_info("TEST", "Step 7: Predict and verify mirrored values", UVM_NONE)
      void'(reg_model.ctrl.predict(32'h0000_003F));
      value = reg_model.ctrl.get_mirrored_value();
      if ((value & 32'h3F) != 32'h3F)
        `uvm_error("TEST", $sformatf("Mirrored value mismatch: expected 0x3F, got 0x%0h", value & 32'h3F))
      else
        `uvm_info("TEST", "PASS: Mirrored value prediction works", UVM_NONE)

      `uvm_info("TEST", "=== RAL Integration Test Complete ===", UVM_NONE)

      phase.drop_objection(this, "Test complete");
    endtask

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "test_ral_integration completed", UVM_NONE)
    endfunction
  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import uvm_ral_test_pkg::*;

  initial begin
    `uvm_info("TB", "UVM RAL Test Starting", UVM_NONE)
    // Run the field operations test by default
    // Other tests can be selected via +UVM_TESTNAME
    run_test("test_reg_field_ops");
  end

endmodule
