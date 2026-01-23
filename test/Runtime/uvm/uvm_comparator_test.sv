//===----------------------------------------------------------------------===//
// UVM Comparator Test - Tests for uvm_in_order_comparator and related classes
//===----------------------------------------------------------------------===//
// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s

`timescale 1ns/1ps

`include "uvm_macros.svh"

package comparator_test_pkg;
  import uvm_pkg::*;

  //==========================================================================
  // Simple Transaction for Comparator Testing
  //==========================================================================
  class simple_item extends uvm_sequence_item;
    `uvm_object_utils(simple_item)

    rand bit [7:0] data;
    rand bit [3:0] addr;
    rand bit [1:0] cmd;

    function new(string name = "simple_item");
      super.new(name);
    endfunction

    virtual function void do_copy(uvm_object rhs);
      simple_item rhs_item;
      super.do_copy(rhs);
      if (!$cast(rhs_item, rhs))
        `uvm_fatal("CAST", "Cast failed in do_copy")
      data = rhs_item.data;
      addr = rhs_item.addr;
      cmd = rhs_item.cmd;
    endfunction

    virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      simple_item rhs_item;
      if (!$cast(rhs_item, rhs))
        return 0;
      return (data == rhs_item.data) && (addr == rhs_item.addr) && (cmd == rhs_item.cmd);
    endfunction

    virtual function string convert2string();
      return $sformatf("addr=%0h, data=%0h, cmd=%0h", addr, data, cmd);
    endfunction

  endclass

  //==========================================================================
  // Transformer class for algorithmic comparator
  //==========================================================================
  class data_transformer extends uvm_object;
    `uvm_object_utils(data_transformer)

    function new(string name = "data_transformer");
      super.new(name);
    endfunction

    // Transform: increment data by 1
    virtual function simple_item transform(simple_item input_item);
      simple_item output_item = new("transformed");
      output_item.data = input_item.data + 1;
      output_item.addr = input_item.addr;
      output_item.cmd = input_item.cmd;
      return output_item;
    endfunction
  endclass

  //==========================================================================
  // Custom algorithmic comparator with transform override
  //==========================================================================
  class my_algorithmic_comparator extends uvm_algorithmic_comparator #(
    simple_item, simple_item, data_transformer);

    `uvm_component_utils(my_algorithmic_comparator)

    function new(string name, uvm_component parent, data_transformer transformer = null);
      super.new(name, parent, transformer);
    endfunction

    // Override transform to use our transformer
    protected virtual function simple_item transform(simple_item t);
      if (m_transformer != null)
        return m_transformer.transform(t);
      return t;
    endfunction
  endclass

  //==========================================================================
  // Scoreboard using in-order comparator
  //==========================================================================
  class comparator_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(comparator_scoreboard)

    // In-order comparator
    uvm_in_order_comparator #(simple_item) in_order_cmp;

    // Built-in comparator
    uvm_in_order_built_in_comparator #(simple_item) builtin_cmp;

    // Algorithmic comparator
    my_algorithmic_comparator algo_cmp;
    data_transformer transformer;

    // Analysis FIFOs for connecting
    uvm_tlm_analysis_fifo #(simple_item) expected_fifo;
    uvm_tlm_analysis_fifo #(simple_item) actual_fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);

      // Create comparators
      in_order_cmp = new("in_order_cmp", this);
      builtin_cmp = new("builtin_cmp", this);

      // Create transformer and algorithmic comparator
      transformer = new("transformer");
      algo_cmp = new("algo_cmp", this, transformer);

      // Create analysis FIFOs
      expected_fifo = new("expected_fifo", this);
      actual_fifo = new("actual_fifo", this);
    endfunction

    virtual function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      // Connections would happen here in a real testbench
    endfunction

    // Test the in-order comparator with matching items
    function void test_matching_items();
      simple_item item1, item2;

      `uvm_info("TEST", "Testing in-order comparator with matching items", UVM_NONE)

      // Create matching items
      item1 = new("item1");
      item1.data = 8'hAB;
      item1.addr = 4'h5;
      item1.cmd = 2'b01;

      item2 = new("item2");
      item2.data = 8'hAB;
      item2.addr = 4'h5;
      item2.cmd = 2'b01;

      // Send to comparator
      in_order_cmp.write_before(item1);
      in_order_cmp.write_after(item2);

      `uvm_info("TEST", $sformatf("In-order comparator matches: %0d, mismatches: %0d",
        in_order_cmp.get_matches(), in_order_cmp.get_mismatches()), UVM_NONE)
    endfunction

    // Test the in-order comparator with mismatching items
    function void test_mismatching_items();
      simple_item item1, item2;

      `uvm_info("TEST", "Testing in-order comparator with mismatching items", UVM_NONE)

      // Create mismatching items
      item1 = new("item1");
      item1.data = 8'hAB;
      item1.addr = 4'h5;
      item1.cmd = 2'b01;

      item2 = new("item2");
      item2.data = 8'hCD;  // Different data
      item2.addr = 4'h5;
      item2.cmd = 2'b01;

      // Send to comparator
      in_order_cmp.write_before(item1);
      in_order_cmp.write_after(item2);

      `uvm_info("TEST", $sformatf("In-order comparator matches: %0d, mismatches: %0d",
        in_order_cmp.get_matches(), in_order_cmp.get_mismatches()), UVM_NONE)
    endfunction

    // Test the built-in comparator
    function void test_builtin_comparator();
      simple_item item1, item2;

      `uvm_info("TEST", "Testing built-in comparator", UVM_NONE)

      // Create matching items
      item1 = new("item1");
      item1.data = 8'h55;
      item1.addr = 4'hA;
      item1.cmd = 2'b10;

      item2 = new("item2");
      item2.data = 8'h55;
      item2.addr = 4'hA;
      item2.cmd = 2'b10;

      // Send to comparator
      builtin_cmp.write_before(item1);
      builtin_cmp.write_after(item2);

      `uvm_info("TEST", $sformatf("Built-in comparator matches: %0d, mismatches: %0d",
        builtin_cmp.get_matches(), builtin_cmp.get_mismatches()), UVM_NONE)
    endfunction

    // Test the algorithmic comparator
    function void test_algorithmic_comparator();
      simple_item before_item, after_item;

      `uvm_info("TEST", "Testing algorithmic comparator with transform", UVM_NONE)

      // Create items where after = transform(before)
      // Transform increments data by 1
      before_item = new("before_item");
      before_item.data = 8'h10;
      before_item.addr = 4'h3;
      before_item.cmd = 2'b00;

      after_item = new("after_item");
      after_item.data = 8'h11;  // 0x10 + 1 = 0x11
      after_item.addr = 4'h3;
      after_item.cmd = 2'b00;

      // Send to algorithmic comparator
      algo_cmp.write_before(before_item);
      algo_cmp.write_after(after_item);

      `uvm_info("TEST", $sformatf("Algorithmic comparator matches: %0d, mismatches: %0d",
        algo_cmp.get_matches(), algo_cmp.get_mismatches()), UVM_NONE)
    endfunction

    // Test ordering - items should be compared in order
    function void test_ordering();
      simple_item item_a1, item_a2, item_b1, item_b2;

      `uvm_info("TEST", "Testing in-order comparison ordering", UVM_NONE)

      // Create first pair
      item_a1 = new("item_a1");
      item_a1.data = 8'h11;
      item_a1.addr = 4'h1;
      item_a1.cmd = 2'b00;

      item_a2 = new("item_a2");
      item_a2.data = 8'h11;
      item_a2.addr = 4'h1;
      item_a2.cmd = 2'b00;

      // Create second pair
      item_b1 = new("item_b1");
      item_b1.data = 8'h22;
      item_b1.addr = 4'h2;
      item_b1.cmd = 2'b01;

      item_b2 = new("item_b2");
      item_b2.data = 8'h22;
      item_b2.addr = 4'h2;
      item_b2.cmd = 2'b01;

      // Create a new comparator for this test
      begin
        uvm_in_order_comparator #(simple_item) order_cmp;
        order_cmp = new("order_cmp", this);

        // Send before items first
        order_cmp.write_before(item_a1);
        order_cmp.write_before(item_b1);

        // Now send after items - should match in order
        order_cmp.write_after(item_a2);
        order_cmp.write_after(item_b2);

        `uvm_info("TEST", $sformatf("Ordering test - matches: %0d, mismatches: %0d",
          order_cmp.get_matches(), order_cmp.get_mismatches()), UVM_NONE)
      end
    endfunction

    // Run all tests
    virtual task run_phase(uvm_phase phase);
      phase.raise_objection(this, "Running comparator tests");

      test_matching_items();
      test_mismatching_items();
      test_builtin_comparator();
      test_algorithmic_comparator();
      test_ordering();

      `uvm_info("TEST", "All comparator tests completed", UVM_NONE)

      phase.drop_objection(this, "Finished comparator tests");
    endtask

  endclass

  //==========================================================================
  // Test Environment
  //==========================================================================
  class comparator_test_env extends uvm_env;
    `uvm_component_utils(comparator_test_env)

    comparator_scoreboard sb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sb = comparator_scoreboard::type_id::create("sb", this);
    endfunction

  endclass

  //==========================================================================
  // Test
  //==========================================================================
  class comparator_test extends uvm_test;
    `uvm_component_utils(comparator_test)

    comparator_test_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    virtual function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = comparator_test_env::type_id::create("env", this);
    endfunction

    virtual function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "Comparator test completed", UVM_NONE)
    endfunction

  endclass

endpackage

//==========================================================================
// Top Module
//==========================================================================
module tb_top;
  import uvm_pkg::*;
  import comparator_test_pkg::*;

  initial begin
    `uvm_info("TB", "Starting UVM Comparator Test", UVM_NONE)
    run_test("comparator_test");
  end

endmodule
