// RUN: crun %s --uvm-path=%S/../../../lib/Runtime/uvm-core --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_reg_predictor setup: create predictor, set map and adapter.

// CHECK: [TEST] predictor created: PASS
// CHECK: [TEST] adapter created: PASS
// CHECK: [TEST] predictor name: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class pred_reg extends uvm_reg;
    `uvm_object_utils(pred_reg)
    rand uvm_reg_field data;

    function new(string name = "pred_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction

    virtual function void build();
      data = uvm_reg_field::type_id::create("data");
      data.configure(this, 32, 0, "RW", 0, 0, 1, 1, 1);
    endfunction
  endclass

  class pred_block extends uvm_reg_block;
    `uvm_object_utils(pred_block)
    pred_reg r;
    function new(string name = "pred_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      r = pred_reg::type_id::create("r");
      r.configure(this);
      r.build();
      default_map = create_map("map", 0, 4, UVM_LITTLE_ENDIAN);
      default_map.add_reg(r, 0);
      lock_model();
    endfunction
  endclass

  class my_adapter extends uvm_reg_adapter;
    `uvm_object_utils(my_adapter)
    function new(string name = "my_adapter");
      super.new(name);
    endfunction
    virtual function uvm_sequence_item reg2bus(const ref uvm_reg_bus_op rw);
      return null;
    endfunction
    virtual function void bus2reg(uvm_sequence_item bus_item, ref uvm_reg_bus_op rw);
    endfunction
  endclass

  class my_bus_item extends uvm_sequence_item;
    `uvm_object_utils(my_bus_item)
    function new(string name = "my_bus_item");
      super.new(name);
    endfunction
  endclass

  class ral_pred_test extends uvm_test;
    `uvm_component_utils(ral_pred_test)
    pred_block blk;
    uvm_reg_predictor#(my_bus_item) pred;
    my_adapter adapter;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      blk = pred_block::type_id::create("blk");
      blk.configure(null, "");
      blk.build();

      pred = uvm_reg_predictor#(my_bus_item)::type_id::create("pred", this);
      adapter = my_adapter::type_id::create("adapter");
      pred.map = blk.default_map;
      pred.adapter = adapter;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      if (pred != null)
        `uvm_info("TEST", "predictor created: PASS", UVM_LOW)
      else `uvm_error("TEST", "predictor created: FAIL")

      if (adapter != null)
        `uvm_info("TEST", "adapter created: PASS", UVM_LOW)
      else `uvm_error("TEST", "adapter created: FAIL")

      if (pred.get_name() == "pred")
        `uvm_info("TEST", "predictor name: PASS", UVM_LOW)
      else `uvm_error("TEST", "predictor name: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ral_pred_test");
endmodule
