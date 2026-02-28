// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: uvm_reg nested class triggers slang non-static member access error

// Integration: RAL register block + config_db + env hierarchy.

// CHECK: [TEST] sub_comp received reg_block via config_db
// CHECK: [TEST] reg 'ctrl' found at offset 0x0
// CHECK: [TEST] reg 'status' found at offset 0x4
// CHECK: [TEST] ral-config-env: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_ral_ctrl_reg extends uvm_reg;
    `uvm_object_utils(integ_ral_ctrl_reg)
    rand uvm_reg_field enable;
    function new(string name = "integ_ral_ctrl_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      enable = uvm_reg_field::create("enable");
      enable.configure(this, 1, 0, "RW", 0, 1'h0, 1, 1, 0);
    endfunction
  endclass

  class integ_ral_status_reg extends uvm_reg;
    `uvm_object_utils(integ_ral_status_reg)
    rand uvm_reg_field busy;
    function new(string name = "integ_ral_status_reg");
      super.new(name, 32, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      busy = uvm_reg_field::create("busy");
      busy.configure(this, 1, 0, "RO", 0, 1'h0, 1, 0, 0);
    endfunction
  endclass

  class integ_ral_block extends uvm_reg_block;
    `uvm_object_utils(integ_ral_block)
    rand integ_ral_ctrl_reg ctrl;
    rand integ_ral_status_reg status;
    function new(string name = "integ_ral_block");
      super.new(name, UVM_NO_COVERAGE);
    endfunction
    virtual function void build();
      default_map = create_map("default_map", 0, 4, UVM_LITTLE_ENDIAN);
      ctrl = integ_ral_ctrl_reg::type_id::create("ctrl");
      ctrl.configure(this, null, "");
      ctrl.build();
      default_map.add_reg(ctrl, 'h0, "RW");
      status = integ_ral_status_reg::type_id::create("status");
      status.configure(this, null, "");
      status.build();
      default_map.add_reg(status, 'h4, "RO");
      lock_model();
    endfunction
  endclass

  class integ_ral_sub_comp extends uvm_component;
    `uvm_component_utils(integ_ral_sub_comp)
    integ_ral_block rb;
    int got_block;
    uvm_reg_addr_t ctrl_off;
    uvm_reg_addr_t status_off;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      got_block = 0;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (uvm_config_db#(integ_ral_block)::get(this, "", "reg_block", rb)) begin
        got_block = 1;
        ctrl_off = rb.ctrl.get_offset();
        status_off = rb.status.get_offset();
      end
    endfunction
  endclass

  class integ_ral_test extends uvm_test;
    `uvm_component_utils(integ_ral_test)
    integ_ral_block reg_blk;
    integ_ral_sub_comp sub;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      reg_blk = integ_ral_block::type_id::create("reg_blk");
      reg_blk.build();
      uvm_config_db#(integ_ral_block)::set(this, "sub", "reg_block", reg_blk);
      sub = integ_ral_sub_comp::type_id::create("sub", this);
    endfunction
    task run_phase(uvm_phase phase);
      int pass = 1;
      phase.raise_objection(this);
      if (sub.got_block)
        `uvm_info("TEST", "sub_comp received reg_block via config_db", UVM_LOW)
      else begin
        `uvm_error("TEST", "sub_comp did NOT receive reg_block")
        pass = 0;
      end
      if (sub.ctrl_off == 0)
        `uvm_info("TEST", "reg 'ctrl' found at offset 0x0", UVM_LOW)
      else begin
        `uvm_error("TEST", $sformatf("ctrl offset=0x%0h", sub.ctrl_off))
        pass = 0;
      end
      if (sub.status_off == 4)
        `uvm_info("TEST", "reg 'status' found at offset 0x4", UVM_LOW)
      else begin
        `uvm_error("TEST", $sformatf("status offset=0x%0h", sub.status_off))
        pass = 0;
      end
      if (pass)
        `uvm_info("TEST", "ral-config-env: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "ral-config-env: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_ral_test");
endmodule
