// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: callbacks + factory override — callback fires on derived instance.

// CHECK: callback fired on derived component
// CHECK: [TEST] derived process() called
// CHECK: [TEST] callbacks-factory: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class integ_cb_base extends uvm_component;
    `uvm_component_utils(integ_cb_base)
    int cb_fired;
    int process_called;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      cb_fired = 0;
      process_called = 0;
    endfunction
    virtual function void do_process();
      process_called = 1;
    endfunction
  endclass

  class integ_cb_derived extends integ_cb_base;
    `uvm_component_utils(integ_cb_derived)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    virtual function void do_process();
      process_called = 1;
      `uvm_info("TEST", "derived process() called", UVM_LOW)
    endfunction
  endclass

  // Callback class
  class integ_cb extends uvm_callback;
    `uvm_object_utils(integ_cb)
    int fired;
    function new(string name = "integ_cb");
      super.new(name);
      fired = 0;
    endfunction
    virtual function void on_event(integ_cb_base comp);
      fired = 1;
      comp.cb_fired = 1;
      `uvm_info("TEST", "callback fired on derived component", UVM_LOW)
    endfunction
  endclass

  class integ_cbf_test extends uvm_test;
    `uvm_component_utils(integ_cbf_test)
    integ_cb_base comp;
    integ_cb my_cb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      // Factory override: base → derived
      integ_cb_base::type_id::set_type_override(integ_cb_derived::get_type());
      comp = integ_cb_base::type_id::create("comp", this);
      my_cb = integ_cb::type_id::create("my_cb");
    endfunction
    task run_phase(uvm_phase phase);
      int pass = 1;
      phase.raise_objection(this);
      // Manually invoke callback
      my_cb.on_event(comp);
      comp.do_process();
      if (!comp.cb_fired) begin
        `uvm_error("TEST", "callback did not fire")
        pass = 0;
      end
      if (!comp.process_called) begin
        `uvm_error("TEST", "process() not called")
        pass = 0;
      end
      if (pass)
        `uvm_info("TEST", "callbacks-factory: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "callbacks-factory: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_cbf_test");
endmodule
