// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test uvm_put_port/uvm_put_imp and uvm_get_port/uvm_get_imp connections.
// CHECK: [TEST] put_imp received value: PASS
// CHECK: [TEST] get_imp provides value: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class put_receiver extends uvm_component;
    `uvm_component_utils(put_receiver)
    uvm_put_imp #(int, put_receiver) put_export;
    int last_val;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      put_export = new("put_export", this);
    endfunction
    task put(int val);
      last_val = val;
    endtask
    function bit try_put(int val);
      last_val = val;
      return 1;
    endfunction
    function bit can_put();
      return 1;
    endfunction
  endclass

  class get_provider extends uvm_component;
    `uvm_component_utils(get_provider)
    uvm_get_imp #(int, get_provider) get_export;
    int provide_val;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      get_export = new("get_export", this);
    endfunction
    task get(output int val);
      val = provide_val;
    endtask
    function bit try_get(output int val);
      val = provide_val;
      return 1;
    endfunction
    function bit can_get();
      return 1;
    endfunction
  endclass

  class put_get_test extends uvm_test;
    `uvm_component_utils(put_get_test)
    uvm_put_port #(int) put_port;
    uvm_get_port #(int) get_port;
    put_receiver receiver;
    get_provider provider;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      put_port = new("put_port", this);
      get_port = new("get_port", this);
      receiver = put_receiver::type_id::create("receiver", this);
      provider = get_provider::type_id::create("provider", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      put_port.connect(receiver.put_export);
      get_port.connect(provider.get_export);
    endfunction
    task run_phase(uvm_phase phase);
      int got;
      phase.raise_objection(this);
      fork begin put_port.put(42); end join_any
      if (receiver.last_val == 42)
        `uvm_info("TEST", "put_imp received value: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "put_imp received value: FAIL")
      provider.provide_val = 99;
      fork begin get_port.get(got); end join_any
      if (got == 99)
        `uvm_info("TEST", "get_imp provides value: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("get_imp provides value: FAIL got %0d", got))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("put_get_test");
endmodule
