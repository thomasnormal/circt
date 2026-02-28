// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// Test nonblocking TLM: uvm_nonblocking_put_port and uvm_nonblocking_put_imp.
// CHECK: [TEST] try_put success: PASS
// CHECK: [TEST] can_put query: PASS
// CHECK: [TEST] try_put value received: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class nb_put_target extends uvm_component;
    `uvm_component_utils(nb_put_target)
    uvm_nonblocking_put_imp #(int, nb_put_target) put_export;
    int last_val;
    bit ready;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      ready = 1;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      put_export = new("put_export", this);
    endfunction
    function bit try_put(int val);
      if (ready) begin
        last_val = val;
        return 1;
      end
      return 0;
    endfunction
    function bit can_put();
      return ready;
    endfunction
  endclass

  class nb_put_test extends uvm_test;
    `uvm_component_utils(nb_put_test)
    uvm_nonblocking_put_port #(int) nb_port;
    nb_put_target target;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      nb_port = new("nb_port", this);
      target = nb_put_target::type_id::create("target", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      nb_port.connect(target.put_export);
    endfunction
    task run_phase(uvm_phase phase);
      bit ok;
      phase.raise_objection(this);
      ok = nb_port.try_put(55);
      if (ok)
        `uvm_info("TEST", "try_put success: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "try_put success: FAIL")
      if (nb_port.can_put())
        `uvm_info("TEST", "can_put query: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "can_put query: FAIL")
      if (target.last_val == 55)
        `uvm_info("TEST", "try_put value received: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("try_put value received: FAIL got %0d", target.last_val))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("nb_put_test");
endmodule
