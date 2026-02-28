// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm-core %s 2>&1 | FileCheck %s
// RUN: circt-verilog --ir-hw --uvm-path=%S/../../../lib/Runtime/uvm-core %s -o %t.mlir
// RUN: circt-sim %t.mlir --top uvm_port_connect_semantic_top --max-time=2000000000 2>&1 | FileCheck %s --check-prefix=SIM
// REQUIRES: slang
// REQUIRES: circt-sim

`include "uvm_macros.svh"
import uvm_pkg::*;

class connect_env extends uvm_env;
  `uvm_component_utils(connect_env)

  uvm_blocking_put_port #(int) put_p;
  uvm_blocking_get_port #(int) get_p;
  uvm_tlm_fifo #(int) fifo;

  function new(string name = "connect_env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    put_p = new("put_p", this);
    get_p = new("get_p", this);
    fifo = new("fifo", this, 2);
  endfunction

  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    put_p.connect(fifo.put_export);
    get_p.connect(fifo.get_peek_export);
  endfunction

  virtual function void end_of_elaboration_phase(uvm_phase phase);
    uvm_port_base #(uvm_tlm_if_base #(int, int)) connected[string];
    int putCount;
    int getCount;
    super.end_of_elaboration_phase(phase);
    put_p.get_connected_to(connected);
    putCount = connected.num();
    get_p.get_connected_to(connected);
    getCount = connected.num();
    if (putCount == 1 && getCount == 1)
      `uvm_info("CONNECT", "CONNECT PASS: put/get connected_to count == 1", UVM_NONE)
    else
      `uvm_error("CONNECT",
                 $sformatf("CONNECT FAIL: put=%0d get=%0d", putCount, getCount))
  endfunction

  virtual task run_phase(uvm_phase phase);
    int value;
    super.run_phase(phase);
    phase.raise_objection(this);
    put_p.put(55);
    get_p.get(value);
    if (value == 55)
      `uvm_info("DATA", "DATA PASS: got expected value 55", UVM_NONE)
    else
      `uvm_error("DATA", $sformatf("DATA FAIL: got %0d", value))
    phase.drop_objection(this);
  endtask
endclass

// CHECK: moore.class.classdecl @connect_env extends @"uvm_pkg::uvm_env"

class connect_test extends uvm_test;
  `uvm_component_utils(connect_test)

  connect_env env;

  function new(string name = "connect_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = connect_env::type_id::create("env", this);
  endfunction
endclass

// CHECK: moore.class.classdecl @connect_test extends @"uvm_pkg::uvm_test"

module uvm_port_connect_semantic_top;
  initial begin
    $display("UVM port connect semantic test");
    run_test("connect_test");
  end
endmodule

// CHECK: moore.module @uvm_port_connect_semantic_top

// SIM: UVM port connect semantic test
// SIM: CONNECT PASS: put/get connected_to count == 1
// SIM: DATA PASS: got expected value 55
// SIM-NOT: UVM_ERROR
// SIM-NOT: UVM_FATAL
