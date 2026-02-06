// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top config_db_tb 2>&1 | FileCheck %s

// Test: UVM config_db set/get functionality.
// Verifies that config_db::set stores values and config_db::get retrieves them.

// CHECK: set_val=42
// CHECK: get_success=1
// CHECK: got_val=42
// CHECK: get_miss=0
// CHECK: [circt-sim] Simulation completed

import uvm_pkg::*;

class my_config extends uvm_object;
  int my_field;

  function new(string name = "my_config");
    super.new(name);
  endfunction
endclass

module config_db_tb();
  initial begin
    int val;
    int got;
    bit success;

    val = 42;
    $display("set_val=%0d", val);
    uvm_config_db#(int)::set(null, "*", "my_int", val);

    success = uvm_config_db#(int)::get(null, "", "my_int", got);
    $display("get_success=%0d", success);
    $display("got_val=%0d", got);

    // Try a key that was never set
    success = uvm_config_db#(int)::get(null, "", "no_such_key", got);
    $display("get_miss=%0d", success);
  end
endmodule
