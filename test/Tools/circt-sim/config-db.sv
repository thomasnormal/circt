// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top config_db_tb 2>&1 | FileCheck %s

// Test: UVM config_db set/get functionality.
// Verifies that config_db::set stores values and config_db::get retrieves them.
// Also verifies writes through dynamic-array-backed output references.

// CHECK: set_val=42
// CHECK: get_success=1
// CHECK: got_val=42
// CHECK: dyn_get_success=1
// CHECK: dyn_got_val=42
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
    int dyn_got[];
    bit success;
    bit dyn_success;

    val = 42;
    $display("set_val=%0d", val);
    uvm_config_db#(int)::set(null, "*", "my_int", val);

    success = uvm_config_db#(int)::get(null, "", "my_int", got);
    $display("get_success=%0d", success);
    $display("got_val=%0d", got);

    dyn_got = new[2];
    dyn_got[0] = -1;
    dyn_got[1] = -2;
    dyn_success = uvm_config_db#(int)::get(null, "", "my_int", dyn_got[1]);
    $display("dyn_get_success=%0d", dyn_success);
    $display("dyn_got_val=%0d", dyn_got[1]);

    // Try a key that was never set
    success = uvm_config_db#(int)::get(null, "", "no_such_key", got);
    $display("get_miss=%0d", success);
  end
endmodule
