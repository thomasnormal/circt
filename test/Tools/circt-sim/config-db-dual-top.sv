// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top hdl_setter --top hvl_getter --max-time 1000000000 2>&1 | FileCheck %s

// Test: config_db set/get across dual-top modules.
// Verifies that hdl_top's initial block (config_db::set) completes before
// hvl_top's initial block (config_db::get) reads the value.
// This is the pattern used by AVIP BFMs: hdl_top sets BFM handles,
// hvl_top's UVM build_phase retrieves them.

// CHECK: hdl_setter: setting value
// CHECK: hvl_getter: get_success=1
// CHECK: hvl_getter: got_val=99
// CHECK: [circt-sim] Simulation completed

import uvm_pkg::*;

// Module that sets a config_db value (simulates hdl_top BFM registration)
module hdl_setter();
  initial begin
    $display("hdl_setter: setting value");
    uvm_config_db#(int)::set(null, "*", "bfm_handle", 99);
  end
endmodule

// Module that gets the config_db value (simulates hvl_top build_phase)
module hvl_getter();
  initial begin
    int val;
    bit success;
    success = uvm_config_db#(int)::get(null, "", "bfm_handle", val);
    $display("hvl_getter: get_success=%0d", success);
    $display("hvl_getter: got_val=%0d", val);
  end
endmodule
