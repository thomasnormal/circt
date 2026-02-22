// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top hdl_setter --top hvl_getter --max-time 1000000000 2>&1 | FileCheck %s
//
// Regression: static string keys written from hdl_top via config_db::set must
// be readable from hvl_top when get() uses a runtime-built (dynamic) string.
// This matches AVIP BFM registration/lookup patterns.
//
// CHECK: hdl_setter: set_done
// CHECK: hvl_getter: get_success=1
// CHECK: hvl_getter: got_val=2468
// CHECK: [circt-sim] Simulation completed

import uvm_pkg::*;

module hdl_setter();
  initial begin
    uvm_config_db#(int)::set(
        null, "*", "i3c_controller_driver_bfm", 2468);
    $display("hdl_setter: set_done");
  end
endmodule

module hvl_getter();
  initial begin
    string field_name;
    int got;
    bit success;

    field_name = $sformatf("%s", "i3c_controller_driver_bfm");
    success = uvm_config_db#(int)::get(null, "", field_name, got);

    $display("hvl_getter: get_success=%0d", success);
    $display("hvl_getter: got_val=%0d", got);
  end
endmodule
