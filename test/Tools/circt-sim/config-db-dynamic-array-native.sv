// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top config_db_dyn_array_tb 2>&1 | FileCheck %s

// Regression: config_db::get must write through references that point into
// native heap memory (dynamic-array elements allocated via new[]).

// CHECK: get0=1 get1=1
// CHECK: slot0_null=0 slot1_null=0
// CHECK: slot0_id=77
// CHECK: slot1_id=88
// CHECK: [circt-sim] Simulation completed

import uvm_pkg::*;

class my_cfg extends uvm_object;
  int id;

  function new(string name = "my_cfg");
    super.new(name);
  endfunction
endclass

module config_db_dyn_array_tb();
  initial begin
    my_cfg src0;
    my_cfg src1;
    my_cfg slots[];
    bit get0;
    bit get1;

    src0 = new("src0");
    src1 = new("src1");
    src0.id = 77;
    src1.id = 88;

    slots = new[2];
    slots[0] = null;
    slots[1] = null;

    uvm_config_db#(my_cfg)::set(null, "*", "cfg_0", src0);
    uvm_config_db#(my_cfg)::set(null, "*", "cfg_1", src1);

    get0 = uvm_config_db#(my_cfg)::get(null, "", "cfg_0", slots[0]);
    get1 = uvm_config_db#(my_cfg)::get(null, "", "cfg_1", slots[1]);

    $display("get0=%0d get1=%0d", get0, get1);
    $display("slot0_null=%0d slot1_null=%0d", slots[0] == null,
             slots[1] == null);
    if (slots[0] != null)
      $display("slot0_id=%0d", slots[0].id);
    if (slots[1] != null)
      $display("slot1_id=%0d", slots[1].id);
  end
endmodule
