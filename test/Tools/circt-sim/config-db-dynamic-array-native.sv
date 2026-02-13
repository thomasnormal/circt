// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top config_db_dyn_array_tb 2>&1 | FileCheck %s

// Regression: config_db::get must write through references that point into
// native heap memory (dynamic-array elements allocated via new[]).

// CHECK: get0=1 get1=1
// CHECK: slot0_null=0 slot1_null=0
// CHECK: slot0_id=77
// CHECK: slot1_id=88
// CHECK: int_get0=1 int_get1=1
// CHECK: int_slot0=1234 int_slot1=5678
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
    my_cfg src_cfgs[];
    my_cfg slots[];
    int src_ints[];
    int int_slots[];
    bit gets[];
    bit int_gets[];

    src_cfgs = new[2];
    src_cfgs[0] = new("src0");
    src_cfgs[1] = new("src1");
    src_cfgs[0].id = 77;
    src_cfgs[1].id = 88;
    src_ints = new[2];
    src_ints[0] = 1234;
    src_ints[1] = 5678;

    slots = new[2];
    int_slots = new[2];
    gets = new[2];
    int_gets = new[2];

    foreach (slots[i]) begin
      slots[i] = null;
      int_slots[i] = -1;
      uvm_config_db#(my_cfg)::set(null, "*", $sformatf("cfg_%0d", i),
                                  src_cfgs[i]);
      uvm_config_db#(int)::set(null, "*", $sformatf("int_cfg_%0d", i),
                               src_ints[i]);
    end

    foreach (slots[i]) begin
      gets[i] = uvm_config_db#(my_cfg)::get(null, "", $sformatf("cfg_%0d", i),
                                            slots[i]);
      int_gets[i] = uvm_config_db#(int)::get(
          null, "", $sformatf("int_cfg_%0d", i), int_slots[i]);
    end

    $display("get0=%0d get1=%0d", gets[0], gets[1]);
    $display("slot0_null=%0d slot1_null=%0d", slots[0] == null,
             slots[1] == null);
    if (slots[0] != null)
      $display("slot0_id=%0d", slots[0].id);
    if (slots[1] != null)
      $display("slot1_id=%0d", slots[1].id);
    $display("int_get0=%0d int_get1=%0d", int_gets[0], int_gets[1]);
    $display("int_slot0=%0d int_slot1=%0d", int_slots[0], int_slots[1]);
  end
endmodule
