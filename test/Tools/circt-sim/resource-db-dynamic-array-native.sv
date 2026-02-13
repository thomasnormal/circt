// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top resource_db_dyn_array_tb 2>&1 | FileCheck %s

// Regression: resource_db::read_by_name must write through references that
// point into native heap memory (dynamic-array elements allocated via new[]).

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

module resource_db_dyn_array_tb();
  initial begin
    my_cfg src0;
    my_cfg src1;
    my_cfg slots[];
    int int_slots[];
    bit get0;
    bit get1;
    bit int_get0;
    bit int_get1;

    src0 = new("src0");
    src1 = new("src1");
    src0.id = 77;
    src1.id = 88;

    slots = new[2];
    slots[0] = null;
    slots[1] = null;
    int_slots = new[2];
    int_slots[0] = -1;
    int_slots[1] = -1;

    uvm_resource_db#(my_cfg)::set("*", "cfg_0", src0, null);
    uvm_resource_db#(my_cfg)::set("*", "cfg_1", src1, null);
    uvm_resource_db#(int)::set("*", "int_cfg_0", 1234, null);
    uvm_resource_db#(int)::set("*", "int_cfg_1", 5678, null);

    get0 = uvm_resource_db#(my_cfg)::read_by_name("*", "cfg_0", slots[0], null);
    get1 = uvm_resource_db#(my_cfg)::read_by_name("*", "cfg_1", slots[1], null);
    int_get0 = uvm_resource_db#(int)::read_by_name("*", "int_cfg_0", int_slots[0], null);
    int_get1 = uvm_resource_db#(int)::read_by_name("*", "int_cfg_1", int_slots[1], null);

    $display("get0=%0d get1=%0d", get0, get1);
    $display("slot0_null=%0d slot1_null=%0d", slots[0] == null,
             slots[1] == null);
    if (slots[0] != null)
      $display("slot0_id=%0d", slots[0].id);
    if (slots[1] != null)
      $display("slot1_id=%0d", slots[1].id);
    $display("int_get0=%0d int_get1=%0d", int_get0, int_get1);
    $display("int_slot0=%0d int_slot1=%0d", int_slots[0], int_slots[1]);
  end
endmodule
