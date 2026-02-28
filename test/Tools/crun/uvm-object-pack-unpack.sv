// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test pack() then unpack() preserves field values.
// Create object with int fields, pack, create new object, unpack, verify fields match.
// NOTE: Uses `uvm_field_int automation macros which are not fully supported.
// No Runtime/uvm/ equivalent exists for pack/unpack. See UVM_COVERAGE.md:
// "Object pack/unpack — Bit ordering issues".

// CHECK: [TEST] pack/unpack int fields: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class packable_obj extends uvm_object;
    `uvm_object_utils_begin(packable_obj)
      `uvm_field_int(a, UVM_ALL_ON)
      `uvm_field_int(b, UVM_ALL_ON)
    `uvm_object_utils_end
    int a;
    int b;
    function new(string name = "packable_obj");
      super.new(name);
    endfunction
  endclass

  class pack_test extends uvm_test;
    `uvm_component_utils(pack_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      packable_obj src, dst;
      bit bitstream[];

      phase.raise_objection(this);

      src = packable_obj::type_id::create("src");
      src.a = 12345;
      src.b = -42;

      src.pack(bitstream);

      dst = packable_obj::type_id::create("dst");
      dst.unpack(bitstream);

      if (dst.a == 12345 && dst.b == -42)
        `uvm_info("TEST", "pack/unpack int fields: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("pack/unpack: FAIL (a=%0d b=%0d)", dst.a, dst.b))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("pack_test");
endmodule
