// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: uvm_packer::get_bits/put_bits not available in our UVM library version

// Probe: manually call packer.pack_field/unpack_field bypassing `uvm_field_int.
// Tests the packing engine directly without macro machinery.

// CHECK: [TEST] pack fields: PASS
// CHECK: [TEST] unpack fields: PASS
// CHECK: [TEST] values match: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_pack_item extends uvm_object;
    `uvm_object_utils(probe_pack_item)

    int addr;
    int data;
    bit [7:0] flags;

    function new(string name = "probe_pack_item");
      super.new(name);
    endfunction

    function void do_pack(uvm_packer packer);
      packer.pack_field_int(addr, 32);
      packer.pack_field_int(data, 32);
      packer.pack_field_int(flags, 8);
    endfunction

    function void do_unpack(uvm_packer packer);
      addr = packer.unpack_field_int(32);
      data = packer.unpack_field_int(32);
      flags = packer.unpack_field_int(8);
    endfunction
  endclass

  class probe_pack_test extends uvm_test;
    `uvm_component_utils(probe_pack_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      probe_pack_item src, dst;
      uvm_packer packer;
      bit bitstream[];

      phase.raise_objection(this);

      src = probe_pack_item::type_id::create("src");
      src.addr = 32'hAAAA;
      src.data = 32'h5555;
      src.flags = 8'hFF;

      // Pack
      packer = new();
      src.do_pack(packer);
      packer.get_bits(bitstream);
      if (bitstream.size() > 0)
        `uvm_info("TEST", "pack fields: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "pack fields: FAIL (empty bitstream)")

      // Unpack
      dst = probe_pack_item::type_id::create("dst");
      packer = new();
      packer.put_bits(bitstream);
      dst.do_unpack(packer);
      `uvm_info("TEST", "unpack fields: PASS", UVM_LOW)

      // Verify
      if (dst.addr == 32'hAAAA && dst.data == 32'h5555 && dst.flags == 8'hFF)
        `uvm_info("TEST", "values match: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("values match: FAIL (addr=%0h data=%0h flags=%0h)",
                   dst.addr, dst.data, dst.flags))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_pack_test");
endmodule
