// RUN: circt-verilog %s --ir-hw | FileCheck %s
// REQUIRES: slang

module SliceProducer(output logic [8:0] data_o);
  assign data_o = 9'h12A;
endmodule

module top;
  typedef struct packed {
    logic [27:0] ptag;
    logic gigapage;
    logic megapage;
    logic a;
    logic d;
    logic u;
    logic x;
    logic w;
    logic r;
  } pte_leaf_s;

  pte_leaf_s r_entry;

  SliceProducer producer(
    .data_o(r_entry[0+:9])
  );
endmodule

// CHECK-LABEL: hw.module private @SliceProducer
// CHECK-LABEL: hw.module @top
// CHECK: builtin.unrealized_conversion_cast %r_entry
// CHECK-SAME: to !llhd.ref<!hw.struct<value: i36, unknown: i36>>
// CHECK: llhd.drv %{{.*}}, %{{.*}} after %{{.*}} : !hw.struct<value: i36, unknown: i36>
