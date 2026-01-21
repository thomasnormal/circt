// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: hw.module @Memory(
// CHECK-SAME: in %clock : i1, in %waddr : i4, in %wdata : i42, in %wenable : i1, in %raddr : i4, out rdata : i42
module Memory(
  input  bit clock,
  input  bit [3:0] waddr,
  input  bit [41:0] wdata,
  input  bit wenable,
  input  bit [3:0] raddr,
  output bit [41:0] rdata
);
  // Verify memory storage as LLHD signal with array type
  // CHECK-DAG: %storage = llhd.sig {{%.+}} : !hw.array<16xi42>
  // CHECK-DAG: llhd.sig name "clock" {{%.+}} : i1
  // CHECK-DAG: llhd.sig name "waddr" {{%.+}} : i4
  // CHECK-DAG: llhd.sig name "wdata" {{%.+}} : i42
  // CHECK-DAG: llhd.sig name "wenable" {{%.+}} : i1

  // Verify process-based write logic
  // CHECK: llhd.process
  // CHECK: hw.array_inject {{%.+}}[{{%.+}}], {{%.+}} : !hw.array<16xi42>

  // Verify read logic
  // CHECK: [[STORAGE:%.+]] = llhd.prb %storage : !hw.array<16xi42>
  // CHECK: [[RDATA:%.+]] = hw.array_get [[STORAGE]][%raddr] : !hw.array<16xi42>
  // CHECK: hw.output [[RDATA]]
  bit [41:0] storage [15:0];
  always_ff @(posedge clock)
    if (wenable)
      storage[waddr] <= wdata;
  assign rdata = storage[raddr];
endmodule
