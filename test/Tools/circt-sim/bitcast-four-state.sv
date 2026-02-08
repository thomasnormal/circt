// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top bitcast_4s_tb 2>&1 | FileCheck %s

// Test: hw.bitcast between flat {value: iN, unknown: iN} and nested
// per-field {value: iM, unknown: iM} struct types preserves field values.
// Regression test for f986412ae.

// CHECK: val8=0xa
// CHECK: val4=0x5
// CHECK: [circt-sim] Simulation completed
module bitcast_4s_tb();
  // Two packed struct fields: an 8-bit and a 4-bit value
  typedef struct packed {
    logic [7:0] field_a;
    logic [3:0] field_b;
  } my_struct_t;

  my_struct_t s;

  initial begin
    s.field_a = 8'hA;
    s.field_b = 4'h5;
    $display("val8=0x%h", s.field_a);
    $display("val4=0x%h", s.field_b);
  end
endmodule
