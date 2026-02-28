// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Regression: dynamic bit-select assignments with out-of-bounds indices must be
// no-ops. Historically, MooreToCore narrowed/clamped the index and aliased an
// in-range bit.

module top;
  logic [7:0] v;
  int idx;
  initial begin
    idx = 8;
    v = 8'h00;
    v[idx] = 1'b1;
    $display("V=%0d B0=%0d B7=%0d", v, v[0], v[7]);
    // CHECK: V=0 B0=0 B7=0
    #1 $finish;
  end
endmodule
