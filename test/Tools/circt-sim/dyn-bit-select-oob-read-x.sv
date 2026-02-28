// RUN: circt-verilog %s --no-uvm-auto-include --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: dynamic OOB packed bit-select reads must produce X for 4-state
// signals and 0 for 2-state signals.

module top;
  logic [7:0] lv;
  bit [7:0] bv;
  int idx;
  logic lbit;
  bit bbit;

  initial begin
    lv = 8'hA5;
    bv = 8'hA5;
    idx = $urandom_range(8, 8);

    lbit = lv[idx];
    bbit = bv[idx];

    $display("IDX=%0d L=%b B=%b", idx, lbit, bbit);
    // CHECK: IDX=8 L=x B=0
    $finish;
  end
endmodule
