// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: drives through llhd.sig.array_get on a memory-backed local
// unpacked array must update the alloca-backed storage.

module top;
  initial begin
    int marker [0:1];
    marker[0] = 1;
    marker[1] = 2;
    $display("M0=%0d M1=%0d", marker[0], marker[1]);
    // CHECK: M0=1 M1=2
    #1 $finish;
  end
endmodule
