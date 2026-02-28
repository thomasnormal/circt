// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression companion for Conversion/MooreToCore/assoc-array-exists.mlir:
// verify observable exists() semantics on integer-key associative arrays.

module top;
  int aa[int];

  initial begin
    // CHECK: exists_42_init=0
    $display("exists_42_init=%0d", aa.exists(42));

    aa[7] = 700;
    aa[42] = 4200;

    // CHECK: exists_7_after_set=1
    $display("exists_7_after_set=%0d", aa.exists(7));
    // CHECK: exists_42_after_set=1
    $display("exists_42_after_set=%0d", aa.exists(42));
    // CHECK: exists_99_after_set=0
    $display("exists_99_after_set=%0d", aa.exists(99));

    aa.delete(42);
    // CHECK: exists_42_after_delete=0
    $display("exists_42_after_delete=%0d", aa.exists(42));
    // CHECK: exists_7_after_delete_42=1
    $display("exists_7_after_delete_42=%0d", aa.exists(7));

    aa.delete();
    // CHECK: exists_7_after_delete_all=0
    $display("exists_7_after_delete_all=%0d", aa.exists(7));

    $finish;
  end
endmodule
