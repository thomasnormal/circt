// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression for issue #37: enum $cast must fail for values not in the enum.
module top;
  typedef enum logic [1:0] {A = 0, B = 1, C = 2} abc_t;
  abc_t e;
  int ok;

  initial begin
    e = B;
    ok = $cast(e, 2);
    // CHECK: ok_valid=1 e_valid=2
    $display("ok_valid=%0d e_valid=%0d", ok, e);

    ok = $cast(e, 7);
    // CHECK: ok_invalid=0 e_after_invalid=2
    $display("ok_invalid=%0d e_after_invalid=%0d", ok, e);
    $finish;
  end
endmodule
