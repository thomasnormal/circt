// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time 100000000 2>&1 | FileCheck %s

// Regression: boolean-form class constraints on 1-bit rand fields must be
// enforced in plain randomize() calls.
// CHECK: BOOL_NOT_OK
// CHECK: BOOL_TRUE_OK
// CHECK-NOT: BOOL_NOT_FAIL
// CHECK-NOT: BOOL_TRUE_FAIL

module top;
  class c_not;
    rand bit b;
    constraint c { !b; }
  endclass

  class c_true;
    rand bit b;
    constraint c { b; }
  endclass

  initial begin
    c_not n = new;
    c_true t = new;
    int failNot = 0;
    int failTrue = 0;

    repeat (128) begin
      void'(n.randomize());
      if (n.b !== 1'b0)
        failNot++;

      void'(t.randomize());
      if (t.b !== 1'b1)
        failTrue++;
    end

    if (failNot == 0)
      $display("BOOL_NOT_OK");
    else
      $display("BOOL_NOT_FAIL count=%0d", failNot);

    if (failTrue == 0)
      $display("BOOL_TRUE_OK");
    else
      $display("BOOL_TRUE_FAIL count=%0d", failTrue);

    $finish;
  end
endmodule
