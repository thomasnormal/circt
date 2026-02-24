// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=1000000 2>&1 | FileCheck %s

module top;
  typedef union tagged {
    void Invalid;
    int Valid;
  } u_int;

  u_int a;
  int b;

  initial begin
    a = tagged Valid(42);
    b = a.Valid;
    $display(":assert: (42 == %0d)", b);
  end
endmodule

// CHECK: :assert: (42 ==
// CHECK-NOT: Invalid tagged union member access
