// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=1000000 2>&1 | FileCheck %s

module top;
  typedef union tagged {
    void Invalid;
    int Valid;
  } u_int;

  u_int a;
  int c;

  initial begin
    a = tagged Invalid;
    c = a.Valid;
  end
endmodule

// CHECK: Fatal: Invalid tagged union member access: Valid
// CHECK: [circt-sim] Simulation finished with exit code 1
