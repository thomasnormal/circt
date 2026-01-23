// RUN: circt-verilog --parse-only %s
// RUN: circt-verilog --parse-only --allow-nonprocedural-dynamic=false %s 2>&1 | FileCheck %s
// CHECK: error: cannot refer to an element or member of a dynamic type outside of a procedural context

module top;
  class C;
    int foo;
  endclass

  C c = new;
  wire [31:0] w;
  assign w = c.foo;
endmodule
