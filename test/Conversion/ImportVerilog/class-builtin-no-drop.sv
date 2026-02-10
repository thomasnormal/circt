// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// CHECK-NOT: Class builtin functions
// CHECK: moore.class.classdecl @c
// CHECK: moore.call_pre_randomize
// CHECK: moore.randomize
// CHECK: moore.call_post_randomize

class c;
  rand int a;
endclass

module top;
  initial begin
    c obj = new();
    void'(obj.randomize());
  end
endmodule
