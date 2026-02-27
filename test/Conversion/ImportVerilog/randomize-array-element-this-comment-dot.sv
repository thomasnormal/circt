// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

class Elem;
  rand int mode;
endclass

module top;
  Elem elems[1];

  initial begin
    elems[0] = new;
    if (!elems[0].randomize() with { this/*inline*/.mode == 3; })
      ;
    if (!elems[0].randomize() with { this // line
                                     .mode == 4; })
      ;
  end
endmodule

// CHECK: moore.randomize
// CHECK: moore.class.property_ref
// CHECK-SAME: @mode
// CHECK: moore.constraint.expr
// CHECK: moore.randomize
// CHECK: moore.class.property_ref
// CHECK-SAME: @mode
// CHECK: moore.constraint.expr
