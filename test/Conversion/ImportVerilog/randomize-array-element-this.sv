// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s

class Elem;
  rand int mode;
endclass

class Container;
  Elem elems[2];

  function void foo();
    if (!elems[0].randomize() with { this.mode == 3; })
      ;
  endfunction
endclass

module top;
  Container c;
  initial begin
    c = new;
    c.foo();
  end
endmodule

// CHECK: moore.randomize
// CHECK: moore.class.property_ref
// CHECK-SAME: @mode
// CHECK: moore.eq
// CHECK: moore.constraint.expr
