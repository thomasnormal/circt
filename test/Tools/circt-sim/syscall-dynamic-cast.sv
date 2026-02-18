// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $cast — dynamic type casting between class types
class Base;
  int x;
  function new();
    x = 10;
  endfunction
endclass

class Derived extends Base;
  int y;
  function new();
    super.new();
    y = 20;
  endfunction
endclass

module top;
  Base b;
  Derived d, d2;
  int ok;

  initial begin
    // Create Derived, assign to Base
    d = new();
    b = d;

    // $cast from Base back to Derived should succeed
    ok = $cast(d2, b);
    // CHECK: cast_ok=1
    $display("cast_ok=%0d", ok);

    // Verify the cast object has the right values
    // CHECK: cast_x=10
    $display("cast_x=%0d", d2.x);
    // CHECK: cast_y=20
    $display("cast_y=%0d", d2.y);

    // $cast from a pure Base to Derived should fail
    b = new();
    ok = $cast(d2, b);
    // CHECK: cast_fail=0
    $display("cast_fail=%0d", ok);

    $finish;
  end
endmodule
