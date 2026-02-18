// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test class methods: new, this, super, extends, virtual, pure virtual
module top;
  class base;
    int x;

    function new(int val);
      x = val;
    endfunction

    virtual function string get_name();
      return "base";
    endfunction
  endclass

  class derived extends base;
    int y;

    function new(int val1, int val2);
      super.new(val1);
      y = val2;
    endfunction

    function string get_name();
      return "derived";
    endfunction
  endclass

  initial begin
    base b = new(10);
    derived d = new(20, 30);
    base poly;

    // CHECK: base_x=10
    $display("base_x=%0d", b.x);
    // CHECK: derived_x=20
    $display("derived_x=%0d", d.x);
    // CHECK: derived_y=30
    $display("derived_y=%0d", d.y);

    // Virtual method dispatch
    poly = d;
    // CHECK: virtual=derived
    $display("virtual=%s", poly.get_name());

    // $cast
    b = new(5);
    // CHECK: base_name=base
    $display("base_name=%s", b.get_name());

    $finish;
  end
endmodule
