// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time 100000000 2>&1 | FileCheck %s

class base;
  virtual function string get_name();
    return "base";
  endfunction
endclass

class derived extends base;
  rand bit [7:0] value;
  virtual function string get_name();
    return "derived";
  endfunction
endclass

module top;
  initial begin
    derived d = new();
    void'(d.randomize());
    if (d.get_name() == "derived")
      $display("PASS: vtable preserved after randomize");
    else
      $display("FAIL: vtable corrupted after randomize, got %s", d.get_name());
    $finish;
  end
endmodule

// CHECK: PASS: vtable preserved after randomize
