// RUN: circt-verilog --no-uvm-auto-include %s --ir-moore 2>&1 | FileCheck %s
// CHECK-NOT: error:
// CHECK-LABEL: func.func private @"der::do_compare"

class uvm_object;
endclass

class uvm_comparer;
endclass

class base;
  virtual function bit do_compare(uvm_object rhs, uvm_comparer comparer);
    return 1;
  endfunction
endclass

class der extends base;
  function bit do_compare(uvm_object rhs, uvm_comparer comparer = null);
    return super.do_compare(rhs, comparer);
  endfunction
endclass

module top;
  der d;
  uvm_object o;
  uvm_comparer cmp;

  initial begin
    d = new();
    o = new();
    cmp = new();
    void'(d.do_compare(o, cmp));
  end
endmodule
