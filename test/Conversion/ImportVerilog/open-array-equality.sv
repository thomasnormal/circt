// RUN: circt-verilog --ir-moore %s | FileCheck %s

class C;
  int arr[];

  function bit eq(C other);
    return arr == other.arr;
  endfunction

  function bit ne(C other);
    return arr != other.arr;
  endfunction
endclass

// CHECK-LABEL: func.func private @"C::eq"
// CHECK: moore.constant 0 : i1
// CHECK: return

// CHECK-LABEL: func.func private @"C::ne"
// CHECK: moore.constant 1 : i1
// CHECK: return

module top;
  initial begin
    C a;
    C b;
    bit e = a.eq(b);
    bit n = a.ne(b);
  end
endmodule
