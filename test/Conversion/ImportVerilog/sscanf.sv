// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test $sscanf system function support
// IEEE 1800-2017 Section 21.3.4 "Reading data from a string"

// CHECK-LABEL: moore.module @test_sscanf_basic()
module test_sscanf_basic;
  int x;
  string s;
  int n;
  initial begin
    s = "123";
    // CHECK: %[[INPUT:.+]] = moore.read %s
    // CHECK: %[[RESULT:.+]] = moore.builtin.sscanf %[[INPUT]], "%d", %x : !moore.ref<i32>
    // CHECK: moore.blocking_assign %n, %[[RESULT]] : i32
    n = $sscanf(s, "%d", x);
  end
endmodule

// CHECK-LABEL: moore.module @test_sscanf_multiple_args()
module test_sscanf_multiple_args;
  int a, b;
  string s;
  int n;
  initial begin
    s = "10 20";
    // CHECK: moore.builtin.sscanf {{.+}}, "%d %d", %a, %b : !moore.ref<i32>, !moore.ref<i32>
    n = $sscanf(s, "%d %d", a, b);
  end
endmodule

// CHECK-LABEL: moore.module @test_sscanf_hex()
module test_sscanf_hex;
  int x;
  string s;
  int n;
  initial begin
    s = "ff";
    // CHECK: moore.builtin.sscanf {{.+}}, "%h", %x : !moore.ref<i32>
    n = $sscanf(s, "%h", x);
  end
endmodule

// CHECK-LABEL: moore.module @test_sscanf_skip()
module test_sscanf_skip;
  int x, y;
  string s;
  int n;
  initial begin
    s = "1 skip 2";
    // CHECK: moore.builtin.sscanf {{.+}}, "%d %*s %d", %x, %y : !moore.ref<i32>, !moore.ref<i32>
    n = $sscanf(s, "%d %*s %d", x, y);
  end
endmodule

// Test UVM-style usage pattern: parsing array index from string
// CHECK-LABEL: func.func private @uvm_get_array_index_int
module test_sscanf_uvm_pattern;
  function automatic int uvm_get_array_index_int(string arg);
    int rt_val;
    // CHECK: moore.builtin.sscanf
    rt_val = $sscanf(arg, "%d", uvm_get_array_index_int);
    return uvm_get_array_index_int;
  endfunction

  int result;
  initial begin
    result = uvm_get_array_index_int("42");
  end
endmodule
