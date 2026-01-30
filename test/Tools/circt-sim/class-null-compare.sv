// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000 2>&1 | FileCheck %s

class MyClass;
  int value;
  static MyClass s_inst;
  
  function new();
    value = 42;
    s_inst = this;
  endfunction
endclass

MyClass global_ref;

module top;
  initial begin
    MyClass local_ref, other_ref;

    // Test uninitialized handles are null
    // CHECK: local_ref is null: YES
    $display("local_ref is null: %s", local_ref == null ? "YES" : "NO");
    // CHECK: global_ref is null: YES
    $display("global_ref is null: %s", global_ref == null ? "YES" : "NO");
    // CHECK: s_inst is null: YES
    $display("s_inst is null: %s", MyClass::s_inst == null ? "YES" : "NO");

    // Create object
    local_ref = new();

    // Test after creation
    // CHECK: local_ref is null after new: NO
    $display("local_ref is null after new: %s", local_ref == null ? "YES" : "NO");
    // CHECK: s_inst is null after new: NO
    $display("s_inst is null after new: %s", MyClass::s_inst == null ? "YES" : "NO");

    // Test assignment
    global_ref = local_ref;
    // CHECK: global_ref is null after assign: NO
    $display("global_ref is null after assign: %s", global_ref == null ? "YES" : "NO");

    // Test equality
    // CHECK: local_ref == s_inst: YES
    $display("local_ref == s_inst: %s", local_ref == MyClass::s_inst ? "YES" : "NO");
    // CHECK: global_ref == local_ref: YES
    $display("global_ref == local_ref: %s", global_ref == local_ref ? "YES" : "NO");

    // CHECK: TEST PASSED
    $display("TEST PASSED");
    $finish;
  end
endmodule
