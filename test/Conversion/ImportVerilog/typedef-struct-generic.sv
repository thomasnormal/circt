// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test case for typedef struct used as generic class parameter
// This mimics the UVM pattern: typedef struct { ... } uvm_acs_name_struct;
// used with uvm_queue#(uvm_acs_name_struct)

// CHECK: moore.class.classdecl @container
typedef struct { string name; string regex; } my_struct;

class container#(type T=int);
  // The key test: T should resolve to the struct type, not stay as TypeAlias
  // CHECK: moore.class.propertydecl @data : !moore.ustruct<{name: string, regex: string}>
  T data;

  function void set_data(T item);
    data = item;
  endfunction
endclass

// CHECK: moore.module @test
// Note: Class instance variables without initialization are currently not emitted
module test;
  container#(my_struct) c;
endmodule
