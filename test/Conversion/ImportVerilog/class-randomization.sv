// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Basic class with rand/randc properties
class simple_rand;
  rand bit [7:0] data;
  rand int count;
  randc bit [3:0] addr;

  function new();
    data = 0;
    count = 0;
    addr = 0;
  endfunction
endclass

// CHECK-LABEL: moore.class @simple_rand
// CHECK: moore.class.property @data : !moore.packed<range<i8>>
// CHECK-SAME: rand_mode = rand
// CHECK: moore.class.property @count : !moore.int
// CHECK-SAME: rand_mode = rand
// CHECK: moore.class.property @addr : !moore.packed<range<i4>>
// CHECK-SAME: rand_mode = randc

module test;
  initial begin
    simple_rand obj = new();
    int success;
    success = obj.randomize();
    $display("data=%0d count=%0d addr=%0d success=%0d", obj.data, obj.count, obj.addr, success);
  end
endmodule

// CHECK-LABEL: moore.procedure always
// CHECK: %[[OBJ:.+]] = moore.class.new @simple_rand
// CHECK: %[[SUCCESS:.+]] = moore.randomize %[[OBJ]]
// CHECK: moore.call @display
