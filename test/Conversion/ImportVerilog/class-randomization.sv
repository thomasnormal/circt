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

// CHECK-LABEL: moore.class.classdecl @simple_rand
// CHECK: moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK: moore.class.propertydecl @count : !moore.i32 rand_mode rand
// CHECK: moore.class.propertydecl @addr : !moore.i4 rand_mode randc

module test;
  initial begin
    simple_rand obj = new();
    int success;
    success = obj.randomize();
    $display("data=%0d count=%0d addr=%0d success=%0d", obj.data, obj.count, obj.addr, success);
  end
endmodule

// CHECK-LABEL: moore.module @test
// CHECK: moore.procedure initial
// CHECK: moore.class.new : <@simple_rand>
// CHECK: moore.randomize {{%.+}} : <@simple_rand>
