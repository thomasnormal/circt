// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

interface test_if(input logic clk);
endinterface

// CHECK-LABEL: moore.module @top
// CHECK-DAG: %ifs_0 = moore.interface.instance @test_if : <virtual_interface<@test_if>>
// CHECK-DAG: %ifs_1 = moore.interface.instance @test_if : <virtual_interface<@test_if>>
module top(input logic clk);
  test_if ifs[2](clk);
endmodule
