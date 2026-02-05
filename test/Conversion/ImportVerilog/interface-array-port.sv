// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

interface test_if(input logic clk);
  logic data;
endinterface

module sink(test_if intf);
  initial intf.data = intf.clk;
endmodule

module top(input logic clk);
  test_if ifs[2](clk);
  genvar i;
  generate
    for (i = 0; i < 2; i++) begin : gen
      sink u_sink(ifs[i]);
    end
  endgenerate
endmodule

// CHECK-LABEL: moore.module private @sink
// CHECK-SAME: (in %[[IFACE:.*]] : !moore.ref<virtual_interface<@test_if>>)
// CHECK: moore.virtual_interface.signal_ref
// CHECK-LABEL: moore.module @top
// CHECK-DAG: %ifs_0 = moore.interface.instance @test_if : <virtual_interface<@test_if>>
// CHECK-DAG: %ifs_1 = moore.interface.instance @test_if : <virtual_interface<@test_if>>
// CHECK: moore.instance "{{.*}}u_sink"
