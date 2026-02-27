// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

class rose_class;
endclass

module SvaRoseClassHandleSupported(input logic clk);
  rose_class h;
  initial h = new();

  property p_rose_class;
    @(posedge clk) $rose(h);
  endproperty
  assert property (p_rose_class);
endmodule

// CHECK-LABEL: moore.module @SvaRoseClassHandleSupported
// CHECK: moore.class_handle_cmp ne
// CHECK: verif.{{(clocked_)?}}assert
