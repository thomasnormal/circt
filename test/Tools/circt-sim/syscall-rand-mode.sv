// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test rand_mode() method
module top;
  class packet;
    rand int data;
    rand int addr;
  endclass

  initial begin
    packet p = new();

    // rand_mode should be 1 (enabled) by default
    // CHECK: rand_mode_on=1
    $display("rand_mode_on=%0d", p.data.rand_mode());

    // Disable randomization of data
    p.data.rand_mode(0);
    // CHECK: rand_mode_off=0
    $display("rand_mode_off=%0d", p.data.rand_mode());

    // addr should still be enabled
    // CHECK: addr_still_on=1
    $display("addr_still_on=%0d", p.addr.rand_mode());

    $finish;
  end
endmodule
