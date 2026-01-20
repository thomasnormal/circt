// RUN: circt-verilog --ir-moore %s | FileCheck %s
// Test for rand_mode() on objects and properties.

class RandModeTxn;
  rand bit [7:0] addr;
  rand bit [7:0] data;
endclass

class RandModeImplicit;
  rand int value;
  function void disable_value();
    // CHECK: moore.rand_mode
    value.rand_mode(0);
  endfunction
endclass

module test;
  RandModeTxn tx;
  RandModeImplicit implicit_tx;

  initial begin
    tx = new();
    implicit_tx = new();
    // CHECK: moore.rand_mode
    tx.addr.rand_mode(0);
    // CHECK: moore.rand_mode
    tx.rand_mode(1);
    // CHECK: moore.rand_mode
    $display("addr mode = %0d", tx.addr.rand_mode());
    implicit_tx.disable_value();
  end
endmodule
