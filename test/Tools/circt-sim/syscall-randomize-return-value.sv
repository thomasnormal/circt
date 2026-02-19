// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that randomize() returns 1 on success AND actually changes the field.
// Bug: randomize() returns success but doesn't actually randomize the fields —
// they stay at their initial value.
// IEEE 1800-2017 Section 18.7: randomize() returns 1 on success.
class packet;
  rand bit [31:0] data;
  rand bit [7:0] addr;

  function new();
    data = 0;
    addr = 0;
  endfunction
endclass

module top;
  initial begin
    packet p = new();
    int ret;
    int data_changed = 0;
    int addr_changed = 0;
    int i;

    for (i = 0; i < 20; i++) begin
      ret = p.randomize();
      if (p.data != 0) data_changed = 1;
      if (p.addr != 0) addr_changed = 1;
    end

    // randomize() should return 1 (success)
    // CHECK: randomize_ret=1
    $display("randomize_ret=%0d", ret);

    // data should have changed from 0 at least once in 20 tries
    // CHECK: data_randomized=1
    $display("data_randomized=%0d", data_changed);

    // addr should have changed from 0 at least once in 20 tries
    // CHECK: addr_randomized=1
    $display("addr_randomized=%0d", addr_changed);

    $finish;
  end
endmodule
