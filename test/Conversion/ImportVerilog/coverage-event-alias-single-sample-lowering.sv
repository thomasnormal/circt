// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include | FileCheck %s
// REQUIRES: slang

module alias_handles;
  logic clk = 0;
  always #5 clk = ~clk;
  logic [3:0] addr = 0;

  covergroup cg @(posedge clk);
    cp_addr: coverpoint addr;
  endgroup

  cg a;
  cg b;

  initial begin
    a = new;
    b = a;
    @(posedge clk);
    addr <= addr + 1;
  end
endmodule

// CHECK-LABEL: hw.module @alias_handles
// CHECK-COUNT-1: llvm.call @__moore_coverpoint_sample
