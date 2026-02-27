// RUN: not circt-verilog %s --ir-llhd --no-uvm-auto-include 2>&1 | FileCheck %s
// REQUIRES: slang

module top;
  logic clk = 0;
  always #5 clk = ~clk;
  logic [3:0] addr = 0;

  covergroup cg @(posedge clk);
    cp_addr: coverpoint addr;
  endgroup

  function void init_cov();
    static cg cov = new;
  endfunction

  initial begin
    init_cov();
    @(posedge clk);
    addr <= addr + 1;
  end
endmodule

// CHECK: error: implicit covergroup event sampling for local variables is not supported
