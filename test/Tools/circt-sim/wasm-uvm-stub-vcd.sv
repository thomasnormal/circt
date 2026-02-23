// Minimal UVM-stub testbench for wasm frontend+sim VCD checks.
`include "uvm_macros.svh"

module wasm_uvm_stub_tb();
  import uvm_pkg::*;

  logic clk = 1'b0;
  logic sig = 1'b0;

  always #5 clk = ~clk;

  initial begin
    $display("uvm stub tb start");
    #7 sig = 1'b1;
    #10 sig = 1'b0;
    #10 $finish;
  end
endmodule
