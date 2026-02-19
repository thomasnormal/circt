// RUN: not circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s
// Test that synchronous PLD array tasks (§21.7) produce a clear compile-time error.
// These are deprecated Verilog-1364 legacy functions not supported by circt-sim.
module top;
  reg [0:3] mem [0:3];
  reg [0:3] addr;
  reg [0:3] result;

  initial begin
    mem[0] = 4'b1100;
    mem[1] = 4'b1010;
    addr = 4'b0110;

    // CHECK: error: unsupported legacy PLD array task '$sync$and$array'
    $sync$and$array(mem, addr, result);
  end
endmodule
