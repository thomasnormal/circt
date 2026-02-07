// RUN: circt-verilog --no-uvm-auto-include --parse-only --top top --max-instance-array=4 %s
// RUN: not circt-verilog --no-uvm-auto-include --parse-only --top top --max-instance-array=2 %s 2>&1 | FileCheck %s

// CHECK: module array exceeded maximum size of 2

module leaf;
endmodule

module top;
  leaf u [0:3] ();
endmodule
