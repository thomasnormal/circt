// RUN: not circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s
// Test that VCD dump tasks produce a clear error rather than silently no-opping.
// VCD waveform dumping ($dumpfile/$dumpvars/$dumpoff/$dumpon/$dumpall/$dumpflush)
// requires signal monitoring infrastructure not yet implemented.
module top;
  reg [7:0] val;

  initial begin
    // CHECK: error: unsupported VCD dump task '$dumpfile'
    $dumpfile("dump_verify_test.vcd");
    $dumpvars(0, top);
    val = 8'hAA;
    $finish;
  end
endmodule
