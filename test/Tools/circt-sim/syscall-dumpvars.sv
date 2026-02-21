// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that VCD dump tasks compile and simulate without errors.
// VCD waveform dumping ($dumpfile/$dumpvars/$dumpoff/$dumpon) now accepted.
module top;
  reg [7:0] val;

  initial begin
    $dumpfile("dump_verify_test.vcd");
    $dumpvars(0, top);
    val = 8'hAA;
    // CHECK: [circt-sim] Simulation completed
    $finish;
  end
endmodule
