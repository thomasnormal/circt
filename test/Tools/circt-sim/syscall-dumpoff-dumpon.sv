// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $dumpoff/$dumpon VCD behavior:
// - Tasks are accepted and processed (not silently dropped)
// - Simulation continues correctly through dump pause/resume
module top;
  reg [7:0] sig;

  initial begin
    $dumpfile("dump.vcd");
    $dumpvars;

    sig = 8'hAA;
    #1;
    // CHECK: before_off=170
    $display("before_off=%0d", sig);

    $dumpoff;
    sig = 8'h55;
    #1;
    // CHECK: during_off=85
    $display("during_off=%0d", sig);

    $dumpon;
    sig = 8'hFF;
    #1;
    // CHECK: after_on=255
    $display("after_on=%0d", sig);

    $finish;
  end
endmodule
