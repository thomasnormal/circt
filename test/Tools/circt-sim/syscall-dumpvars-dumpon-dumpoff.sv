// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $dumpoff/$dumpon VCD control tasks.
// Bug: All dump tasks ($dumpfile, $dumpvars, $dumpoff, $dumpon, $dumpall)
// are silent no-ops — no VCD file is generated.
// IEEE 1800-2017 Section 21.7: These tasks control VCD dump output.
//
// This test at minimum verifies the tasks are accepted and the simulation
// completes, AND that a VCD file is produced.
module top;
  reg [7:0] counter;
  integer vcd_fd;

  initial begin
    $dumpfile("__test_dumpon_off__.vcd");
    $dumpvars(0, top);
    counter = 0;

    // Phase 1: dumping active
    counter = 1;
    #1;
    counter = 2;
    #1;

    // Pause dumping
    $dumpoff;
    counter = 3;
    #1;
    counter = 4;
    #1;

    // Resume dumping
    $dumpon;
    counter = 5;
    #1;

    // Force all values to be dumped
    $dumpall;
    #1;

    // Verify the VCD file exists
    vcd_fd = $fopen("__test_dumpon_off__.vcd", "r");
    // CHECK: vcd_exists=1
    $display("vcd_exists=%0d", vcd_fd != 0);
    if (vcd_fd != 0) $fclose(vcd_fd);

    // CHECK: completed
    $display("completed");
    $finish;
  end
endmodule
