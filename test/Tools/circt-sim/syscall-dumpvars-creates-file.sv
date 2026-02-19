// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that $dumpfile/$dumpvars produce a VCD file on disk.
// Bug: $dumpfile/$dumpvars are silent no-ops — no file is created.
// IEEE 1800-2017 Section 21.7: $dumpfile opens a VCD file for writing.
//
// This test uses $fopen("r") to verify the VCD file was created.
module top;
  reg [7:0] val;
  integer vcd_fd;

  initial begin
    $dumpfile("__test_dumpvars_verify__.vcd");
    $dumpvars(0, top);

    // Toggle val to generate VCD events
    val = 8'hAA;
    #1;
    val = 8'h55;
    #1;

    // Try to open the VCD file for reading — if it was created, fd != 0
    vcd_fd = $fopen("__test_dumpvars_verify__.vcd", "r");
    // CHECK: vcd_file_exists=1
    $display("vcd_file_exists=%0d", vcd_fd != 0);

    if (vcd_fd != 0) $fclose(vcd_fd);
    $finish;
  end
endmodule
