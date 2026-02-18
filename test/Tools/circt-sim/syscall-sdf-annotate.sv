// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $sdf_annotate — annotate timing from SDF file
// In most simulators this is a compile-time directive that modifies delays.
// We test that it processes the file and doesn't error on valid input.
module top;
  integer fd;
  reg [7:0] out;

  initial begin
    // Create a minimal valid SDF file
    fd = $fopen("test_timing.sdf", "w");
    $fwrite(fd, "(DELAYFILE\n");
    $fwrite(fd, "  (SDFVERSION \"3.0\")\n");
    $fwrite(fd, "  (DESIGN \"top\")\n");
    $fwrite(fd, "  (TIMESCALE 1ns)\n");
    $fwrite(fd, ")\n");
    $fclose(fd);

    // $sdf_annotate should accept and process the file
    // It returns void but should not produce an error message
    $sdf_annotate("test_timing.sdf");

    // Verify sim continues normally after annotation
    out = 8'hAB;
    // CHECK: after_sdf=ab
    $display("after_sdf=%h", out);
    $finish;
  end
endmodule
