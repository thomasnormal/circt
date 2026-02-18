// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $dumpfile, $dumpvars, $dumpoff, $dumpon, $dumpall, $dumplimit, $dumpflush
module top;
  reg [7:0] val;
  integer fd;
  reg [8*80-1:0] line;

  initial begin
    $dumpfile("dump_verify_test.vcd");
    $dumpvars(0, top);

    val = 8'hAA;
    #1;

    $dumpoff;
    val = 8'hBB;
    #1;

    $dumpon;
    val = 8'hCC;
    #1;

    $dumpall;
    $dumpflush;

    // Verify VCD file was actually created with content
    fd = $fopen("dump_verify_test.vcd", "r");
    // CHECK: vcd_opened=1
    $display("vcd_opened=%0d", fd != 0);

    // VCD files start with $date or similar header
    $fgets(line, fd);
    // CHECK: vcd_has_content=1
    $display("vcd_has_content=%0d", line != "");
    $fclose(fd);

    $finish;
  end
endmodule
