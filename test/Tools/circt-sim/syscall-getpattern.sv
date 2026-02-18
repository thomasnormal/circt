// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $getpattern — load value from file into register (Verilog-2001)
module top;
  integer fd;
  reg [31:0] pattern;

  initial begin
    // Write a known pattern file
    fd = $fopen("pattern_test.dat", "w");
    $fwrite(fd, "DEADBEEF\n");
    $fclose(fd);

    // $getpattern should load the value
    pattern = $getpattern("pattern_test.dat");
    // CHECK: pattern=deadbeef
    $display("pattern=%h", pattern);
    $finish;
  end
endmodule
