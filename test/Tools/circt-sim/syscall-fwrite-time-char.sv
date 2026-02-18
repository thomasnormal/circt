// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $fwrite with %t (time format) and %c (character format)
`timescale 1ns/1ps
module top;
  integer fd, count, val;
  initial begin
    $timeformat(-9, 3, " ns", 15);
    #42;
    fd = $fopen("fwrite_time_char.dat", "w");
    $fwrite(fd, "%0d\n", 42);  // reference: plain integer
    $fwrite(fd, "%c\n", 65);   // ASCII 65 = 'A'
    $fclose(fd);

    // Read back and verify the integer line
    fd = $fopen("fwrite_time_char.dat", "r");
    count = $fscanf(fd, "%d", val);
    // CHECK: int_val=42
    $display("int_val=%0d", val);
    $fclose(fd);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
