// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test $fgetc — character-level file reading
module top;
  integer fd;
  integer ch;

  initial begin
    // Write a test file
    fd = $fopen("fgetc_test2.dat", "w");
    $fwrite(fd, "AB\nCD\n");
    $fclose(fd);

    // Re-open for reading
    fd = $fopen("fgetc_test2.dat", "r");

    // $fgetc reads one character, returns its ASCII value
    ch = $fgetc(fd);
    // 'A' = 65
    // CHECK: fgetc1=65
    $display("fgetc1=%0d", ch);

    ch = $fgetc(fd);
    // 'B' = 66
    // CHECK: fgetc2=66
    $display("fgetc2=%0d", ch);

    ch = $fgetc(fd);
    // newline = 10
    // CHECK: fgetc3=10
    $display("fgetc3=%0d", ch);

    // Read 'C' = 67
    ch = $fgetc(fd);
    // CHECK: fgetc4=67
    $display("fgetc4=%0d", ch);

    // Read 'D' = 68
    ch = $fgetc(fd);
    // CHECK: fgetc5=68
    $display("fgetc5=%0d", ch);

    $fclose(fd);
    $finish;
  end
endmodule
