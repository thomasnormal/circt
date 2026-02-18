// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test $ftell, $fseek, $rewind — file position manipulation
module top;
  integer fd;
  integer pos;
  integer ch;
  integer ret;

  initial begin
    // Write a known file
    fd = $fopen("ftell_test.dat", "w");
    $fwrite(fd, "ABCDEF");
    $fclose(fd);

    // Re-open for reading
    fd = $fopen("ftell_test.dat", "r");

    // Initial position should be 0
    pos = $ftell(fd);
    // CHECK: ftell_start=0
    $display("ftell_start=%0d", pos);

    // Read 3 characters
    ch = $fgetc(fd);
    ch = $fgetc(fd);
    ch = $fgetc(fd);

    // Position should now be 3
    pos = $ftell(fd);
    // CHECK: ftell_after3=3
    $display("ftell_after3=%0d", pos);

    // $fseek to position 1 (SEEK_SET = 0)
    ret = $fseek(fd, 1, 0);
    // CHECK: fseek_ret=0
    $display("fseek_ret=%0d", ret);

    // Read character at position 1 — should be 'B' = 66
    ch = $fgetc(fd);
    // CHECK: char_at_1=66
    $display("char_at_1=%0d", ch);

    // $rewind — back to start
    $rewind(fd);
    pos = $ftell(fd);
    // CHECK: ftell_rewind=0
    $display("ftell_rewind=%0d", pos);

    // Read first char — should be 'A' = 65
    ch = $fgetc(fd);
    // CHECK: char_at_0=65
    $display("char_at_0=%0d", ch);

    $fclose(fd);
    $finish;
  end
endmodule
