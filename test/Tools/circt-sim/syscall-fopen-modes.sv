// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $fopen with different modes, $fclose, and verify data integrity
module top;
  integer fd;
  integer ch;

  initial begin
    // Write mode
    fd = $fopen("fopen_modes_test.dat", "w");
    // $fopen should return non-zero on success
    // CHECK: fopen_w_ok=1
    $display("fopen_w_ok=%0d", fd != 0);
    $fwrite(fd, "HELLO");
    $fclose(fd);

    // Read mode — verify contents
    fd = $fopen("fopen_modes_test.dat", "r");
    // CHECK: fopen_r_ok=1
    $display("fopen_r_ok=%0d", fd != 0);
    ch = $fgetc(fd);
    // 'H' = 72
    // CHECK: read_H=72
    $display("read_H=%0d", ch);
    ch = $fgetc(fd);
    // 'E' = 69
    // CHECK: read_E=69
    $display("read_E=%0d", ch);
    $fclose(fd);

    // Append mode
    fd = $fopen("fopen_modes_test.dat", "a");
    // CHECK: fopen_a_ok=1
    $display("fopen_a_ok=%0d", fd != 0);
    $fwrite(fd, "XY");
    $fclose(fd);

    // Read again — should see HELLOXY
    fd = $fopen("fopen_modes_test.dat", "r");
    // Skip 5 chars (HELLO)
    ch = $fgetc(fd);
    ch = $fgetc(fd);
    ch = $fgetc(fd);
    ch = $fgetc(fd);
    ch = $fgetc(fd);
    // Now read appended chars
    ch = $fgetc(fd);
    // 'X' = 88
    // CHECK: read_X=88
    $display("read_X=%0d", ch);
    ch = $fgetc(fd);
    // 'Y' = 89
    // CHECK: read_Y=89
    $display("read_Y=%0d", ch);
    $fclose(fd);

    // Open invalid file — should return 0
    fd = $fopen("/nonexistent/path/file.txt", "r");
    // CHECK: fopen_invalid=0
    $display("fopen_invalid=%0d", fd);

    $finish;
  end
endmodule
