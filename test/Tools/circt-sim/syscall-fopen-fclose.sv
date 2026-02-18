// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $fopen and $fclose with various modes
module top;
  integer fd;

  initial begin
    // Open for writing
    fd = $fopen("fopen_test.dat", "w");
    // CHECK: fopen_w_ok=1
    $display("fopen_w_ok=%0d", fd != 0);
    $fwrite(fd, "hello world\n");
    $fclose(fd);

    // Open for reading
    fd = $fopen("fopen_test.dat", "r");
    // CHECK: fopen_r_ok=1
    $display("fopen_r_ok=%0d", fd != 0);
    $fclose(fd);

    // Open for append
    fd = $fopen("fopen_test.dat", "a");
    // CHECK: fopen_a_ok=1
    $display("fopen_a_ok=%0d", fd != 0);
    $fclose(fd);

    // Open non-existent file for reading (should fail)
    fd = $fopen("nonexistent_file_12345.dat", "r");
    // CHECK: fopen_fail=1
    $display("fopen_fail=%0d", fd == 0);

    $finish;
  end
endmodule
