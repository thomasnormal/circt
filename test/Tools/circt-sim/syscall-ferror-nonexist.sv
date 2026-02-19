// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $ferror returns non-zero for invalid fd and zero for valid fd.
// Bug: $ferror is stubbed to always return 0.
module top;
  integer fd, errcode;
  reg [8*128-1:0] errmsg;

  initial begin
    // First: open a valid file, $ferror should return 0
    fd = $fopen("ferror_valid_test.dat", "w");
    $fwrite(fd, "test data\n");
    errcode = $ferror(fd, errmsg);
    // CHECK: ferror_valid=0
    $display("ferror_valid=%0d", errcode);
    $fclose(fd);

    // Second: open a nonexistent file for reading — $fopen returns 0 on failure
    fd = $fopen("__nonexistent_file_ferror_bug_test__.dat", "r");
    // CHECK: fopen_fd=0
    $display("fopen_fd=%0d", fd);

    // $ferror on an invalid fd should return non-zero error code
    errcode = $ferror(fd, errmsg);
    // CHECK: ferror_invalid_nonzero=1
    $display("ferror_invalid_nonzero=%0d", errcode != 0);
    $finish;
  end
endmodule
