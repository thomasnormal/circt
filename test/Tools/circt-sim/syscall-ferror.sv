// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $ferror — returns error status for file descriptor
module top;
  integer fd, errcode;
  reg [8*128-1:0] errmsg;

  initial begin
    fd = $fopen("ferror_test.dat", "w");
    $fwrite(fd, "test data\n");
    $fclose(fd);

    // Reopen for reading — no error expected
    fd = $fopen("ferror_test.dat", "r");
    errcode = $ferror(fd, errmsg);
    // CHECK: ferror_ok=0
    $display("ferror_ok=%0d", errcode);
    $fclose(fd);

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
