// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $ferror — returns real error status (not hardcoded 0).
// Bug: $ferror was stubbed to always return constant 0.
module top;
  integer fd, errcode;
  reg [8*128-1:0] errmsg;

  initial begin
    // Happy path: valid file, no error → should return 0
    fd = $fopen("ferror_test.dat", "w");
    $fwrite(fd, "test data\n");
    $fclose(fd);

    fd = $fopen("ferror_test.dat", "r");
    errcode = $ferror(fd, errmsg);
    // CHECK: ferror_valid=0
    $display("ferror_valid=%0d", errcode);
    $fclose(fd);

    // Error path: invalid fd (0 = failed open) → should return non-zero
    // This is the critical test — old constant-0 stub would return 0 here.
    fd = $fopen("__nonexistent_ferror_test__.dat", "r");
    errcode = $ferror(fd, errmsg);
    // CHECK: ferror_invalid_nonzero=1
    $display("ferror_invalid_nonzero=%0d", errcode != 0);

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
