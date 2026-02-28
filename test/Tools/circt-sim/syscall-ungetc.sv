// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: $ungetc must push back the last read character.
// Test $ungetc — push character back to file stream
module top;
  integer fd, c, r;

  initial begin
    fd = $fopen("ungetc_test.dat", "w");
    $fwrite(fd, "AB");
    $fclose(fd);

    fd = $fopen("ungetc_test.dat", "r");
    c = $fgetc(fd);
    // CHECK: first=65
    $display("first=%0d", c);  // 'A' = 65

    // Push 'A' back
    r = $ungetc(c, fd);
    // CHECK: ungetc_ret=65
    $display("ungetc_ret=%0d", r);

    // Read again — should get 'A' again
    c = $fgetc(fd);
    // CHECK: after_ungetc=65
    $display("after_ungetc=%0d", c);

    $fclose(fd);
    $finish;
  end
endmodule
