// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $ungetc — push character back to file stream
module top;
  integer fd, c;

  initial begin
    fd = $fopen("ungetc_test.dat", "w");
    $fwrite(fd, "AB");
    $fclose(fd);

    fd = $fopen("ungetc_test.dat", "r");
    c = $fgetc(fd);
    // CHECK: first=65
    $display("first=%0d", c);  // 'A' = 65

    // Push 'A' back
    $ungetc(c, fd);

    // Read again — should get 'A' again
    c = $fgetc(fd);
    // CHECK: after_ungetc=65
    $display("after_ungetc=%0d", c);

    $fclose(fd);
    $finish;
  end
endmodule
