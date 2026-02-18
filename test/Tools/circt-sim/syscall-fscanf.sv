// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  integer fd, count;
  integer val1, val2;
  reg [8*20-1:0] str1;

  initial begin
    // Write a test file
    fd = $fopen("fscanf_test.dat", "w");
    $fwrite(fd, "42 hello\n");
    $fwrite(fd, "99 world\n");
    $fclose(fd);

    // Read it back with $fscanf
    fd = $fopen("fscanf_test.dat", "r");

    count = $fscanf(fd, "%d %s", val1, str1);
    // CHECK: count=2 val1=42 str1=hello
    $display("count=%0d val1=%0d str1=%s", count, val1, str1);

    count = $fscanf(fd, "%d %s", val2, str1);
    // CHECK: count=2 val2=99 str1=world
    $display("count=%0d val2=%0d str1=%s", count, val2, str1);

    $fclose(fd);
    $finish;
  end
endmodule
