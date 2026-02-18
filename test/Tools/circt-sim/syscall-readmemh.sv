// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg [7:0] mem [0:3];
  integer i, fd;

  initial begin
    // Write a hex memory file
    fd = $fopen("readmemh_test.dat", "w");
    $fwrite(fd, "DE\n");
    $fwrite(fd, "AD\n");
    $fwrite(fd, "BE\n");
    $fwrite(fd, "EF\n");
    $fclose(fd);

    // Read with $readmemh
    $readmemh("readmemh_test.dat", mem);

    // CHECK: mem[0]=de
    // CHECK: mem[1]=ad
    // CHECK: mem[2]=be
    // CHECK: mem[3]=ef
    for (i = 0; i < 4; i = i + 1)
      $display("mem[%0d]=%h", i, mem[i]);

    $finish;
  end
endmodule
