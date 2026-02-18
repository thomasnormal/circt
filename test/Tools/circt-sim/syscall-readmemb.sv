// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg [7:0] mem [0:3];
  reg [7:0] mem2 [0:3];
  integer i, fd;

  initial begin
    // Write a binary memory file manually
    fd = $fopen("readmemb_test.dat", "w");
    $fwrite(fd, "10101010\n");
    $fwrite(fd, "11001100\n");
    $fwrite(fd, "11110000\n");
    $fwrite(fd, "00001111\n");
    $fclose(fd);

    // Read it with $readmemb
    $readmemb("readmemb_test.dat", mem);

    // CHECK: mem[0]=aa
    // CHECK: mem[1]=cc
    // CHECK: mem[2]=f0
    // CHECK: mem[3]=0f
    for (i = 0; i < 4; i = i + 1)
      $display("mem[%0d]=%h", i, mem[i]);

    $finish;
  end
endmodule
