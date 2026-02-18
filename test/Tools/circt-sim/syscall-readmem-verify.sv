// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test $readmemh and $readmemb — verify loaded values match file contents
module top;
  reg [7:0] memh [0:3];
  reg [7:0] memb [0:3];
  integer fd;

  initial begin
    // Create hex memory file
    fd = $fopen("readmem_verify_h.dat", "w");
    $fwrite(fd, "AA\nBB\nCC\nDD\n");
    $fclose(fd);

    // Create binary memory file
    fd = $fopen("readmem_verify_b.dat", "w");
    $fwrite(fd, "00000001\n00000010\n00000100\n00001000\n");
    $fclose(fd);

    // Load hex
    $readmemh("readmem_verify_h.dat", memh);
    // CHECK: memh0=aa
    $display("memh0=%h", memh[0]);
    // CHECK: memh1=bb
    $display("memh1=%h", memh[1]);
    // CHECK: memh2=cc
    $display("memh2=%h", memh[2]);
    // CHECK: memh3=dd
    $display("memh3=%h", memh[3]);

    // Load binary
    $readmemb("readmem_verify_b.dat", memb);
    // CHECK: memb0=1
    $display("memb0=%0d", memb[0]);
    // CHECK: memb1=2
    $display("memb1=%0d", memb[1]);
    // CHECK: memb2=4
    $display("memb2=%0d", memb[2]);
    // CHECK: memb3=8
    $display("memb3=%0d", memb[3]);

    $finish;
  end
endmodule
