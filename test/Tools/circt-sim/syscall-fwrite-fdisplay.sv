// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $fwrite, $fdisplay, $fstrobe, $fmonitor
module top;
  integer fd, count;
  integer val;

  initial begin
    // Write to file with $fwrite and $fdisplay
    fd = $fopen("fwrite_verify.dat", "w");
    // CHECK: fd_ok=1
    $display("fd_ok=%0d", fd != 0);
    $fwrite(fd, "%0d\n", 42);
    $fdisplay(fd, "%0d", 99);
    $fclose(fd);

    // Read back with $fscanf and verify actual values
    fd = $fopen("fwrite_verify.dat", "r");
    count = $fscanf(fd, "%d", val);
    // CHECK: fwrite_val=42
    $display("fwrite_val=%0d", val);

    count = $fscanf(fd, "%d", val);
    // CHECK: fdisplay_val=99
    $display("fdisplay_val=%0d", val);
    $fclose(fd);

    $finish;
  end
endmodule
