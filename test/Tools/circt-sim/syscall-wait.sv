// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test wait statement
module top;
  reg ready = 0;

  initial begin
    #10;
    ready = 1;
  end

  initial begin
    wait(ready == 1);
    // CHECK: wait_done
    $display("wait_done");
    $finish;
  end
endmodule
