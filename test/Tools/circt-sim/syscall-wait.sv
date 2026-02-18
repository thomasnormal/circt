// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=50000 2>&1 | FileCheck %s
// Test wait statement — currently hangs because wait() doesn't wake on variable change.
// Using --max-time to prevent infinite hang; test will pass when wait() is implemented.
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
