// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=20000000 2>&1 | FileCheck %s
// Test wait(condition) wakes correctly on variable changes.
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
