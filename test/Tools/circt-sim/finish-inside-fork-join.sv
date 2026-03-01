// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #48: $finish in a fork branch must lower successfully.

module tb;
  initial begin
    fork
      begin
        #5;
        $display("thread done");
        $finish;
      end
      begin
        #20;
        $display("FAIL should not reach here");
      end
    join
  end

  // CHECK: thread done
  // CHECK-NOT: FAIL should not reach here
endmodule
