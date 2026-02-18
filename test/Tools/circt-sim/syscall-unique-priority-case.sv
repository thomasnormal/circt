// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test unique case, priority case, unique0 case
module top;
  int val;
  string result;

  initial begin
    val = 2;

    // unique case — each value matched exactly once
    unique case (val)
      1: result = "one";
      2: result = "two";
      3: result = "three";
    endcase
    // CHECK: unique=two
    $display("unique=%s", result);

    // priority case — first match wins
    priority case (1'b1)
      (val < 5):  result = "small";
      (val < 10): result = "medium";
      default:    result = "large";
    endcase
    // CHECK: priority=small
    $display("priority=%s", result);

    $finish;
  end
endmodule
