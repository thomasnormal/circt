// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Ensure concurrent assertions with simple action blocks preserve diagnostics.
module ActionBlockAssert(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @ActionBlockAssert
  // CHECK: verif.assert {{.*}} label "fail"
  assert property (@(posedge clk) disable iff (rst) a |=> b) else $error("fail");
endmodule

module ActionBlockSeverity(input logic clk, a, b);
  logic shadow;
  logic cond;
  // CHECK-LABEL: moore.module @ActionBlockSeverity
  // CHECK: verif.assert {{.*}} label "fatal_fail"
  assert property (@(posedge clk) a |-> b) else $fatal(1, "fatal_fail");

  // CHECK: verif.assert {{.*}} label "warn_fail"
  assert property (@(posedge clk) b |-> a) else begin
    $warning("warn_fail");
  end

  // CHECK: verif.assert {{.*}} label "disp_fail"
  assert property (@(posedge clk) a |=> b) else $display("disp_fail");

  // CHECK: verif.assert {{.*}} label "multi_stmt_disp_fail"
  assert property (@(posedge clk) b |=> a) else begin
    shadow = a;
    $display("multi_stmt_disp_fail");
  end

  // CHECK: verif.assert {{.*}} label "nested_if_disp_fail"
  assert property (@(posedge clk) a |-> b) else begin
    if (cond)
      $display("nested_if_disp_fail");
  end

  // CHECK: verif.assert {{.*}} label "nested_case_disp_fail"
  assert property (@(posedge clk) b |-> a) else begin
    case (shadow)
      1'b1: $display("nested_case_disp_fail");
      default: ;
    endcase
  end
endmodule
