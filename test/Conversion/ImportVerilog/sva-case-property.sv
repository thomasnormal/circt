// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_case_property(input logic clk, sel, a, b, c);
  // CHECK-LABEL: moore.module @sva_case_property

  // CHECK: comb.icmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk)
    case (sel)
      1'b0: a;
      1'b1: b;
      default: c;
    endcase
  );

  // CHECK: comb.icmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk)
    case (sel)
      1'b0: a;
      1'b1: b;
    endcase
  );
endmodule
