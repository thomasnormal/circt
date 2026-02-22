// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_case_property_string(input logic clk, input string sel, input logic a, b);
  // CHECK-LABEL: moore.module @sva_case_property_string

  // CHECK: moore.string_cmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk)
    case (sel)
      "on": a;
      default: b;
    endcase
  );
endmodule
