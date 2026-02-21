// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_case_property(input logic clk, input logic [1:0] sel,
                         input logic a, b, c);
  // CHECK-LABEL: moore.module @sva_case_property

  // CHECK: moore.eq
  // CHECK: verif.assert
  assert property (@(posedge clk)
    case (sel)
      2'b00: a;
      2'b01: b;
      default: c;
    endcase
  );

  // CHECK: moore.eq
  // CHECK: verif.assert
  assert property (@(posedge clk)
    case (sel)
      2'b10: a;
      2'b11: b;
    endcase
  );
endmodule
