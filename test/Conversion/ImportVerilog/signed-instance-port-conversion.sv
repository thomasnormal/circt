// RUN: circt-verilog %s --ir-moore | FileCheck %s

module child_signed_out(output logic signed [3:0] out);
  assign out = -1;
endmodule

module child_unsigned_out(output logic [3:0] out);
  assign out = 4'hf;
endmodule

module top;
  logic signed [7:0] s8;
  logic [7:0] u8;

  child_signed_out us(.out(s8));
  child_unsigned_out uu(.out(u8));

  // CHECK-LABEL: moore.module @top
  // CHECK: %[[SOUT:.*]] = moore.instance "us" @child_signed_out() -> (out: !moore.l4)
  // CHECK: %[[SEXT:.*]] = moore.sext %[[SOUT]] : l4 -> l8
  // CHECK: %[[UOUT:.*]] = moore.instance "uu" @child_unsigned_out() -> (out: !moore.l4)
  // CHECK: %[[ZEXT:.*]] = moore.zext %[[UOUT]] : l4 -> l8
endmodule
