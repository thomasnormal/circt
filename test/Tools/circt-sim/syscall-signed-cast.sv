// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $signed and $unsigned — sign extension and truncation effects
module top;
  logic [7:0] u8;
  logic signed [7:0] s8;
  logic [15:0] u16;
  int result;

  initial begin
    // $signed on 8'hFF should give -1 when compared as signed
    u8 = 8'hFF;
    s8 = $signed(u8);
    // CHECK: signed_ff=-1
    $display("signed_ff=%0d", s8);

    // $unsigned on -1 (8-bit) should give 255
    s8 = -1;
    u8 = $unsigned(s8);
    // CHECK: unsigned_neg1=255
    $display("unsigned_neg1=%0d", u8);

    // Sign extension: $signed(8'h80) extended to 16-bit should be 0xFF80
    u8 = 8'h80;
    u16 = $signed(u8);
    // CHECK: sign_ext=ff80
    $display("sign_ext=%h", u16);

    // Arithmetic right shift on signed value
    s8 = -4;  // 8'b11111100
    result = s8 >>> 1;  // arithmetic shift: -4 >>> 1 = -2
    // CHECK: arith_shift=-2
    $display("arith_shift=%0d", result);

    // Comparison: signed vs unsigned matters
    u8 = 8'hFF;  // unsigned 255
    // In unsigned comparison: 255 > 1
    // CHECK: unsigned_gt=1
    $display("unsigned_gt=%0d", u8 > 8'd1);

    // As signed: -1 < 1
    // CHECK: signed_lt=1
    $display("signed_lt=%0d", $signed(u8) < $signed(8'd1));

    $finish;
  end
endmodule
