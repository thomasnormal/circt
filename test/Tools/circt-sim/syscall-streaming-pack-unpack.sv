// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test streaming operators: {<<{...}} and {>>{...}}
module top;
  logic [7:0] a, a_rev;
  logic [31:0] packed_val;
  logic [7:0] b0, b1, b2, b3;

  initial begin
    a = 8'b10110001;

    // Bit reverse with left-streaming (assign to temp first)
    a_rev = {<<{a}};
    // CHECK: reversed=10001101
    $display("reversed=%b", a_rev);

    // Pack bytes into word
    b0 = 8'hDE; b1 = 8'hAD; b2 = 8'hBE; b3 = 8'hEF;
    packed_val = {>>{b0, b1, b2, b3}};
    // CHECK: packed=deadbeef
    $display("packed=%h", packed_val);

    // Unpack word into bytes
    packed_val = 32'hCAFEBABE;
    {>>{b0, b1, b2, b3}} = packed_val;
    // CHECK: b0=ca
    $display("b0=%h", b0);
    // CHECK: b1=fe
    $display("b1=%h", b1);
    // CHECK: b2=ba
    $display("b2=%h", b2);
    // CHECK: b3=be
    $display("b3=%h", b3);

    $finish;
  end
endmodule
