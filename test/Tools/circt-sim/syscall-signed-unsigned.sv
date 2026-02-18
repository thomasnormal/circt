// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $signed and $unsigned system functions
module top;
  logic [7:0] u;
  logic signed [7:0] s;

  initial begin
    u = 8'hFF;  // 255 unsigned

    // $signed interprets as signed
    s = $signed(u);
    // CHECK: signed_ff=-1
    $display("signed_ff=%0d", s);

    // $unsigned interprets as unsigned
    s = -1;
    // CHECK: unsigned_neg1=255
    $display("unsigned_neg1=%0d", $unsigned(s));

    u = 8'h7F;
    s = $signed(u);
    // CHECK: signed_7f=127
    $display("signed_7f=%0d", s);

    $finish;
  end
endmodule
