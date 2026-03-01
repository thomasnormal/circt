// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir
// Regression for issue #65: streaming operators over fixed unpacked arrays.
module tb;
  logic [7:0] bytes[4];
  logic [31:0] word;

  initial begin
    bytes[0] = 8'hAA;
    bytes[1] = 8'hBB;
    bytes[2] = 8'hCC;
    bytes[3] = 8'hDD;

    word = {>> 8 {bytes}};
    $finish;
  end
endmodule
