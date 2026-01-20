// RUN: circt-verilog --parse-only --allow-nonprocedural-dynamic %s

module top;
  class C;
    int foo;
  endclass

  C c = new;
  wire [31:0] w;
  assign w = c.foo;
endmodule
