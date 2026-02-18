// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  logic [7:0] byte_val;
  int int_val;
  logic [31:0] word;
  logic single;

  typedef struct packed {
    logic [7:0] a;
    logic [15:0] b;
  } my_struct_t;
  my_struct_t s;

  initial begin
    // CHECK: bits_byte=8
    $display("bits_byte=%0d", $bits(byte_val));
    // CHECK: bits_int=32
    $display("bits_int=%0d", $bits(int_val));
    // CHECK: bits_word=32
    $display("bits_word=%0d", $bits(word));
    // CHECK: bits_single=1
    $display("bits_single=%0d", $bits(single));
    // CHECK: bits_struct=24
    $display("bits_struct=%0d", $bits(s));
    $finish;
  end
endmodule
