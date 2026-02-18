// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $typename system function
module top;
  int i;
  logic [7:0] byte_val;
  string s;

  typedef enum {RED, GREEN, BLUE} color_t;
  color_t c;

  initial begin
    // $typename returns string representation of the type
    // CHECK: type_int=int
    $display("type_int=%s", $typename(i));

    // CHECK: type_logic=logic[7:0]
    $display("type_logic=%s", $typename(byte_val));

    // CHECK: type_string=string
    $display("type_string=%s", $typename(s));

    $finish;
  end
endmodule
