// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 | FileCheck %s --check-prefix=IV
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  logic [7:0] v;
  logic b_logic;
  bit b_bit;

  initial begin
    v = 8'b0000_0001;
    b_logic = v[8];
    b_bit = v[8];
    $display("B_LOGIC=%b B_BIT=%b", b_logic, b_bit);
    // CHECK: B_LOGIC=x B_BIT=0
    #1 $finish;
  end
endmodule

// IV: warning: constant index 8 is out of bounds for range [7:0]
