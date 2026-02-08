// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// CHECK: a=42 b=0

typedef struct {
    int a = 42;
    int b;
} my_struct;

module top;
    my_struct s;
    initial begin
        $display("a=%0d b=%0d", s.a, s.b);
        $finish;
    end
endmodule
