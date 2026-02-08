// RUN: circt-verilog --ir-moore %s | FileCheck %s

typedef struct {
    int a = 42;
    int b;
} my_struct;

module top;
    // CHECK: %[[C42:.*]] = moore.constant 42 : i32
    // CHECK: %[[C0:.*]] = moore.constant 0 : i32
    // CHECK: %[[STRUCT:.*]] = moore.struct_create %[[C42]], %[[C0]] : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>
    // CHECK: %s = moore.variable %[[STRUCT]] : <ustruct<{a: i32, b: i32}>>
    my_struct s;
    initial begin
        $display("a=%0d b=%0d", s.a, s.b);
    end
endmodule
