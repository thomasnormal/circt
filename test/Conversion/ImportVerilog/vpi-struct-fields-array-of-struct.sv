// RUN: circt-verilog %s --no-uvm-auto-include | FileCheck %s

typedef struct {
  logic a;
  logic [7:0] b[0:2];
} rec_t;

module top(output rec_t s[0:1]);
endmodule

// CHECK: hw.module @top
// CHECK: vpi.array_bounds = {s = {left = 0 : i32, right = 1 : i32}}
// CHECK: vpi.struct_fields = {s = [{name = "a", width = 1 : i32}, {element_width = 8 : i32, is_array = true, left_bound = 0 : i32, name = "b", num_elements = 3 : i32, right_bound = 2 : i32, width = 24 : i32}]}
