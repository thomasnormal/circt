// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test lvalue streaming with packed types (structs, packed arrays)
// This tests the fix for "lvalue streaming expected IntType" error that
// occurred when using streaming operators on the left-hand side of
// assignments with packed struct or array types.

// CHECK-LABEL: moore.module @LvalueStreamingStruct
module LvalueStreamingStruct;
  typedef struct packed {
    logic [7:0] a;
    logic [7:0] b;
  } my_struct;

  my_struct s;
  logic [15:0] c;

  // CHECK: moore.procedure initial {
  // CHECK:   [[S_CONV:%[0-9]+]] = moore.conversion %s : !moore.ref<struct<{a: l8, b: l8}>> -> !moore.ref<l16>
  // CHECK:   moore.extract_ref [[S_CONV]] from 0
  // CHECK:   moore.extract_ref [[S_CONV]] from 8
  // CHECK:   moore.concat_ref
  initial begin
    c = 16'hABCD;
    // lvalue streaming with struct - should convert struct ref to bit vector ref
    {<< 8 {s}} = c;
  end
endmodule

// CHECK-LABEL: moore.module @LvalueStreamingPackedArray
module LvalueStreamingPackedArray;
  logic [3:0][3:0] arr;  // 16-bit packed array
  logic [15:0] c;

  // CHECK: moore.procedure initial {
  // CHECK:   [[ARR_CONV:%[0-9]+]] = moore.conversion %arr : !moore.ref<array<4 x l4>> -> !moore.ref<l16>
  // CHECK:   moore.extract_ref [[ARR_CONV]] from 0
  initial begin
    c = 16'h1234;
    // lvalue streaming with packed array - should convert packed array ref to bit vector ref
    {<< 4 {arr}} = c;
  end
endmodule

// CHECK-LABEL: moore.module @LvalueStreamingMultipleOperands
module LvalueStreamingMultipleOperands;
  typedef struct packed {
    logic [7:0] x;
    logic [7:0] y;
  } coord_t;

  coord_t p1;
  coord_t p2;
  logic [31:0] data;

  // CHECK: moore.procedure initial {
  // CHECK:   [[P1_CONV:%[0-9]+]] = moore.conversion %p1 : !moore.ref<struct<{x: l8, y: l8}>> -> !moore.ref<l16>
  // CHECK:   [[P2_CONV:%[0-9]+]] = moore.conversion %p2 : !moore.ref<struct<{x: l8, y: l8}>> -> !moore.ref<l16>
  // CHECK:   moore.concat_ref [[P1_CONV]], [[P2_CONV]]
  initial begin
    data = 32'hDEADBEEF;
    // lvalue streaming with multiple struct operands
    {<< 8 {p1, p2}} = data;
  end
endmodule

// CHECK-LABEL: moore.module @LvalueStreamingNoSlice
module LvalueStreamingNoSlice;
  typedef struct packed {
    logic [7:0] a;
    logic [7:0] b;
  } my_struct;

  my_struct s;
  logic [15:0] c;

  // CHECK: moore.procedure initial {
  // CHECK:   [[S_CONV2:%[0-9]+]] = moore.conversion %s : !moore.ref<struct<{a: l8, b: l8}>> -> !moore.ref<l16>
  // CHECK:   [[C_READ:%[0-9]+]] = moore.read %c
  // CHECK:   moore.blocking_assign [[S_CONV2]], [[C_READ]]
  initial begin
    c = 16'hABCD;
    // lvalue streaming without slice size (getSliceSize() == 0)
    // This should work with conversion but without extract_ref operations
    {>> {s}} = c;
  end
endmodule

// CHECK-LABEL: moore.module @LvalueStreamingDynArray
module LvalueStreamingDynArray;
  bit arr[];
  int val;

  // CHECK: moore.procedure initial {
  // CHECK:   [[ARR_VAR:%[0-9]+]] = moore.dyn_array.new
  // CHECK:   moore.blocking_assign %arr, [[ARR_VAR]]
  // CHECK:   [[VAL_READ:%[0-9]+]] = moore.read %val
  // CHECK:   moore.stream_unpack %arr, [[VAL_READ]] right_to_left
  initial begin
    val = 32'hDEADBEEF;
    arr = new[32];
    // lvalue streaming with dynamic array - uses StreamUnpackOp
    { << bit { arr }} = val;
  end
endmodule
