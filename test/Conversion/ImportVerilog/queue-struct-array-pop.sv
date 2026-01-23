// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

// Test queue pop operations with struct types containing arrays.
// This tests the fix for large hw::StructType elements in queue pop operations.
// The fix ensures that struct types are properly converted to LLVM types
// when used in llvm.alloca and llvm.load operations.

// Define a struct with array members (similar to AXI4 transfer structs)
typedef struct {
  bit [3:0] id;
  bit [31:0] addr;
  bit [7:0] len;
  bit [31:0] data[16];  // Array member
  bit [1:0] resp[16];   // Another array member
  int count;
} transfer_t;

module test_queue_struct_pop;
  // Queue of structs with array members
  transfer_t tx_queue[$];
  transfer_t tx;

  initial begin
    // Test pop_back with struct containing arrays
    // The alloca should use LLVM struct type with array members
    // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i4, i32, i8, array<16 x i32>, array<16 x i2>, i32)>
    // CHECK: llvm.call @__moore_queue_pop_back_ptr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    tx = tx_queue.pop_back();

    // Test pop_front with struct containing arrays
    // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i4, i32, i8, array<16 x i32>, array<16 x i2>, i32)>
    // CHECK: llvm.call @__moore_queue_pop_front_ptr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    // The load should also use LLVM struct type
    // CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i4, i32, i8, array<16 x i32>, array<16 x i2>, i32)>
    tx = tx_queue.pop_front();
  end
endmodule

// Test with larger array sizes (stress test for type conversion)
typedef struct {
  bit [3:0] arid;
  bit [31:0] araddr;
  bit [7:0] arlen;
  bit [2:0] arsize;
  bit [1:0] arburst;
  bit [3:0] rid;
  // Large arrays similar to AXI4 AVIP's axi4_read_transfer_char_s
  bit [31:0] rdata[64];
  bit [1:0] rresp[64];
  bit [3:0] ruser[64];
  int no_of_wait_states;
} large_transfer_t;

module test_queue_large_struct_pop;
  large_transfer_t response_queue[$];
  large_transfer_t response;

  initial begin
    // Pop operations with large struct types must use LLVM types
    // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i4, i32, i8, i3, i2, i4, array<64 x i32>, array<64 x i2>, array<64 x i4>, i32)>
    // CHECK: llvm.call @__moore_queue_pop_back_ptr
    response = response_queue.pop_back();

    // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i4, i32, i8, i3, i2, i4, array<64 x i32>, array<64 x i2>, array<64 x i4>, i32)>
    // CHECK: llvm.call @__moore_queue_pop_front_ptr
    // CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i4, i32, i8, i3, i2, i4, array<64 x i32>, array<64 x i2>, array<64 x i4>, i32)>
    response = response_queue.pop_front();
  end
endmodule
