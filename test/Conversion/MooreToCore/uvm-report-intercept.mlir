// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s
// UVM report function interception for 5-arg signature (minimal UVM stubs).

// Test that UVM report function calls are intercepted and converted to runtime function calls.
// The test includes a Moore constant to trigger conversion context.

module {
  // Declare the UVM report functions (as they would appear from slang/ImportVerilog)
  // These match the signature: (id_struct, msg_struct, verbosity, filename_struct, line)
  // where *_struct is !llvm.struct<(ptr, i64)>
  func.func private @"uvm_pkg::uvm_report_error"(%id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_warning"(%id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_info"(%id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_fatal"(%id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  // Test free function interception (5 args)
  // The function has a moore.constant to force conversion to run
  func.func @test_free_functions() -> !moore.i8 {
    %null_ptr = llvm.mlir.zero : !llvm.ptr
    %zero_i64 = hw.constant 0 : i64
    %verbosity = hw.constant 200 : i32
    %line = hw.constant 42 : i32

    // Add a moore constant to trigger conversion
    %moore_val = moore.constant 42 : i8

    // Create a string struct with null ptr and zero length
    %str_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %str0 = llvm.insertvalue %null_ptr, %str_undef[0] : !llvm.struct<(ptr, i64)>
    %str = llvm.insertvalue %zero_i64, %str0[1] : !llvm.struct<(ptr, i64)>

    // CHECK: llvm.call @__moore_uvm_report_error
    func.call @"uvm_pkg::uvm_report_error"(%str, %str, %verbosity, %str, %line)
      : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_warning
    func.call @"uvm_pkg::uvm_report_warning"(%str, %str, %verbosity, %str, %line)
      : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_info
    func.call @"uvm_pkg::uvm_report_info"(%str, %str, %verbosity, %str, %line)
      : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_fatal
    func.call @"uvm_pkg::uvm_report_fatal"(%str, %str, %verbosity, %str, %line)
      : (!llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    return %moore_val : !moore.i8
  }

  // Declare the class method versions (6 args with self pointer)
  func.func private @"uvm_pkg::uvm_report_object::uvm_report_error"(%self: !llvm.ptr, %id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_object::uvm_report_warning"(%self: !llvm.ptr, %id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_object::uvm_report_info"(%self: !llvm.ptr, %id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  func.func private @"uvm_pkg::uvm_report_object::uvm_report_fatal"(%self: !llvm.ptr, %id: !llvm.struct<(ptr, i64)>, %msg: !llvm.struct<(ptr, i64)>, %verbosity: i32, %filename: !llvm.struct<(ptr, i64)>, %line: i32) {
    return
  }

  // Test class method interception (6 args with self)
  func.func @test_class_methods() -> !moore.i8 {
    %null_ptr = llvm.mlir.zero : !llvm.ptr
    %zero_i64 = hw.constant 0 : i64
    %verbosity = hw.constant 200 : i32
    %line = hw.constant 42 : i32

    // Add a moore constant to trigger conversion
    %moore_val = moore.constant 42 : i8

    // Create a string struct with null ptr and zero length
    %str_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %str0 = llvm.insertvalue %null_ptr, %str_undef[0] : !llvm.struct<(ptr, i64)>
    %str = llvm.insertvalue %zero_i64, %str0[1] : !llvm.struct<(ptr, i64)>

    // CHECK: llvm.call @__moore_uvm_report_error
    func.call @"uvm_pkg::uvm_report_object::uvm_report_error"(%null_ptr, %str, %str, %verbosity, %str, %line)
      : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_warning
    func.call @"uvm_pkg::uvm_report_object::uvm_report_warning"(%null_ptr, %str, %str, %verbosity, %str, %line)
      : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_info
    func.call @"uvm_pkg::uvm_report_object::uvm_report_info"(%null_ptr, %str, %str, %verbosity, %str, %line)
      : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    // CHECK: llvm.call @__moore_uvm_report_fatal
    func.call @"uvm_pkg::uvm_report_object::uvm_report_fatal"(%null_ptr, %str, %str, %verbosity, %str, %line)
      : (!llvm.ptr, !llvm.struct<(ptr, i64)>, !llvm.struct<(ptr, i64)>, i32, !llvm.struct<(ptr, i64)>, i32) -> ()

    return %moore_val : !moore.i8
  }
}
