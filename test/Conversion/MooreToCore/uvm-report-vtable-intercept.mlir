// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s
// Test UVM report method interception via vtable dispatch.
// When UVM report methods (uvm_report_info, uvm_report_warning, etc.) are called
// through vtable dispatch, they should be intercepted and converted to runtime
// function calls (__moore_uvm_report_info, __moore_uvm_report_warning, etc.).

// Class declarations for uvm_report_object hierarchy
moore.class.classdecl @"uvm_pkg::uvm_report_object" {
  moore.class.methoddecl @uvm_report_info -> @"uvm_pkg::uvm_report_object::uvm_report_info" : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  moore.class.methoddecl @uvm_report_warning -> @"uvm_pkg::uvm_report_object::uvm_report_warning" : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  moore.class.methoddecl @uvm_report_error -> @"uvm_pkg::uvm_report_object::uvm_report_error" : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  moore.class.methoddecl @uvm_report_fatal -> @"uvm_pkg::uvm_report_object::uvm_report_fatal" : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
}

// VTable for uvm_report_object with report methods
moore.vtable @"uvm_pkg::uvm_report_object"::@vtable {
  moore.vtable_entry @uvm_report_info -> @"uvm_pkg::uvm_report_object::uvm_report_info"
  moore.vtable_entry @uvm_report_warning -> @"uvm_pkg::uvm_report_object::uvm_report_warning"
  moore.vtable_entry @uvm_report_error -> @"uvm_pkg::uvm_report_object::uvm_report_error"
  moore.vtable_entry @uvm_report_fatal -> @"uvm_pkg::uvm_report_object::uvm_report_fatal"
}

// UVM report method implementations (stubs)
func.func private @"uvm_pkg::uvm_report_object::uvm_report_info"(
    %self: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) {
  return
}

func.func private @"uvm_pkg::uvm_report_object::uvm_report_warning"(
    %self: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) {
  return
}

func.func private @"uvm_pkg::uvm_report_object::uvm_report_error"(
    %self: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) {
  return
}

func.func private @"uvm_pkg::uvm_report_object::uvm_report_fatal"(
    %self: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) {
  return
}

// CHECK-LABEL: func.func @test_uvm_report_info_vtable
// CHECK: llvm.call @__moore_uvm_report_info
func.func @test_uvm_report_info_vtable(
    %obj: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) -> !moore.i8 {
  %result = moore.constant 1 : i8

  // Load uvm_report_info from vtable and call it
  // This should be intercepted and converted to __moore_uvm_report_info runtime call
  %fn = moore.vtable.load_method %obj : @uvm_report_info of <@"uvm_pkg::uvm_report_object"> ->
      (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  func.call_indirect %fn(%obj, %id, %msg, %verbosity, %filename, %line)
      : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()

  return %result : !moore.i8
}

// CHECK-LABEL: func.func @test_uvm_report_warning_vtable
// CHECK: llvm.call @__moore_uvm_report_warning
func.func @test_uvm_report_warning_vtable(
    %obj: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) -> !moore.i8 {
  %result = moore.constant 1 : i8

  %fn = moore.vtable.load_method %obj : @uvm_report_warning of <@"uvm_pkg::uvm_report_object"> ->
      (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  func.call_indirect %fn(%obj, %id, %msg, %verbosity, %filename, %line)
      : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()

  return %result : !moore.i8
}

// CHECK-LABEL: func.func @test_uvm_report_error_vtable
// CHECK: llvm.call @__moore_uvm_report_error
func.func @test_uvm_report_error_vtable(
    %obj: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) -> !moore.i8 {
  %result = moore.constant 1 : i8

  %fn = moore.vtable.load_method %obj : @uvm_report_error of <@"uvm_pkg::uvm_report_object"> ->
      (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  func.call_indirect %fn(%obj, %id, %msg, %verbosity, %filename, %line)
      : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()

  return %result : !moore.i8
}

// CHECK-LABEL: func.func @test_uvm_report_fatal_vtable
// CHECK: llvm.call @__moore_uvm_report_fatal
func.func @test_uvm_report_fatal_vtable(
    %obj: !moore.class<@"uvm_pkg::uvm_report_object">,
    %id: !moore.string,
    %msg: !moore.string,
    %verbosity: !moore.i32,
    %filename: !moore.string,
    %line: !moore.i32) -> !moore.i8 {
  %result = moore.constant 1 : i8

  %fn = moore.vtable.load_method %obj : @uvm_report_fatal of <@"uvm_pkg::uvm_report_object"> ->
      (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()
  func.call_indirect %fn(%obj, %id, %msg, %verbosity, %filename, %line)
      : (!moore.class<@"uvm_pkg::uvm_report_object">, !moore.string, !moore.string, !moore.i32, !moore.string, !moore.i32) -> ()

  return %result : !moore.i8
}
