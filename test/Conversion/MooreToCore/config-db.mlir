// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_config_db_set(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i32)
// CHECK-DAG: llvm.func @__moore_config_db_get(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, !llvm.ptr, i64) -> i32

// Define a class for config_db context
moore.class.classdecl @my_component {
  moore.class.propertydecl @name : !moore.string
}

// Define an interface for virtual interface testing
moore.interface @my_interface {
  moore.interface.signal @clk : !moore.l1
  moore.interface.signal @data : !moore.l32
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database Set Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_config_db_set
func.func @test_config_db_set(%value: !moore.i32) {
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.call @__moore_config_db_set
  moore.uvm.config_db.set "*.driver", "my_value", %value : !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database Set with Context
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_config_db_set_with_context
func.func @test_config_db_set_with_context(%ctx: !moore.class<@my_component>, %value: !moore.i32) {
  // CHECK: llvm.call @__moore_config_db_set
  moore.uvm.config_db.set %ctx : !moore.class<@my_component>, "uvm_test_top.env", "config_val", %value : !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database Get Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_config_db_get
func.func @test_config_db_get(%ctx: !moore.class<@my_component>) -> (i1, !moore.i32) {
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_config_db_get
  // CHECK: arith.cmpi ne
  // CHECK: llvm.load %[[ALLOCA]] : !llvm.ptr -> i32
  %found, %value = moore.uvm.config_db.get %ctx : !moore.class<@my_component>, "", "my_value" -> !moore.i32
  return %found, %value : i1, !moore.i32
}

//===----------------------------------------------------------------------===//
// UVM Configuration Database with Virtual Interface Type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_config_db_vif_set
func.func @test_config_db_vif_set(%vif: !moore.virtual_interface<@my_interface>) {
  // CHECK: llvm.call @__moore_config_db_set
  moore.uvm.config_db.set "uvm_test_top.*", "vif", %vif : !moore.virtual_interface<@my_interface>
  return
}

// CHECK-LABEL: func.func @test_config_db_vif_get
func.func @test_config_db_vif_get(%ctx: !moore.class<@my_component>) -> (i1, !moore.virtual_interface<@my_interface>) {
  // CHECK: llvm.call @__moore_config_db_get
  %found, %vif = moore.uvm.config_db.get %ctx : !moore.class<@my_component>, "", "vif" -> !moore.virtual_interface<@my_interface>
  return %found, %vif : i1, !moore.virtual_interface<@my_interface>
}
