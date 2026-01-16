// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Verify malloc is declared (appears at module top)
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr

//===----------------------------------------------------------------------===//
// Interface declarations with signals and modports
//===----------------------------------------------------------------------===//

/// Test interface declaration with signals and modports (should be erased after lowering)
moore.interface @axi_bus {
  moore.interface.signal @clk : !moore.l1
  moore.interface.signal @rst_n : !moore.l1
  moore.interface.signal @addr : !moore.l32
  moore.interface.signal @data : !moore.l64
  moore.interface.signal @valid : !moore.l1
  moore.interface.signal @ready : !moore.l1
  moore.interface.modport @master (output @addr, output @data, output @valid, input @ready)
  moore.interface.modport @slave (input @addr, input @data, input @valid, output @ready)
}

// Interface declarations should be completely erased
// CHECK-NOT: moore.interface
// CHECK-NOT: moore.interface.signal
// CHECK-NOT: moore.interface.modport

//===----------------------------------------------------------------------===//
// Interface instance allocation
//===----------------------------------------------------------------------===//

/// Test interface instance lowering to malloc
/// The interface struct is: (i1, i1, i32, i64, i1, i1)
/// Size: i1 + i1 + i32 + i64 + i1 + i1 = 1+1+4+8+1+1 = 16 bytes (packed size)
// CHECK-LABEL: func.func @test_interface_instance
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:   %[[PTR:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return
func.func @test_interface_instance() -> !moore.ref<virtual_interface<@axi_bus>> {
  %bus = moore.interface.instance @axi_bus : !moore.ref<virtual_interface<@axi_bus>>
  return %bus : !moore.ref<virtual_interface<@axi_bus>>
}

//===----------------------------------------------------------------------===//
// Virtual interface modport get
//===----------------------------------------------------------------------===//

/// Test virtual_interface.get lowering (should pass through the pointer)
// CHECK-LABEL: func.func @test_vif_modport_get
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   return %[[VIF]] : !llvm.ptr
func.func @test_vif_modport_get(%vif: !moore.virtual_interface<@axi_bus>) -> !moore.virtual_interface<@axi_bus::@master> {
  %master = moore.virtual_interface.get %vif @master : !moore.virtual_interface<@axi_bus> -> !moore.virtual_interface<@axi_bus::@master>
  return %master : !moore.virtual_interface<@axi_bus::@master>
}

/// Test slave modport get
// CHECK-LABEL: func.func @test_vif_slave_modport
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   return %[[VIF]] : !llvm.ptr
func.func @test_vif_slave_modport(%vif: !moore.virtual_interface<@axi_bus>) -> !moore.virtual_interface<@axi_bus::@slave> {
  %slave = moore.virtual_interface.get %vif @slave : !moore.virtual_interface<@axi_bus> -> !moore.virtual_interface<@axi_bus::@slave>
  return %slave : !moore.virtual_interface<@axi_bus::@slave>
}

//===----------------------------------------------------------------------===//
// Signal access through virtual interface
//===----------------------------------------------------------------------===//

/// Test signal access through modport view
/// The interface struct should look like:
///   !llvm.struct<"interface.axi_bus", (i1, i1, i32, i64, i1, i1)>
/// where:
///   index 0 = clk   (i1)
///   index 1 = rst_n (i1)
///   index 2 = addr  (i32)
///   index 3 = data  (i64)
///   index 4 = valid (i1)
///   index 5 = ready (i1)

// CHECK-LABEL: func.func @test_signal_through_modport
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[VIF]][%{{.*}}, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"interface.axi_bus", (i1, i1, i32, i64, i1, i1)>
// CHECK:   %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return %[[REF]] : !llhd.ref<i32>
func.func @test_signal_through_modport(%vif: !moore.virtual_interface<@axi_bus>) -> !moore.ref<l32> {
  %addr_ref = moore.virtual_interface.signal_ref %vif[@addr] : !moore.virtual_interface<@axi_bus> -> !moore.ref<l32>
  return %addr_ref : !moore.ref<l32>
}

/// Test data signal access (64-bit)
// CHECK-LABEL: func.func @test_data_signal
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[VIF]][%{{.*}}, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"interface.axi_bus", (i1, i1, i32, i64, i1, i1)>
// CHECK:   %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref<i64>
// CHECK:   return %[[REF]] : !llhd.ref<i64>
func.func @test_data_signal(%vif: !moore.virtual_interface<@axi_bus>) -> !moore.ref<l64> {
  %data_ref = moore.virtual_interface.signal_ref %vif[@data] : !moore.virtual_interface<@axi_bus> -> !moore.ref<l64>
  return %data_ref : !moore.ref<l64>
}

//===----------------------------------------------------------------------===//
// Simple interface (minimal example)
//===----------------------------------------------------------------------===//

/// Simple interface with just two signals
moore.interface @simple_if {
  moore.interface.signal @a : !moore.l8
  moore.interface.signal @b : !moore.l8
}

// CHECK-NOT: moore.interface @simple_if

/// Test simple interface instance
/// The interface struct size is 2 bytes (2 x i8)
// CHECK-LABEL: func.func @test_simple_instance
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:   %[[PTR:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return
func.func @test_simple_instance() -> !moore.ref<virtual_interface<@simple_if>> {
  %inst = moore.interface.instance @simple_if : !moore.ref<virtual_interface<@simple_if>>
  return %inst : !moore.ref<virtual_interface<@simple_if>>
}
