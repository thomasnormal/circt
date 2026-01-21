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

/// Test interface instance lowering to malloc + signal wrapping.
/// The interface struct is: (struct<i1,i1>, struct<i1,i1>, struct<i32,i32>, struct<i64,i64>, struct<i1,i1>, struct<i1,i1>)
/// with 4-state value/unknown pairs for each signal.
/// The interface instance creates a signal holding the malloc'd pointer.
/// This enables virtual interface binding (vif = interface_instance) to work
/// by probing the signal to get the interface pointer.
// CHECK-LABEL: func.func @test_interface_instance
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant
// CHECK:   %[[PTR:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   %[[SIG:.*]] = llhd.sig %[[PTR]] : !llvm.ptr
// CHECK:   return %[[SIG]] : !llhd.ref<!llvm.ptr>
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
/// The interface struct uses 4-state representation (value/unknown pairs):
///   !llvm.struct<"interface.axi_bus", (struct<(i1, i1)>, ...)>
/// Signal index in struct:
///   index 0 = clk
///   index 1 = rst_n
///   index 2 = addr
///   index 3 = data
///   index 4 = valid
///   index 5 = ready

// CHECK-LABEL: func.func @test_signal_through_modport
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[VIF]][{{.*}}, 2]
// CHECK:   %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref
// CHECK:   return %[[REF]]
func.func @test_signal_through_modport(%vif: !moore.virtual_interface<@axi_bus>) -> !moore.ref<l32> {
  %addr_ref = moore.virtual_interface.signal_ref %vif[@addr] : !moore.virtual_interface<@axi_bus> -> !moore.ref<l32>
  return %addr_ref : !moore.ref<l32>
}

/// Test data signal access (64-bit)
// CHECK-LABEL: func.func @test_data_signal
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[VIF]][{{.*}}, 3]
// CHECK:   %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref
// CHECK:   return %[[REF]]
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

/// Test simple interface instance with signal wrapping.
/// The interface struct uses 4-state representation for signals.
// CHECK-LABEL: func.func @test_simple_instance
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant
// CHECK:   %[[PTR:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   %[[SIG:.*]] = llhd.sig %[[PTR]] : !llvm.ptr
// CHECK:   return %[[SIG]] : !llhd.ref<!llvm.ptr>
func.func @test_simple_instance() -> !moore.ref<virtual_interface<@simple_if>> {
  %inst = moore.interface.instance @simple_if : !moore.ref<virtual_interface<@simple_if>>
  return %inst : !moore.ref<virtual_interface<@simple_if>>
}

//===----------------------------------------------------------------------===//
// Ref<virtual_interface> to virtual_interface conversion
//===----------------------------------------------------------------------===//

/// Test conversion from ref<virtual_interface> to virtual_interface.
/// This is a common pattern when assigning an interface instance to a
/// virtual interface variable (vif = intf;).
// CHECK-LABEL: func.func @test_ref_to_vif_conversion
// CHECK-SAME: (%[[REF:.*]]: !llhd.ref<!llvm.ptr>)
// CHECK:   %[[VIF:.*]] = llhd.prb %[[REF]] : !llvm.ptr
// CHECK:   return %[[VIF]] : !llvm.ptr
func.func @test_ref_to_vif_conversion(%ref: !moore.ref<virtual_interface<@simple_if>>) -> !moore.virtual_interface<@simple_if> {
  %vif = moore.conversion %ref : !moore.ref<virtual_interface<@simple_if>> -> !moore.virtual_interface<@simple_if>
  return %vif : !moore.virtual_interface<@simple_if>
}
