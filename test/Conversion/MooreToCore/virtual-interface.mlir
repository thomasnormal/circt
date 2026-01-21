// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

/// Test interface declaration (should be erased after lowering)
moore.interface @my_bus {
  moore.interface.signal @clk : !moore.l1
  moore.interface.signal @data : !moore.l8
  moore.interface.signal @valid : !moore.l1
}

// CHECK-NOT: moore.interface

/// Check that virtual_interface.signal_ref lowers to GEP
///
/// The interface struct uses 4-state representation (value/unknown pairs):
///   !llvm.struct<"interface.my_bus", (struct<(i1, i1)>, struct<(i8, i8)>, struct<(i1, i1)>)>
/// where:
///   index 0 = clk  (4-state i1)
///   index 1 = data (4-state i8)
///   index 2 = valid (4-state i1)

// CHECK-LABEL: func.func @test_vif_signal_ref
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[VIF]][{{.*}}, 1]
// CHECK:   %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref
// CHECK:   return %[[REF]]
func.func @test_vif_signal_ref(%vif: !moore.virtual_interface<@my_bus>) -> !moore.ref<l8> {
  %data_ref = moore.virtual_interface.signal_ref %vif[@data] : !moore.virtual_interface<@my_bus> -> !moore.ref<l8>
  return %data_ref : !moore.ref<l8>
}

/// Test accessing the first signal (clk at index 0)
// CHECK-LABEL: func.func @test_vif_first_signal
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP1:.*]] = llvm.getelementptr %[[VIF]][{{.*}}, 0]
// CHECK:   %[[REF1:.*]] = builtin.unrealized_conversion_cast %[[GEP1]] : !llvm.ptr to !llhd.ref
// CHECK:   return %[[REF1]]
func.func @test_vif_first_signal(%vif: !moore.virtual_interface<@my_bus>) -> !moore.ref<l1> {
  %clk_ref = moore.virtual_interface.signal_ref %vif[@clk] : !moore.virtual_interface<@my_bus> -> !moore.ref<l1>
  return %clk_ref : !moore.ref<l1>
}

/// Test accessing the last signal (valid at index 2)
// CHECK-LABEL: func.func @test_vif_last_signal
// CHECK-SAME: (%[[VIF:.*]]: !llvm.ptr)
// CHECK:   %[[GEP2:.*]] = llvm.getelementptr %[[VIF]][{{.*}}, 2]
// CHECK:   %[[REF2:.*]] = builtin.unrealized_conversion_cast %[[GEP2]] : !llvm.ptr to !llhd.ref
// CHECK:   return %[[REF2]]
func.func @test_vif_last_signal(%vif: !moore.virtual_interface<@my_bus>) -> !moore.ref<l1> {
  %valid_ref = moore.virtual_interface.signal_ref %vif[@valid] : !moore.virtual_interface<@my_bus> -> !moore.ref<l1>
  return %valid_ref : !moore.ref<l1>
}
