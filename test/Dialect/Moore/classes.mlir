// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

module {
// CHECK-LABEL: moore.class.classdecl @Plain {
// CHECK: }
moore.class.classdecl @Plain {
}

// CHECK-LABEL:   moore.class.classdecl @I {
// CHECK:   }
moore.class.classdecl @I {
}

// CHECK-LABEL:   moore.class.classdecl @Base {
// CHECK:   }
// CHECK:   moore.class.classdecl @Derived extends @Base {
// CHECK:   }
moore.class.classdecl @Base {
}
moore.class.classdecl @Derived extends @Base {
}

// CHECK-LABEL:   moore.class.classdecl @IBase {
// CHECK:   }
// CHECK:   moore.class.classdecl @IExt extends @IBase {
// CHECK:   }

moore.class.classdecl @IBase {
}
moore.class.classdecl @IExt extends @IBase {
}

// CHECK-LABEL:   moore.class.classdecl @IU {
// CHECK:   }
// CHECK:   moore.class.classdecl @C1 implements [@IU] {
// CHECK:   }
moore.class.classdecl @IU {
}
moore.class.classdecl @C1 implements [@IU] {
}

// CHECK-LABEL:   moore.class.classdecl @I1 {
// CHECK:   }
// CHECK:   moore.class.classdecl @I2 {
// CHECK:   }
// CHECK:   moore.class.classdecl @C2 implements [@I1, @I2] {
// CHECK:   }
moore.class.classdecl @I1 {
}
moore.class.classdecl @I2 {
}
moore.class.classdecl @C2 implements [@I1, @I2] {
}

// CHECK-LABEL:   moore.class.classdecl @B {
// CHECK:   }
// CHECK:   moore.class.classdecl @J1 {
// CHECK:   }
// CHECK:   moore.class.classdecl @J2 {
// CHECK:   }
// CHECK:   moore.class.classdecl @D extends @B implements [@J1, @J2] {
// CHECK:   }
moore.class.classdecl @B {
}
moore.class.classdecl @J1 {
}
moore.class.classdecl @J2 {
}
moore.class.classdecl @D extends @B implements [@J1, @J2] {
}

// CHECK-LABEL:   moore.class.classdecl @PropertyCombo {
// CHECK-NEXT:     moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:     moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:     moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK:   }
moore.class.classdecl @PropertyCombo {
  moore.class.propertydecl @pubAutoI32 : !moore.i32
  moore.class.propertydecl @protStatL18 : !moore.l18
  moore.class.propertydecl @localAutoI32 : !moore.i32
}

// Test randomization support
// CHECK-LABEL:   moore.class.classdecl @Randomizable {
// CHECK-NEXT:     moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK-NEXT:     moore.class.propertydecl @mode : !moore.i8 rand_mode randc
// CHECK-NEXT:     moore.class.propertydecl @fixed : !moore.i16
// CHECK-NEXT:     moore.constraint.block @valid_range {
// CHECK-NEXT:     }
// CHECK:   }
moore.class.classdecl @Randomizable {
  moore.class.propertydecl @data : !moore.i32 rand_mode rand
  moore.class.propertydecl @mode : !moore.i8 rand_mode randc
  moore.class.propertydecl @fixed : !moore.i16
  moore.constraint.block @valid_range {
  }
}

// Test static and pure constraint blocks
// CHECK-LABEL:   moore.class.classdecl @ConstraintVariants {
// CHECK-NEXT:     moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK-NEXT:     moore.constraint.block @normal {
// CHECK-NEXT:     }
// CHECK-NEXT:     moore.constraint.block static @static_c {
// CHECK-NEXT:     }
// CHECK-NEXT:     moore.constraint.block pure @pure_c {
// CHECK-NEXT:     }
// CHECK:   }
moore.class.classdecl @ConstraintVariants {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @normal {
  }
  moore.constraint.block static @static_c {
  }
  moore.constraint.block pure @pure_c {
  }
}

// Test randomize operation
// CHECK-LABEL: func.func @test_randomize
// CHECK:   %[[OBJ:.*]] = moore.class.new : !moore.class<@Randomizable>
// CHECK:   %[[SUCCESS:.*]] = moore.randomize %[[OBJ]] : !moore.class<@Randomizable>
func.func @test_randomize() {
  %obj = moore.class.new : !moore.class<@Randomizable>
  %success = moore.randomize %obj : !moore.class<@Randomizable>
  return
}

/// Check that vtables roundtrip

// CHECK-LABEL:  moore.vtable @testClass::@vtable {
// CHECK:    moore.vtable @realFunctionClass::@vtable {
// CHECK:      moore.vtable @virtualFunctionClass::@vtable {
// CHECK:        moore.vtable_entry @subroutine -> @"testClass::subroutine"
// CHECK:      }
// CHECK:      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK:    }
// CHECK:    moore.vtable_entry @subroutine -> @"testClass::subroutine"
// CHECK:    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK:  }

  moore.class.classdecl @virtualFunctionClass {
    moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>) -> ()
  }
  moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
    moore.class.methoddecl @testSubroutine : (!moore.class<@realFunctionClass>) -> ()
  }
  moore.class.classdecl @testClass implements [@realFunctionClass] {
    moore.class.methoddecl @subroutine -> @"testClass::subroutine" : (!moore.class<@testClass>) -> ()
    moore.class.methoddecl @testSubroutine -> @"testClass::testSubroutine" : (!moore.class<@testClass>) -> ()
  }
  moore.vtable @testClass::@vtable {
    moore.vtable @realFunctionClass::@vtable {
      moore.vtable @virtualFunctionClass::@vtable {
        moore.vtable_entry @subroutine -> @"testClass::subroutine"
      }
      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
    }
    moore.vtable_entry @subroutine -> @"testClass::subroutine"
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  func.func private @"testClass::subroutine"(%arg0: !moore.class<@testClass>) {
    return
  }
  func.func private @"testClass::testSubroutine"(%arg0: !moore.class<@testClass>) {
    return
  }

// CHECK-LABEL:  moore.vtable @tClass::@vtable {
// CHECK:    moore.vtable @testClass::@vtable {
// CHECK:      moore.vtable @realFunctionClass::@vtable {
// CHECK:        moore.vtable @virtualFunctionClass::@vtable {
// CHECK:          moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK:        }
// CHECK:        moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK:      }
// CHECK:      moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK:      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK:    }
// CHECK:    moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK:  }
  moore.class.classdecl @tClass extends @testClass {
    moore.class.methoddecl @subroutine -> @"tClass::subroutine" : (!moore.class<@tClass>) -> ()
  }
  moore.vtable @tClass::@vtable {
    moore.vtable @testClass::@vtable {
      moore.vtable @realFunctionClass::@vtable {
        moore.vtable @virtualFunctionClass::@vtable {
          moore.vtable_entry @subroutine -> @"tClass::subroutine"
        }
        moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
      }
      moore.vtable_entry @subroutine -> @"tClass::subroutine"
      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
    }
    moore.vtable_entry @subroutine -> @"tClass::subroutine"
  }
  func.func private @"tClass::subroutine"(%arg0: !moore.class<@tClass>) {
    return
  }

//===----------------------------------------------------------------------===//
// Interface and Virtual Interface Tests
//===----------------------------------------------------------------------===//

// Test basic interface declaration with signals
// CHECK-LABEL: moore.interface @my_bus {
// CHECK-NEXT:    moore.interface.signal @clk : !moore.l1
// CHECK-NEXT:    moore.interface.signal @data : !moore.l32
// CHECK-NEXT:    moore.interface.signal @valid : !moore.l1
// CHECK-NEXT:    moore.interface.signal @ready : !moore.l1
// CHECK:       }
moore.interface @my_bus {
  moore.interface.signal @clk : !moore.l1
  moore.interface.signal @data : !moore.l32
  moore.interface.signal @valid : !moore.l1
  moore.interface.signal @ready : !moore.l1
}

// Test interface with modports
// CHECK-LABEL: moore.interface @handshake_if {
// CHECK-NEXT:    moore.interface.signal @clk : !moore.l1
// CHECK-NEXT:    moore.interface.signal @data : !moore.l8
// CHECK-NEXT:    moore.interface.signal @valid : !moore.l1
// CHECK-NEXT:    moore.interface.signal @ready : !moore.l1
// CHECK-NEXT:    moore.interface.modport @driver (output @clk, output @data, output @valid, input @ready)
// CHECK-NEXT:    moore.interface.modport @monitor (input @clk, input @data, input @valid, input @ready)
// CHECK:       }
moore.interface @handshake_if {
  moore.interface.signal @clk : !moore.l1
  moore.interface.signal @data : !moore.l8
  moore.interface.signal @valid : !moore.l1
  moore.interface.signal @ready : !moore.l1
  moore.interface.modport @driver (output @clk, output @data, output @valid, input @ready)
  moore.interface.modport @monitor (input @clk, input @data, input @valid, input @ready)
}

// Test interface instance and virtual interface type
// CHECK-LABEL: moore.module @test_interface_instance
// CHECK:         %[[INST:.*]] = moore.interface.instance @handshake_if : !moore.ref<!moore.virtual_interface<@handshake_if>>
moore.module @test_interface_instance() {
  %bus = moore.interface.instance @handshake_if : !moore.ref<!moore.virtual_interface<@handshake_if>>
  moore.output
}

// Test virtual interface get modport
// CHECK-LABEL: func.func @test_vif_modport
// CHECK-SAME:    (%[[VIF:.*]]: !moore.virtual_interface<@handshake_if>)
// CHECK:         %[[DRIVER:.*]] = moore.virtual_interface.get %[[VIF]] @driver : !moore.virtual_interface<@handshake_if> -> !moore.virtual_interface<@handshake_if::@driver>
func.func @test_vif_modport(%vif: !moore.virtual_interface<@handshake_if>) {
  %driver = moore.virtual_interface.get %vif @driver : !moore.virtual_interface<@handshake_if> -> !moore.virtual_interface<@handshake_if::@driver>
  return
}

// Test virtual interface signal reference
// CHECK-LABEL: func.func @test_vif_signal_ref
// CHECK-SAME:    (%[[VIF:.*]]: !moore.virtual_interface<@handshake_if>)
// CHECK:         %[[DATA_REF:.*]] = moore.virtual_interface.signal_ref %[[VIF]][@data] : !moore.virtual_interface<@handshake_if> -> !moore.ref<!moore.l8>
// CHECK:         %[[DATA:.*]] = moore.read %[[DATA_REF]] : <l8>
func.func @test_vif_signal_ref(%vif: !moore.virtual_interface<@handshake_if>) {
  %data_ref = moore.virtual_interface.signal_ref %vif[@data] : !moore.virtual_interface<@handshake_if> -> !moore.ref<!moore.l8>
  %data = moore.read %data_ref : <l8>
  return
}

// Test virtual interface in class property (UVM driver pattern)
// CHECK-LABEL: moore.class.classdecl @MyDriver {
// CHECK-NEXT:    moore.class.propertydecl @vif : !moore.virtual_interface<@handshake_if::@driver>
// CHECK:       }
moore.class.classdecl @MyDriver {
  moore.class.propertydecl @vif : !moore.virtual_interface<@handshake_if::@driver>
}

// Test interface with inout and ref modport directions
// CHECK-LABEL: moore.interface @bidir_if {
// CHECK-NEXT:    moore.interface.signal @bidir : !moore.l16
// CHECK-NEXT:    moore.interface.modport @port (inout @bidir)
// CHECK:       }
moore.interface @bidir_if {
  moore.interface.signal @bidir : !moore.l16
  moore.interface.modport @port (inout @bidir)
}

}
