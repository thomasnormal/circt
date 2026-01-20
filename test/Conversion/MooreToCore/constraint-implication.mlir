// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Constraint Implication Operator Tests - Iteration 62 Track B
// Tests for ConstraintImplicationOp and ConstraintIfElseOp lowering.
// IEEE 1800-2017 Section 18.5.6 "Implication constraints"
// IEEE 1800-2017 Section 18.5.7 "if-else constraints"
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Basic implication constraint lowering
// Corresponds to: constraint c { mode -> data > 0; }
//===----------------------------------------------------------------------===//

moore.class.classdecl @BasicImplication {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @data : !moore.i32 rand_mode rand
  moore.constraint.block @c_impl {
  ^bb0(%mode: !moore.i1, %data: !moore.i32):
    %c0 = moore.constant 0 : i32
    moore.constraint.implication %mode : i1 {
      %gt = moore.sgt %data, %c0 : i32 -> i1
      moore.constraint.expr %gt : i1
    }
  }
}

// CHECK-LABEL: func.func @test_basic_implication
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.expr
func.func @test_basic_implication(%obj: !moore.class<@BasicImplication>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@BasicImplication>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 2: Multiple implication constraints in one block
//===----------------------------------------------------------------------===//

moore.class.classdecl @MultiImplication {
  moore.class.propertydecl @op_type : !moore.i2 rand_mode rand
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @c_multi {
  ^bb0(%op_type: !moore.i2, %value: !moore.i32):
    %c0 = moore.constant 0 : i2
    %c1 = moore.constant 1 : i2
    %c100 = moore.constant 100 : i32
    %c200 = moore.constant 200 : i32
    %is_zero = moore.eq %op_type, %c0 : i2 -> i1
    %is_one = moore.eq %op_type, %c1 : i2 -> i1
    // First implication: op_type == 0 -> value < 100
    moore.constraint.implication %is_zero : i1 {
      %lt = moore.slt %value, %c100 : i32 -> i1
      moore.constraint.expr %lt : i1
    }
    // Second implication: op_type == 1 -> value > 200
    moore.constraint.implication %is_one : i1 {
      %gt = moore.sgt %value, %c200 : i32 -> i1
      moore.constraint.expr %gt : i1
    }
  }
}

// CHECK-LABEL: func.func @test_multi_implication
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.implication
func.func @test_multi_implication(%obj: !moore.class<@MultiImplication>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@MultiImplication>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 3: Basic if-else constraint lowering
// Corresponds to: constraint c { if (mode) x > 0; else x < 0; }
//===----------------------------------------------------------------------===//

moore.class.classdecl @BasicIfElse {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @c_ifelse {
  ^bb0(%mode: !moore.i1, %x: !moore.i32):
    %c0 = moore.constant 0 : i32
    moore.constraint.if_else %mode : i1 {
      %gt = moore.sgt %x, %c0 : i32 -> i1
      moore.constraint.expr %gt : i1
    } else {
      %lt = moore.slt %x, %c0 : i32 -> i1
      moore.constraint.expr %lt : i1
    }
  }
}

// CHECK-LABEL: func.func @test_basic_ifelse
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.if_else
// CHECK-NOT: moore.constraint.expr
func.func @test_basic_ifelse(%obj: !moore.class<@BasicIfElse>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@BasicIfElse>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 4: if-only constraint (no else branch)
//===----------------------------------------------------------------------===//

moore.class.classdecl @IfOnly {
  moore.class.propertydecl @valid : !moore.i1 rand_mode rand
  moore.class.propertydecl @data : !moore.i32 rand_mode rand
  moore.constraint.block @c_if_only {
  ^bb0(%valid: !moore.i1, %data: !moore.i32):
    %c0 = moore.constant 0 : i32
    moore.constraint.if_else %valid : i1 {
      %ge = moore.sge %data, %c0 : i32 -> i1
      moore.constraint.expr %ge : i1
    }
  }
}

// CHECK-LABEL: func.func @test_if_only
// CHECK-NOT: moore.constraint.if_else
func.func @test_if_only(%obj: !moore.class<@IfOnly>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@IfOnly>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 5: Nested if-else constraints
//===----------------------------------------------------------------------===//

moore.class.classdecl @NestedIfElse {
  moore.class.propertydecl @level : !moore.i2 rand_mode rand
  moore.class.propertydecl @value : !moore.i8 rand_mode rand
  moore.constraint.block @c_nested_ifelse {
  ^bb0(%level: !moore.i2, %value: !moore.i8):
    %c0 = moore.constant 0 : i2
    %c1 = moore.constant 1 : i2
    %c50 = moore.constant 50 : i8
    %c100 = moore.constant 100 : i8
    %c150 = moore.constant 150 : i8
    %is_low = moore.eq %level, %c0 : i2 -> i1
    %is_med = moore.eq %level, %c1 : i2 -> i1
    moore.constraint.if_else %is_low : i1 {
      %lt = moore.ult %value, %c50 : i8 -> i1
      moore.constraint.expr %lt : i1
    } else {
      moore.constraint.if_else %is_med : i1 {
        %lt = moore.ult %value, %c100 : i8 -> i1
        moore.constraint.expr %lt : i1
      } else {
        %lt = moore.ult %value, %c150 : i8 -> i1
        moore.constraint.expr %lt : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_nested_ifelse
// CHECK-NOT: moore.constraint.if_else
func.func @test_nested_ifelse(%obj: !moore.class<@NestedIfElse>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@NestedIfElse>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 6: Soft constraint with implication
// Corresponds to: constraint c { mode -> soft value == 100; }
//===----------------------------------------------------------------------===//

moore.class.classdecl @SoftImplication {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @c_soft_impl {
  ^bb0(%mode: !moore.i1, %value: !moore.i32):
    %c100 = moore.constant 100 : i32
    moore.constraint.implication %mode : i1 {
      %eq = moore.eq %value, %c100 : i32 -> i1
      moore.constraint.expr %eq : i1 soft
    }
  }
}

// CHECK-LABEL: func.func @test_soft_implication
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.expr
func.func @test_soft_implication(%obj: !moore.class<@SoftImplication>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SoftImplication>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 7: Soft constraint with if-else
//===----------------------------------------------------------------------===//

moore.class.classdecl @SoftIfElse {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @c_soft_ifelse {
  ^bb0(%mode: !moore.i1, %value: !moore.i32):
    %c0 = moore.constant 0 : i32
    %c100 = moore.constant 100 : i32
    moore.constraint.if_else %mode : i1 {
      %eq1 = moore.eq %value, %c100 : i32 -> i1
      moore.constraint.expr %eq1 : i1 soft
    } else {
      %eq0 = moore.eq %value, %c0 : i32 -> i1
      moore.constraint.expr %eq0 : i1 soft
    }
  }
}

// CHECK-LABEL: func.func @test_soft_ifelse
// CHECK-NOT: moore.constraint.if_else
// CHECK-NOT: moore.constraint.expr
func.func @test_soft_ifelse(%obj: !moore.class<@SoftIfElse>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SoftIfElse>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 8: Mixed implication and if-else in same block
//===----------------------------------------------------------------------===//

moore.class.classdecl @MixedImplIfElse {
  moore.class.propertydecl @type_sel : !moore.i2 rand_mode rand
  moore.class.propertydecl @len : !moore.i8 rand_mode rand
  moore.class.propertydecl @addr : !moore.i16 rand_mode rand
  moore.constraint.block @c_mixed {
  ^bb0(%type_sel: !moore.i2, %len: !moore.i8, %addr: !moore.i16):
    %c0 = moore.constant 0 : i2
    %c1 = moore.constant 1 : i2
    %c16 = moore.constant 16 : i8
    %h1000 = moore.constant 4096 : i16
    %is_zero = moore.eq %type_sel, %c0 : i2 -> i1
    %is_one = moore.eq %type_sel, %c1 : i2 -> i1
    // Implication constraint
    moore.constraint.implication %is_zero : i1 {
      %lt = moore.ult %len, %c16 : i8 -> i1
      moore.constraint.expr %lt : i1
    }
    // If-else constraint
    moore.constraint.if_else %is_one : i1 {
      %lt_addr = moore.ult %addr, %h1000 : i16 -> i1
      moore.constraint.expr %lt_addr : i1
    } else {
      %ge_addr = moore.uge %addr, %h1000 : i16 -> i1
      moore.constraint.expr %ge_addr : i1
    }
  }
}

// CHECK-LABEL: func.func @test_mixed
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.if_else
func.func @test_mixed(%obj: !moore.class<@MixedImplIfElse>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@MixedImplIfElse>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 9: Chained implications (a -> b, b -> c)
//===----------------------------------------------------------------------===//

moore.class.classdecl @ChainedImpl {
  moore.class.propertydecl @a : !moore.i1 rand_mode rand
  moore.class.propertydecl @b : !moore.i1 rand_mode rand
  moore.class.propertydecl @c : !moore.i1 rand_mode rand
  moore.constraint.block @c_chain {
  ^bb0(%a: !moore.i1, %b: !moore.i1, %c: !moore.i1):
    // a -> b
    moore.constraint.implication %a : i1 {
      moore.constraint.expr %b : i1
    }
    // b -> c
    moore.constraint.implication %b : i1 {
      moore.constraint.expr %c : i1
    }
  }
}

// CHECK-LABEL: func.func @test_chained
// CHECK-NOT: moore.constraint.implication
func.func @test_chained(%obj: !moore.class<@ChainedImpl>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ChainedImpl>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 10: Bidirectional implication (x iff y pattern)
//===----------------------------------------------------------------------===//

moore.class.classdecl @BidirImpl {
  moore.class.propertydecl @x : !moore.i1 rand_mode rand
  moore.class.propertydecl @y : !moore.i1 rand_mode rand
  moore.constraint.block @c_bidir {
  ^bb0(%x: !moore.i1, %y: !moore.i1):
    // x -> y
    moore.constraint.implication %x : i1 {
      moore.constraint.expr %y : i1
    }
    // y -> x
    moore.constraint.implication %y : i1 {
      moore.constraint.expr %x : i1
    }
  }
}

// CHECK-LABEL: func.func @test_bidir
// CHECK-NOT: moore.constraint.implication
func.func @test_bidir(%obj: !moore.class<@BidirImpl>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@BidirImpl>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 11: Implication with range constraint inside
//===----------------------------------------------------------------------===//

moore.class.classdecl @ImplWithRange {
  moore.class.propertydecl @cmd : !moore.i2 rand_mode rand
  moore.class.propertydecl @payload : !moore.i8 rand_mode rand
  moore.constraint.block @c_impl_range {
  ^bb0(%cmd: !moore.i2, %payload: !moore.i8):
    %c1 = moore.constant 1 : i2
    %is_cmd1 = moore.eq %cmd, %c1 : i2 -> i1
    moore.constraint.implication %is_cmd1 : i1 {
      // payload inside {[10:100]}
      moore.constraint.inside %payload, [10, 100] : !moore.i8
    }
  }
}

// CHECK-LABEL: func.func @test_impl_with_range
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.inside
func.func @test_impl_with_range(%obj: !moore.class<@ImplWithRange>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ImplWithRange>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 12: UVM-style transaction with implication and if-else
//===----------------------------------------------------------------------===//

moore.class.classdecl @UVMTransaction {
  moore.class.propertydecl @kind : !moore.i2 rand_mode rand
  moore.class.propertydecl @addr : !moore.i32 rand_mode rand
  moore.class.propertydecl @burst_len : !moore.i4 rand_mode rand
  moore.constraint.block @c_kind {
  ^bb0(%kind: !moore.i2):
    // kind inside {[0:3]}
    moore.constraint.inside %kind, [0, 3] : !moore.i2
  }
  moore.constraint.block @c_addr {
  ^bb0(%kind: !moore.i2, %addr: !moore.i32):
    %c1 = moore.constant 1 : i2
    %c2 = moore.constant 2 : i2
    %is_read = moore.eq %kind, %c1 : i2 -> i1
    %is_write = moore.eq %kind, %c2 : i2 -> i1
    moore.constraint.if_else %is_read : i1 {
      // Read: addr < 0x10000
      %h10000 = moore.constant 65536 : i32
      %lt = moore.ult %addr, %h10000 : i32 -> i1
      moore.constraint.expr %lt : i1
    } else {
      moore.constraint.if_else %is_write : i1 {
        // Write: addr >= 0x10000
        %h10000 = moore.constant 65536 : i32
        %ge = moore.uge %addr, %h10000 : i32 -> i1
        moore.constraint.expr %ge : i1
      }
    }
  }
  moore.constraint.block @c_burst {
  ^bb0(%kind: !moore.i2, %burst_len: !moore.i4):
    %c3 = moore.constant 3 : i2
    %is_burst = moore.eq %kind, %c3 : i2 -> i1
    moore.constraint.implication %is_burst : i1 {
      // burst_len inside {[1:15]}
      moore.constraint.inside %burst_len, [1, 15] : !moore.i4
    }
  }
}

// CHECK-LABEL: func.func @test_uvm_transaction
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.if_else
func.func @test_uvm_transaction(%obj: !moore.class<@UVMTransaction>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@UVMTransaction>
  return %success : i1
}
