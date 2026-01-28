// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect --split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Basic verif.assert, verif.assume, verif.cover conversions with i1 operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_assert
// CHECK-SAME:  ([[ARG0:%.+]]: i1)
// CHECK-DAG:     [[TRUE_BV:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-DAG:     [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK:         [[EQ:%.+]] = smt.eq [[CAST]], [[TRUE_BV]]
// CHECK:         [[NEG:%.+]] = smt.not [[EQ]]
// CHECK:         smt.assert [[NEG]]
// CHECK:         return
func.func @test_assert(%cond: i1) {
  verif.assert %cond : i1
  return
}

// -----

// CHECK-LABEL: func.func @test_assume
// CHECK-SAME:  ([[ARG0:%.+]]: i1)
// CHECK-DAG:     [[TRUE_BV:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-DAG:     [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK:         [[EQ:%.+]] = smt.eq [[CAST]], [[TRUE_BV]]
// CHECK:         smt.assert [[EQ]]
// CHECK:         return
func.func @test_assume(%cond: i1) {
  verif.assume %cond : i1
  return
}

// -----

// Test verif.cover - cover ops are lowered to smt.assert (not negated, like assume)
// This is because cover checking tests reachability.

// CHECK-LABEL: func.func @test_cover
// CHECK-SAME:  ([[ARG0:%.+]]: i1)
// CHECK-DAG:     [[TRUE_BV:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-DAG:     [[CAST:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : i1 to !smt.bv<1>
// CHECK:         [[EQ:%.+]] = smt.eq [[CAST]], [[TRUE_BV]]
// CHECK:         smt.assert [[EQ]]
// CHECK:         return
func.func @test_cover(%cond: i1) {
  verif.cover %cond : i1
  return
}

// -----

// Test multiple verif ops in sequence

// CHECK-LABEL: func.func @test_multiple_ops
// CHECK:         smt.assert
// CHECK:         smt.assert
// CHECK:         smt.assert
// CHECK:         return
func.func @test_multiple_ops(%a: i1, %b: i1, %c: i1) {
  verif.assert %a : i1
  verif.assume %b : i1
  verif.cover %c : i1
  return
}

// -----

// Test assert with label (label is dropped during conversion)

// CHECK-LABEL: func.func @test_assert_with_label
// CHECK:         smt.assert
// CHECK:         return
func.func @test_assert_with_label(%cond: i1) {
  verif.assert %cond label "my_assertion" : i1
  return
}

// -----

//===----------------------------------------------------------------------===//
// BMC with clocked circuit (seq.clock handling)
//===----------------------------------------------------------------------===//

// Test BMC with clocked assertions - verifies that clock edge detection works
// for register updates in bounded model checking.

// CHECK-LABEL: func.func @test_bmc_clocked() -> i1
// CHECK:         smt.solver
// CHECK:         func.call @bmc_init
// CHECK:         scf.for
// CHECK:           func.call @bmc_circuit
// CHECK:           func.call @bmc_loop
// Verify clock edge detection for registers
// CHECK:           smt.bv.not
// CHECK:           smt.bv.and
// CHECK:           smt.check
// CHECK:         }
// CHECK:       }
func.func @test_bmc_clocked() -> (i1) {
  %bmc = verif.bmc bound 3 num_regs 1 initial_values [unit]
  init {
    %clk = seq.const_clock low
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %from_clock = seq.from_clock %clk
    %neg = hw.constant -1 : i1
    %toggled = comb.xor %from_clock, %neg : i1
    %new_clk = seq.to_clock %toggled
    verif.yield %new_clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %reg_state: i8):
    // Property: register value is always less than 100
    %c100 = hw.constant 100 : i8
    %in_range = comb.icmp ult %reg_state, %c100 : i8
    verif.assert %in_range : i1
    // Next register value
    %c1 = hw.constant 1 : i8
    %next = comb.add %reg_state, %c1 : i8
    verif.yield %next : i8
  }
  func.return %bmc : i1
}

// -----

//===----------------------------------------------------------------------===//
// Wider bitvector types
//===----------------------------------------------------------------------===//

// Test with wider integer types to verify bitvector conversion

// CHECK-LABEL: func.func @test_wider_types
// CHECK:         builtin.unrealized_conversion_cast {{%.+}} : i1 to !smt.bv<1>
// CHECK:         smt.assert
func.func @test_wider_types(%a: i32, %b: i32) {
  %cond = comb.icmp eq %a, %b : i32
  verif.assert %cond : i1
  return
}
