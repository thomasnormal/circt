// RUN: circt-opt %s --convert-comb-to-smt | FileCheck %s

// Test conversion of llvm.intr.ctpop (popcount) to SMT bitvector operations.
// This is used for $countones, $onehot, and $onehot0 in SystemVerilog.

// CHECK-LABEL: func @test_ctpop_4bit
// CHECK-SAME: ([[ARG:%.+]]: !smt.bv<4>)
func.func @test_ctpop_4bit(%a: !smt.bv<4>) -> !smt.bv<4> {
  %arg = builtin.unrealized_conversion_cast %a : !smt.bv<4> to i4

  // The popcount is computed by extracting each bit, zero-extending it,
  // and summing all the extended bits.
  //
  // CHECK:      [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<4>
  // CHECK-NEXT: [[BIT0:%.+]] = smt.bv.extract [[ARG]] from 0 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[ZEROS3:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<3>
  // CHECK-NEXT: [[EXT0:%.+]] = smt.bv.concat [[ZEROS3]], [[BIT0]] : !smt.bv<3>, !smt.bv<1>
  // CHECK-NEXT: [[SUM0:%.+]] = smt.bv.add [[ZERO]], [[EXT0]] : !smt.bv<4>
  // CHECK-NEXT: [[BIT1:%.+]] = smt.bv.extract [[ARG]] from 1 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[ZEROS3_1:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<3>
  // CHECK-NEXT: [[EXT1:%.+]] = smt.bv.concat [[ZEROS3_1]], [[BIT1]] : !smt.bv<3>, !smt.bv<1>
  // CHECK-NEXT: [[SUM1:%.+]] = smt.bv.add [[SUM0]], [[EXT1]] : !smt.bv<4>
  // CHECK-NEXT: [[BIT2:%.+]] = smt.bv.extract [[ARG]] from 2 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[ZEROS3_2:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<3>
  // CHECK-NEXT: [[EXT2:%.+]] = smt.bv.concat [[ZEROS3_2]], [[BIT2]] : !smt.bv<3>, !smt.bv<1>
  // CHECK-NEXT: [[SUM2:%.+]] = smt.bv.add [[SUM1]], [[EXT2]] : !smt.bv<4>
  // CHECK-NEXT: [[BIT3:%.+]] = smt.bv.extract [[ARG]] from 3 : (!smt.bv<4>) -> !smt.bv<1>
  // CHECK-NEXT: [[ZEROS3_3:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<3>
  // CHECK-NEXT: [[EXT3:%.+]] = smt.bv.concat [[ZEROS3_3]], [[BIT3]] : !smt.bv<3>, !smt.bv<1>
  // CHECK-NEXT: [[RESULT:%.+]] = smt.bv.add [[SUM2]], [[EXT3]] : !smt.bv<4>
  %result = "llvm.intr.ctpop"(%arg) : (i4) -> i4

  %out = builtin.unrealized_conversion_cast %result : i4 to !smt.bv<4>
  return %out : !smt.bv<4>
}

// CHECK-LABEL: func @test_ctpop_1bit
// CHECK-SAME: ([[ARG:%.+]]: !smt.bv<1>)
func.func @test_ctpop_1bit(%a: !smt.bv<1>) -> !smt.bv<1> {
  %arg = builtin.unrealized_conversion_cast %a : !smt.bv<1> to i1

  // For 1-bit input, popcount is just the input itself.
  // CHECK:      [[ZERO:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  // CHECK-NEXT: [[BIT0:%.+]] = smt.bv.extract [[ARG]] from 0 : (!smt.bv<1>) -> !smt.bv<1>
  // CHECK-NEXT: [[RESULT:%.+]] = smt.bv.add [[ZERO]], [[BIT0]] : !smt.bv<1>
  %result = "llvm.intr.ctpop"(%arg) : (i1) -> i1

  %out = builtin.unrealized_conversion_cast %result : i1 to !smt.bv<1>
  return %out : !smt.bv<1>
}

// CHECK-LABEL: func @test_ctpop_8bit
// CHECK-SAME: ([[ARG:%.+]]: !smt.bv<8>)
func.func @test_ctpop_8bit(%a: !smt.bv<8>) -> !smt.bv<8> {
  %arg = builtin.unrealized_conversion_cast %a : !smt.bv<8> to i8

  // Just verify it compiles and produces SMT operations.
  // CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<8>
  // CHECK: smt.bv.extract
  // CHECK: smt.bv.concat
  // CHECK: smt.bv.add
  %result = "llvm.intr.ctpop"(%arg) : (i8) -> i8

  %out = builtin.unrealized_conversion_cast %result : i8 to !smt.bv<8>
  return %out : !smt.bv<8>
}
