// RUN: circt-bmc --emit-mlir -b 2 --module top %s | FileCheck %s
// RUN: circt-bmc --run-smtlib --z3-path=%S/Inputs/fake-z3-unsat.sh -b 2 --module top %s | FileCheck %s --check-prefix=SMTLIB

// This uses an equivalent derived clock expression before its definition.
// Graph regions permit use-before-def, which blocks CSE from merging the
// duplicate clock expressions. Lower-to-BMC should still treat them as one.
module {
  hw.module @top(in %clk: !hw.struct<value: i1, unknown: i1>, in %in: i1) {
    %true = hw.constant true
    %value = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %not_unknown = comb.xor %unknown, %true : i1
    %clk_a = comb.and bin %value, %not_unknown : i1
    %c0 = seq.to_clock %clk_a
    %r0 = seq.compreg %in, %c0 : i1

    %c1 = seq.to_clock %clk_b
    %r1 = seq.compreg %in, %c1 : i1
    %clk_b = comb.and bin %value, %not_unknown : i1

    verif.assert %r0 : i1
    verif.assert %r1 : i1
    hw.output
  }
}

// CHECK: func.func @top
// SMTLIB: BMC_RESULT=UNSAT
