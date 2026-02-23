// RUN: circt-bmc -b 1 --emit-mlir --assume-known-inputs --module top %s | FileCheck %s --check-prefix=STRICT
// RUN: circt-bmc -b 1 --emit-mlir --assume-known-inputs --x-optimistic --module top %s | FileCheck %s --check-prefix=OPT

// `--x-optimistic` should plumb through circt-bmc to VerifToSMT and affect
// LEC mismatches on 4-state outputs.
hw.module @top(in %clk: i1, in %in: i1) {
  verif.assert %in : i1
  hw.output
}

func.func @main() -> i1 {
  %0 = verif.lec : i1 first {
  ^bb0(%in: !hw.struct<value: i2, unknown: i2>):
    verif.yield %in : !hw.struct<value: i2, unknown: i2>
  } second {
  ^bb0(%in: !hw.struct<value: i2, unknown: i2>):
    %val = hw.struct_extract %in["value"] : !hw.struct<value: i2, unknown: i2>
    %unk = hw.struct_extract %in["unknown"] : !hw.struct<value: i2, unknown: i2>
    %zero = hw.constant 0 : i2
    %any = comb.icmp ne %unk, %zero : i2
    %all = hw.constant -1 : i2
    %unk2 = comb.mux %any, %all, %unk : i2
    %out = hw.struct_create (%val, %unk2) : !hw.struct<value: i2, unknown: i2>
    verif.yield %out : !hw.struct<value: i2, unknown: i2>
  }
  return %0 : i1
}

// STRICT-LABEL: func.func @main() -> i1 {
// STRICT: %c1_out0 = smt.declare_fun "c1_out0" : !smt.bv<4>
// STRICT: %c2_out0 = smt.declare_fun "c2_out0" : !smt.bv<4>
// STRICT: smt.distinct %c1_out0, %c2_out0 : !smt.bv<4>

// OPT-LABEL: func.func @main() -> i1 {
// OPT: %c1_out0 = smt.declare_fun "c1_out0" : !smt.bv<4>
// OPT: %c2_out0 = smt.declare_fun "c2_out0" : !smt.bv<4>
// OPT: smt.bv.extract %c1_out0 from 0 : (!smt.bv<4>) -> !smt.bv<2>
// OPT: smt.bv.extract %c2_out0 from 0 : (!smt.bv<4>) -> !smt.bv<2>
// OPT: smt.bv.xor
// OPT: smt.bv.and
// OPT-NOT: smt.distinct %c1_out0, %c2_out0 : !smt.bv<4>
