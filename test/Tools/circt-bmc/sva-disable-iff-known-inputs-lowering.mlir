// RUN: circt-bmc -b 3 --ignore-asserts-until=1 --module top --emit-mlir %s | FileCheck %s
//
// Regression: disable iff guards should force knownness on contributing
// 4-state inputs even when --assume-known-inputs is disabled.

module {
  hw.module @top(in %clk : !hw.struct<value: i1, unknown: i1>, in %rst : !hw.struct<value: i1, unknown: i1>, in %in : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true

    %clk_value = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %clk_unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %clk_known = comb.xor %clk_unknown, %true : i1
    %clk_bool = comb.and bin %clk_value, %clk_known : i1

    %rst_value = hw.struct_extract %rst["value"] : !hw.struct<value: i1, unknown: i1>
    %rst_unknown = hw.struct_extract %rst["unknown"] : !hw.struct<value: i1, unknown: i1>
    %rst_known = comb.xor %rst_unknown, %true : i1
    %rst_bool = comb.and bin %rst_value, %rst_known : i1

    %in_value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
    %in_unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
    %in_known = comb.xor %in_unknown, %true : i1
    %in_bool = comb.and bin %in_value, %in_known : i1

    %d1 = ltl.delay %in_bool, 1, 0 : i1
    %impl = ltl.implication %in_bool, %d1 : i1, !ltl.sequence
    %prop = ltl.or %rst_bool, %impl {sva.disable_iff} : i1, !ltl.property
    verif.clocked_assert %prop, posedge %clk_bool label "disable-known" : !ltl.property
    hw.output
  }
}

// CHECK: %[[RST:.*]] = smt.declare_fun "rst" : !smt.bv<2>
// CHECK: %[[RST_UNK:.*]] = smt.bv.extract %[[RST]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: %[[RST_KNOWN:.*]] = smt.eq %[[RST_UNK]], %{{.*}} : !smt.bv<1>
// CHECK: smt.assert %[[RST_KNOWN]]
