// RUN: circt-bmc -b 2 --module m_unsat %s | FileCheck %s --check-prefix=UNSAT
// RUN: circt-bmc -b 2 --module m_sat %s | FileCheck %s --check-prefix=SAT

// UNSAT: BMC_RESULT=UNSAT
// SAT: BMC_RESULT=SAT

module {
  hw.module @m_unsat(in %clk : !hw.struct<value: i1, unknown: i1>, in %a : !hw.struct<value: i1, unknown: i1>, in %b : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %value = hw.struct_extract %a["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %a["unknown"] : !hw.struct<value: i1, unknown: i1>
    %0 = comb.xor %unknown, %true : i1
    %1 = comb.and bin %value, %0 : i1
    %value_0 = hw.struct_extract %b["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_1 = hw.struct_extract %b["unknown"] : !hw.struct<value: i1, unknown: i1>
    %2 = comb.xor %unknown_1, %true : i1
    %3 = comb.and bin %value_0, %2 : i1
    %4 = ltl.implication %1, %3 : i1, i1
    %5 = ltl.or %4, %true {sva.disable_iff} : !ltl.property, i1
    %value_2 = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_3 = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %6 = comb.xor %unknown_3, %true : i1
    %7 = comb.and bin %value_2, %6 : i1
    verif.clocked_assert %5, posedge %7 : !ltl.property
    hw.output
  }
  hw.module @m_sat(in %clk : !hw.struct<value: i1, unknown: i1>, in %a : !hw.struct<value: i1, unknown: i1>, in %b : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %false = hw.constant false
    %value = hw.struct_extract %a["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %a["unknown"] : !hw.struct<value: i1, unknown: i1>
    %0 = comb.xor %unknown, %true : i1
    %1 = comb.and bin %value, %0 : i1
    %value_0 = hw.struct_extract %b["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_1 = hw.struct_extract %b["unknown"] : !hw.struct<value: i1, unknown: i1>
    %2 = comb.xor %unknown_1, %true : i1
    %3 = comb.and bin %value_0, %2 : i1
    %4 = ltl.implication %1, %3 : i1, i1
    %5 = ltl.or %4, %false {sva.disable_iff} : !ltl.property, i1
    %value_2 = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_3 = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %6 = comb.xor %unknown_3, %true : i1
    %7 = comb.and bin %value_2, %6 : i1
    verif.clocked_assert %5, posedge %7 : !ltl.property
    hw.output
  }
}
