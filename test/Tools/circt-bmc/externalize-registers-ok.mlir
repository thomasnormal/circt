// RUN: circt-opt --externalize-registers %s | FileCheck %s

hw.module @clk_conv(in %clk: !hw.struct<value: i1, unknown: i1>, in %in: i1, out out: i1) {
  %value = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
  %true = hw.constant true
  %not_unknown = comb.xor %unknown, %true : i1
  %clk_bool = comb.and bin %value, %not_unknown : i1
  %clk_clock = seq.to_clock %clk_bool
  %reg = seq.compreg %in, %clk_clock : i1
  hw.output %reg : i1
}

// CHECK-LABEL: hw.module @clk_conv(
// CHECK-SAME: in %clk : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %in : i1
// CHECK-SAME: in %reg_state : i1
// CHECK-SAME: out out : i1
// CHECK-SAME: out reg_next : i1
// CHECK-SAME: attributes {{{.*}}initial_values = [unit], num_regs = 1 : i32}
// CHECK-NOT: seq.compreg
