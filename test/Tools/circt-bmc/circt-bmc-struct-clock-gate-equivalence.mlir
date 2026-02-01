// RUN: circt-opt --externalize-registers --hw-aggregate-to-comb --hw-convert-bitcasts --lower-to-bmc="top-module=top bound=2" %s | FileCheck %s

// CHECK: verif.bmc
// CHECK: bmc_clock_keys = ["arg{{[0-9]+}}{{(:inv)?}}"]
// CHECK: init {
// CHECK: seq.to_clock
// CHECK-NOT: seq.to_clock
// CHECK: loop {

module {
  hw.module @top(in %clk: !hw.struct<value: i1, unknown: i1>, in %in: i1) {
    %true = hw.constant true
    %value = hw.struct_extract %clk["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %clk["unknown"] : !hw.struct<value: i1, unknown: i1>
    %not_unknown = comb.xor %unknown, %true : i1
    %gated = comb.and bin %value, %not_unknown : i1
    %c0 = seq.to_clock %value
    %r0 = seq.compreg %in, %c0 : i1
    %c1 = seq.to_clock %gated
    %r1 = seq.compreg %in, %c1 : i1
    verif.assert %r0 : i1
    verif.assert %r1 : i1
    hw.output
  }
}
