// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=top bound=2" %s | FileCheck %s

hw.module @top(in %clk: !hw.struct<value: i1, unknown: i1>) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %clk_sig = llhd.sig %clk : !hw.struct<value: i1, unknown: i1>
  llhd.drv %clk_sig, %clk after %t0 : !hw.struct<value: i1, unknown: i1>
  %clk_prb = llhd.prb %clk_sig : !hw.struct<value: i1, unknown: i1>
  %val0, %unk0 = hw.struct_explode %clk_prb : !hw.struct<value: i1, unknown: i1>
  %cat0 = comb.concat %val0, %unk0 : i1, i1
  %v0 = comb.extract %cat0 from 1 : (i2) -> i1
  %u0 = comb.extract %cat0 from 0 : (i2) -> i1
  %not_u0 = comb.xor %u0, %true : i1
  %clk_i1 = comb.and %v0, %not_u0 : i1

  %dut_clk = llhd.sig name "dut/clk" %clk_prb : !hw.struct<value: i1, unknown: i1>
  llhd.drv %dut_clk, %clk_prb after %t0 : !hw.struct<value: i1, unknown: i1>
  llhd.drv %dut_clk, %clk_prb after %t0 : !hw.struct<value: i1, unknown: i1>
  %dut_prb = llhd.prb %dut_clk : !hw.struct<value: i1, unknown: i1>
  %val1, %unk1 = hw.struct_explode %dut_prb : !hw.struct<value: i1, unknown: i1>
  %cat1 = comb.concat %val1, %unk1 : i1, i1
  %v1 = comb.extract %cat1 from 1 : (i2) -> i1
  %u1 = comb.extract %cat1 from 0 : (i2) -> i1
  %not_u1 = comb.xor %u1, %true : i1
  %dut_clk_i1 = comb.and %v1, %not_u1 : i1

  %c0 = seq.to_clock %clk_i1
  %c1 = seq.to_clock %dut_clk_i1
  %r0 = seq.compreg %true, %c0 : i1
  %r1 = seq.compreg %true, %c1 : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK: bmc_input_names = ["bmc_clock",
// CHECK-NOT: bmc_clock_1
