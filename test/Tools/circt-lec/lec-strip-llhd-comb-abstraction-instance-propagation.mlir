// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // CHECK-LABEL: hw.module @leaf(
  // CHECK-SAME: in %a : i1
  // CHECK-SAME: in %llhd_comb : i1
  // CHECK: circt.bmc_abstracted_llhd_interface_inputs = 1 : i32
  // CHECK-NOT: llhd.combinational
  // CHECK: hw.output %llhd_comb : i1
  hw.module @leaf(in %a : i1, out out_o : i1) {
    %comb = llhd.combinational -> i1 {
      cf.br ^header(%a : i1)
    ^header(%x: i1):
      cf.cond_br %x, ^body, ^exit
    ^body:
      %y = comb.xor %x, %x : i1
      cf.br ^header(%y : i1)
    ^exit:
      llhd.yield %x : i1
    }
    hw.output %comb : i1
  }

  // CHECK-LABEL: hw.module @top(
  // CHECK-SAME: in %a : i1
  // CHECK-SAME: in %llhd_comb : i1
  // CHECK-NOT: circt.bmc_abstracted_llhd_interface_inputs
  // CHECK: %[[OUT:.*]] = hw.instance "u" @leaf(a: %a: i1, llhd_comb: %llhd_comb: i1) -> (out_o: i1)
  // CHECK: hw.output %[[OUT]] : i1
  hw.module @top(in %a : i1, out out_o : i1) {
    %u.out_o = hw.instance "u" @leaf(a: %a: i1) -> (out_o: i1)
    hw.output %u.out_o : i1
  }
}
