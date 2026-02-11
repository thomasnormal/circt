// RUN: not circt-opt --strip-llhd-interface-signals='strict-llhd=1' %s 2>&1 | FileCheck %s

module {
  hw.module @m(in %a : i1, out out_o : i1) {
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
}

// CHECK: LLHD combinational control flow requires abstraction; rerun without --strict-llhd
// CHECK: failed to lower llhd.combinational for LEC
