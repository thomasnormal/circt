// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_ref bound=4" %s | FileCheck %s

hw.module @llhd_ref(in %sig: !llhd.ref<i1>, out out: !llhd.ref<i1>) {
  hw.output %sig : !llhd.ref<i1>
}

// CHECK: verif.bmc
// CHECK: ^bb0(%[[SIG:.*]]: !llhd.ref<i1>):
// CHECK: %[[PRB:.*]] = llhd.prb %[[SIG]] : i1
// CHECK: verif.yield %[[PRB]]
