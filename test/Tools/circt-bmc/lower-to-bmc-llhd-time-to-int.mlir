// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=top bound=2" %s | FileCheck %s

module {
  hw.module @top() attributes {num_regs = 0 : i32, initial_values = []} {
    %t = llhd.constant_time <0ns, 0d, 0e>
    %i = llhd.time_to_int %t
    %c0_i64 = hw.constant 0 : i64
    %eq = comb.icmp eq %i, %c0_i64 : i64
    verif.assume %eq : i1
    hw.output
  }
}

// CHECK: verif.bmc
// CHECK: verif.assume
