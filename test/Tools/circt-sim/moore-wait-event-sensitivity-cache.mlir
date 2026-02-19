// RUN: env CIRCT_SIM_TRACE_WAIT_EVENT_CACHE=1 circt-sim %s --max-time=2000000000 2>&1 | FileCheck %s

// CHECK: [WAIT-EVENT-CACHE] store proc=
// CHECK: [WAIT-EVENT-CACHE] hit proc=
// CHECK: [circt-sim] Simulation completed

hw.module @MooreWaitEventSensitivityCacheTest() {
  %true = hw.constant true
  %false = hw.constant false
  %c500000_i64 = arith.constant 500000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %clk = llhd.sig %false : i1
  %doneSig = llhd.sig %false : i1

  // Free-running clock with 500ps half period.
  %0 = llhd.process -> i1 {
    cf.br ^bb1(%false : i1)
  ^bb1(%clk_val: i1):
    %delay = llhd.int_to_time %c500000_i64
    llhd.wait yield (%clk_val : i1), delay %delay, ^bb2
  ^bb2:
    %clk_inv = comb.xor %clk_val, %true : i1
    cf.br ^bb1(%clk_inv : i1)
  }
  llhd.drv %clk, %0 after %eps : i1

  // Re-enter the same moore.wait_event op twice to verify cache reuse.
  %1 = llhd.process -> i1 {
    %c0 = hw.constant 0 : i32
    %c1 = hw.constant 1 : i32
    %c2 = hw.constant 2 : i32
    cf.br ^loop(%c0 : i32)
  ^loop(%count: i32):
    moore.wait_event {
      %clk_prb = llhd.prb %clk : i1
      %clk_moore = builtin.unrealized_conversion_cast %clk_prb : i1 to !moore.l1
      moore.detect_event posedge %clk_moore : l1
    }
    %next = comb.add %count, %c1 : i32
    %doneCond = comb.icmp eq %next, %c2 : i32
    cf.cond_br %doneCond, ^done, ^loop(%next : i32)
  ^done:
    llhd.halt %true : i1
  }
  llhd.drv %doneSig, %1 after %eps : i1

  hw.output
}
