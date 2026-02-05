// RUN: circt-opt --strip-llhd-processes %s | FileCheck %s --check-prefix=CHECK
// RUN: circt-opt --strip-llhd-processes --externalize-registers %s | FileCheck %s --check-prefix=CHECK-EXT

module {
  hw.module @top() {
    %true = hw.constant true
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %sig_init : !hw.struct<value: i1, unknown: i1>
    %p:2 = llhd.process -> !hw.struct<value: i1, unknown: i1>, i1 {
      cf.br ^bb1(%sig_init, %true : !hw.struct<value: i1, unknown: i1>, i1)
    ^bb1(%v: !hw.struct<value: i1, unknown: i1>, %en: i1):
      llhd.wait yield (%v, %en : !hw.struct<value: i1, unknown: i1>, i1), delay %t0, ^bb1
    }
    llhd.drv %sig, %p#0 after %t0 if %p#1 : !hw.struct<value: i1, unknown: i1>
    hw.output
  }
  // CHECK: hw.module @top
  // CHECK: llhd_process_result
  // CHECK: llhd.drv
  // CHECK-NOT: llhd.process
  // CHECK-EXT: hw.module @top
  // CHECK-EXT-NOT: llhd.process
  // CHECK: hw.module @assert_proc
  // CHECK: llhd.prb
  // CHECK: hw.struct_extract
  // CHECK: verif.assert
  // CHECK-NOT: llhd.process
  hw.module @assert_proc() {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %sig_init : !hw.struct<value: i1, unknown: i1>
    %prb = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    %p:1 = llhd.process -> !hw.struct<value: i1, unknown: i1> {
      cf.br ^bb1(%prb : !hw.struct<value: i1, unknown: i1>)
    ^bb1(%cur: !hw.struct<value: i1, unknown: i1>):
      %value = hw.struct_extract %cur["value"] : !hw.struct<value: i1, unknown: i1>
      verif.assert %value : i1
      llhd.wait yield (%cur : !hw.struct<value: i1, unknown: i1>), delay %t0, ^bb1
    }
    llhd.drv %sig, %p#0 after %t0 : !hw.struct<value: i1, unknown: i1>
    hw.output
  }

  // CHECK: hw.module @assert_entry_proc
  // CHECK: verif.assert
  // CHECK-NOT: llhd.process
  // CHECK-EXT: hw.module @assert_entry_proc
  // CHECK-EXT: verif.assert
  // CHECK-EXT-NOT: llhd.process
  hw.module @assert_entry_proc() {
    %true = hw.constant true
    llhd.process {
      verif.assert %true {bmc.final} : i1
      llhd.halt
    }
    hw.output
  }

  // CHECK: hw.module @clocked_assert_proc
  // CHECK: verif.clocked_assert
  // CHECK-NOT: llhd.process
  // CHECK-EXT: hw.module @clocked_assert_proc
  // CHECK-EXT: verif.clocked_assert
  // CHECK-EXT-NOT: llhd.process
  hw.module @clocked_assert_proc(in %clk : i1) {
    %true = hw.constant true
    llhd.process {
      cf.br ^bb1(%true, %clk : i1, i1)
    ^bb1(%cond: i1, %clock: i1):
      verif.clocked_assert %cond, posedge %clock : i1
      llhd.halt
    }
    hw.output
  }

  // CHECK: hw.module @child
  // CHECK: llhd_process_result
  // CHECK: llhd_process_result_0
  // CHECK: hw.module @parent
  // CHECK: hw.instance "u" @child(in: %{{.*}}, llhd_process_result: %{{.*}}, llhd_process_result_0: %{{.*}}) -> ()
  hw.module @child(in %in : !hw.struct<value: i1, unknown: i1>) {
    %true = hw.constant true
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %sig_init : !hw.struct<value: i1, unknown: i1>
    %p:2 = llhd.process -> !hw.struct<value: i1, unknown: i1>, i1 {
      cf.br ^bb1(%sig_init, %true : !hw.struct<value: i1, unknown: i1>, i1)
    ^bb1(%v: !hw.struct<value: i1, unknown: i1>, %en: i1):
      llhd.wait yield (%v, %en : !hw.struct<value: i1, unknown: i1>, i1), delay %t0, ^bb1
    }
    llhd.drv %sig, %p#0 after %t0 if %p#1 : !hw.struct<value: i1, unknown: i1>
    hw.output
  }

  hw.module @parent() {
    %sig_init = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    hw.instance "u" @child(in: %sig_init : !hw.struct<value: i1, unknown: i1>) -> ()
    hw.output
  }
}
