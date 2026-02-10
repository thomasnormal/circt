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
  // CHECK-LABEL: hw.module @top
  // CHECK-NOT: llhd_process_result
  // CHECK-NOT: circt.bmc_abstracted_llhd_process_results
  // CHECK-NOT: llhd.drv
  // CHECK-NOT: llhd.process
  // CHECK-EXT-LABEL: hw.module @top
  // CHECK-EXT-NOT: llhd.process
  // CHECK-LABEL: hw.module @assert_proc
  // CHECK-SAME: circt.bmc_abstracted_llhd_process_results = 1 : i32
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

  // CHECK-LABEL: hw.module @assert_entry_proc
  // CHECK: verif.assert
  // CHECK-NOT: llhd.process
  // CHECK-EXT-LABEL: hw.module @assert_entry_proc
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

  // CHECK-LABEL: hw.module @clocked_assert_proc
  // CHECK: verif.clocked_assert
  // CHECK-NOT: llhd.process
  // CHECK-EXT-LABEL: hw.module @clocked_assert_proc
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

  // CHECK-LABEL: hw.module @child
  // CHECK-NOT: llhd_process_result
  // CHECK-NOT: circt.bmc_abstracted_llhd_process_results
  // CHECK-LABEL: hw.module @parent
  // CHECK: hw.instance "u" @child(in: %{{.*}}: !hw.struct<value: i1, unknown: i1>) -> ()
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

  // CHECK-LABEL: hw.module @dead_result_drive
  // CHECK-NOT: llhd_process_result
  // CHECK-NOT: circt.bmc_abstracted_llhd_process_results
  // CHECK-NOT: llhd.drv %cycle
  // CHECK-NOT: llhd.process
  hw.module @dead_result_drive() {
    %c0_i32 = hw.constant 0 : i32
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %cycle = llhd.sig %c0_i32 : i32
    %unused = llhd.prb %cycle : i32
    %p:1 = llhd.process -> i32 {
      cf.br ^bb1(%c0_i32 : i32)
    ^bb1(%cur: i32):
      llhd.wait yield (%cur : i32), delay %t0, ^bb1
    }
    llhd.drv %cycle, %p#0 after %t0 : i32
    hw.output
  }

  // CHECK-LABEL: hw.module @dead_result_helper_process
  // CHECK-NOT: llhd_process_result
  // CHECK-NOT: circt.bmc_abstracted_llhd_process_results
  // CHECK-NOT: llhd.process
  hw.module @dead_result_helper_process() {
    %false = hw.constant false
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %src = llhd.sig %false : i1
    %sink = llhd.sig %false : i1
    %p:1 = llhd.process -> i1 {
      cf.br ^bb1(%false : i1)
    ^bb1(%v: i1):
      llhd.wait yield (%v : i1), delay %t0, ^bb1
    }
    llhd.drv %src, %p#0 after %t0 : i1
    llhd.process {
      %probe = llhd.prb %src : i1
      llhd.drv %sink, %probe after %t0 : i1
      llhd.halt
    }
    hw.output
  }
}
