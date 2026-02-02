// RUN: circt-lec --emit-smtlib -c1=ref -c2=dut %s | FileCheck %s

module {
  hw.module @ref(in %data_i : !hw.struct<value: i1, unknown: i1>,
                 out data_o : !hw.struct<value: i1, unknown: i1>) {
    %false = hw.constant false
    %zero = hw.struct_create (%false, %false) : !hw.struct<value: i1, unknown: i1>
    hw.output %zero : !hw.struct<value: i1, unknown: i1>
  }

  hw.module @dut(in %data_i : !hw.struct<value: i1, unknown: i1>,
                 out data_o : !hw.struct<value: i1, unknown: i1>) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %data_i : !hw.struct<value: i1, unknown: i1>
    %comb = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
      llhd.yield %data_i : !hw.struct<value: i1, unknown: i1>
    }
    llhd.drv %sig, %comb after %t0 : !hw.struct<value: i1, unknown: i1>
    %prb = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    hw.output %prb : !hw.struct<value: i1, unknown: i1>
  }
}

// CHECK: declare-const data_i
// CHECK: declare-const c1_out0
// CHECK: declare-const c2_out0
// CHECK: (= c1_out0
// CHECK: (= c2_out0 data_i
// CHECK: distinct c1_out0 c2_out0
