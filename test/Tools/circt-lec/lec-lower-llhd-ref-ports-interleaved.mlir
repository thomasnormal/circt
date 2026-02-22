// RUN: circt-opt --lower-llhd-ref-ports %s | FileCheck %s

// Regression test: lower-llhd-ref-ports assumed getPorts() returns inputs
// before outputs, but hw::ModuleType stores ports in their declared
// (interleaved) order. This caused type corruption when modules had
// interleaved in/out ports (e.g. ibex_controller).

// CHECK-LABEL: hw.module private @child
// CHECK-SAME:  in %clk : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME:  in %data_in : !hw.struct<value: i32, unknown: i32>
// CHECK-SAME:  out data_out : !hw.struct<value: i32, unknown: i32>
// CHECK-SAME:  out ref_out1 : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME:  out ref_out2 : !hw.struct<value: i1, unknown: i1>

// CHECK-LABEL: hw.module @parent
// CHECK-SAME:  in %clk : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME:  in %data : !hw.struct<value: i32, unknown: i32>
// CHECK-SAME:  out result : !hw.struct<value: i32, unknown: i32>
module {
  // Ports are intentionally interleaved (in, out, in, out, ...) to trigger
  // the bug where flat port indices were confused with input/output indices.
  hw.module private @child(
    in %clk : !hw.struct<value: i1, unknown: i1>,
    out data_out : !hw.struct<value: i32, unknown: i32>,
    in %data_in : !hw.struct<value: i32, unknown: i32>,
    out ref_out1 : !llhd.ref<!hw.struct<value: i1, unknown: i1>>,
    out ref_out2 : !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  ) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = hw.constant 0 : i2
    %init1 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %sig1 = llhd.sig %init1 : !hw.struct<value: i1, unknown: i1>
    %sig2 = llhd.sig %init1 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig1, %init1 after %0 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig2, %init1 after %0 : !hw.struct<value: i1, unknown: i1>
    hw.output %data_in, %sig1, %sig2 : !hw.struct<value: i32, unknown: i32>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  }

  hw.module @parent(
    in %clk : !hw.struct<value: i1, unknown: i1>,
    in %data : !hw.struct<value: i32, unknown: i32>,
    out result : !hw.struct<value: i32, unknown: i32>
  ) {
    %out, %rout1, %rout2 = hw.instance "child_inst" @child(
      clk: %clk: !hw.struct<value: i1, unknown: i1>,
      data_in: %data: !hw.struct<value: i32, unknown: i32>
    ) -> (data_out: !hw.struct<value: i32, unknown: i32>, ref_out1: !llhd.ref<!hw.struct<value: i1, unknown: i1>>, ref_out2: !llhd.ref<!hw.struct<value: i1, unknown: i1>>)
    %val1 = llhd.prb %rout1 : !hw.struct<value: i1, unknown: i1>
    hw.output %out : !hw.struct<value: i32, unknown: i32>
  }
}
