// RUN: circt-bmc %s --module=top -b 1 --ignore-asserts-until=0 --emit-mlir | FileCheck %s

// LowerToBMC can introduce mixed struct/array aggregate destructuring in
// verif.bmc circuit regions. Ensure the post-LowerToBMC aggregate lowering
// removes these forms before HW->SMT conversion.
// CHECK: func.func @top()
// CHECK: smt.solver
// CHECK-NOT: hw.struct_explode
// CHECK-NOT: hw.struct_extract

module {
  hw.module @top(
      in %clk_i: !hw.struct<value: i1, unknown: i1>,
      in %s: !hw.struct<
        a: !hw.struct<value: i2, unknown: i2>,
        b: !hw.array<2xstruct<value: i3, unknown: i3>>>) {
    %clkv = hw.struct_extract %clk_i["value"] : !hw.struct<value: i1, unknown: i1>
    %a = hw.struct_extract %s["a"] :
      !hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.array<2xstruct<value: i3, unknown: i3>>>
    %av = hw.struct_extract %a["value"] : !hw.struct<value: i2, unknown: i2>
    %idx = hw.constant 1 : i1
    %arr = hw.struct_extract %s["b"] :
      !hw.struct<a: !hw.struct<value: i2, unknown: i2>, b: !hw.array<2xstruct<value: i3, unknown: i3>>>
    %elem = hw.array_get %arr[%idx] : !hw.array<2xstruct<value: i3, unknown: i3>>, i1
    %v = hw.struct_extract %elem["value"] : !hw.struct<value: i3, unknown: i3>
    %c1 = hw.constant 1 : i3
    %ok0 = comb.icmp eq %v, %c1 : i3
    %c0_2 = hw.constant 0 : i2
    %ok1 = comb.icmp eq %av, %c0_2 : i2
    %ok = comb.and %ok0, %ok1 : i1
    verif.clocked_assert %ok, posedge %clkv : i1
    hw.output
  }
}
