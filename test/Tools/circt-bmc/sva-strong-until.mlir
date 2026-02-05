// RUN: circt-opt %s --lower-sva-to-ltl | FileCheck %s --check-prefix=CHECK-LTL
// RUN: circt-opt %s --lower-sva-to-ltl --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc="top-module=sva_strong_until bound=10" | FileCheck %s --check-prefix=CHECK-BMC

// End-to-end SVA -> LTL -> BMC coverage for strong until (s_until).

// CHECK-LTL-LABEL: hw.module @sva_strong_until
// CHECK-LTL: ltl.until %{{.*}}, %{{.*}} : i1, i1
// CHECK-LTL: ltl.eventually %{{.*}} : i1
// CHECK-LTL: ltl.and {{.*}}, {{.*}} : !ltl.property, !ltl.property
// CHECK-LTL: verif.clocked_assert

// CHECK-BMC-LABEL: func.func @sva_strong_until
// CHECK-BMC: verif.bmc bound 20
hw.module @sva_strong_until(
  in %clk: i1,
  in %a: i1,
  in %b: i1,
  out out: i1
) {
  %until = sva.prop.until %a, %b strong : i1, i1
  %clocked = sva.prop.clock %until, posedge %clk : !sva.property
  sva.clocked_assert %clocked, posedge %clk : !sva.property
  hw.output %b : i1
}
