// RUN: circt-opt %s --lower-sva-to-ltl | FileCheck %s --check-prefix=CHECK-LTL
// RUN: circt-opt %s --lower-sva-to-ltl --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc="top-module=sva_multiclock bound=5 allow-multi-clock" | FileCheck %s --check-prefix=CHECK-BMC

// End-to-end SVA -> LTL -> BMC coverage for multiple clocked assertions.

// CHECK-LTL-LABEL: hw.module @sva_multiclock
// CHECK-LTL: ltl.clock {{.*}}, posedge %clk0 : !ltl.property
// CHECK-LTL: ltl.clock {{.*}}, posedge %clk1 : !ltl.property
// CHECK-LTL: verif.clocked_assert
// CHECK-LTL: verif.clocked_assert

// CHECK-BMC-LABEL: func.func @sva_multiclock
// CHECK-BMC: verif.bmc bound 20
// CHECK-BMC: loop
// CHECK-BMC: ^bb0(%{{.*}}: !seq.clock, %{{.*}}: !seq.clock, %{{.*}}: i32):

hw.module @sva_multiclock(in %clk0 : i1, in %clk1 : i1, in %a : i1, in %b : i1) {
  %prop0 = sva.prop.not %a : i1
  %prop1 = sva.prop.not %b : i1
  %clocked0 = sva.prop.clock %prop0, posedge %clk0 : !sva.property
  %clocked1 = sva.prop.clock %prop1, posedge %clk1 : !sva.property
  sva.clocked_assert %clocked0, posedge %clk0 : !sva.property
  sva.clocked_assert %clocked1, posedge %clk1 : !sva.property
  hw.output
}
