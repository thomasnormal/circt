// RUN: circt-opt %s --lower-sva-to-ltl | FileCheck %s --check-prefix=CHECK-LTL
// RUN: circt-opt %s --lower-sva-to-ltl --lower-clocked-assert-like --lower-ltl-to-core --externalize-registers --lower-to-bmc="top-module=sva_unbounded_until bound=10" | FileCheck %s --check-prefix=CHECK-BMC

// End-to-end SVA -> LTL -> BMC coverage for:
//   a ##[*] b |=> c until d

// CHECK-LTL-LABEL: hw.module @sva_unbounded_until
// CHECK-LTL-DAG: ltl.delay %{{.*}}, 0 : i1
// CHECK-LTL-DAG: ltl.delay %{{.*}}, 1, 0
// CHECK-LTL-DAG: ltl.concat
// CHECK-LTL-DAG: ltl.until %{{.*}}, %{{.*}} : i1, i1
// CHECK-LTL: ltl.implication
// CHECK-LTL: verif.clocked_assert

// CHECK-BMC-LABEL: func.func @sva_unbounded_until
// CHECK-BMC: verif.bmc bound 20

hw.module @sva_unbounded_until(
  in %clk: i1,
  in %a: i1,
  in %b: i1,
  in %c: i1,
  in %d: i1,
  out out: i1
) {
  %delay_b = sva.seq.delay %b, 0 : i1
  %seq = sva.seq.concat %a, %delay_b : i1, !sva.sequence
  %until = sva.prop.until %c, %d : i1, i1
  %prop = sva.prop.implication %seq, %until : !sva.sequence, !sva.property
  %clocked = sva.prop.clock %prop, posedge %clk : !sva.property
  sva.clocked_assert %clocked, posedge %clk : !sva.property
  hw.output %d : i1
}
