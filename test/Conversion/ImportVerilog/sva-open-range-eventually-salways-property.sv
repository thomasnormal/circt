// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// XFAIL: *
// Slang currently rejects open-ended `$` bounds for these property forms.

module SVAOpenRangeEventuallySAlwaysProperty(input logic clk, a, b);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // eventually[m:$] is weak and should lower as weak-eventually over
  // the property shifted by m cycles.
  // CHECK: %[[T:.*]] = hw.constant true
  // CHECK: %[[D1:.*]] = ltl.delay %[[T]], 1, 0 : i1
  // CHECK: %[[P1:.*]] = ltl.implication %[[D1]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[E1:.*]] = ltl.eventually %[[P1]] {ltl.weak} : !ltl.property
  // CHECK: verif.assert %[[E1]] : !ltl.property
  assert property (eventually [1:$] p);

  // s_always[m:$] is strong and should require finite progress to the lower
  // bound before enforcing strong always from that point onward.
  // CHECK: %[[S2:.*]] = ltl.and %[[D1]], %[[P1]] : !ltl.sequence, !ltl.property
  // CHECK: %[[N2:.*]] = ltl.not %[[S2]] : !ltl.property
  // CHECK: %[[E2:.*]] = ltl.eventually %[[N2]] : !ltl.property
  // CHECK: %[[A2:.*]] = ltl.not %[[E2]] : !ltl.property
  // CHECK: verif.assert %[[A2]] : !ltl.property
  assert property (s_always [1:$] p);
endmodule
