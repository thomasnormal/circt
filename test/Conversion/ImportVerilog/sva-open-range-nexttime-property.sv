// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s

module SVAOpenRangeNexttimeProperty(input logic clk, a, b, c, d);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // nexttime[m:$] is weak and should lower as weak-eventually over
  // the property shifted by m cycles.
  // CHECK: %[[T:.*]] = hw.constant true
  // CHECK: %[[D2:.*]] = ltl.delay %[[T]], 2, 0 : i1
  // CHECK: %[[P2:.*]] = ltl.implication %[[D2]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[E2:.*]] = ltl.eventually %[[P2]] {ltl.weak} : !ltl.property
  // CHECK: verif.assert %[[E2]] : !ltl.property
  assert property (nexttime [2:$] p);

  // s_nexttime[m:$] is strong and should require eventual satisfaction
  // of the shifted property while still requiring progress to the lower bound.
  // CHECK: %[[D3:.*]] = ltl.delay %[[T]], 3, 0 : i1
  // CHECK: %[[P3:.*]] = ltl.implication %[[D3]], %{{.*}} : !ltl.sequence, !ltl.property
  // CHECK: %[[S3:.*]] = ltl.and %[[D3]], %[[P3]] : !ltl.sequence, !ltl.property
  // CHECK: %[[E3:.*]] = ltl.eventually %[[S3]] : !ltl.property
  // CHECK: verif.assert %[[E3]] : !ltl.property
  assert property (s_nexttime [3:$] p);

  assert property (@(posedge clk) c |-> d);
endmodule
