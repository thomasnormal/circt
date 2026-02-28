// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// XFAIL: *
// Slang currently rejects open-ended `$` bounds for `nexttime` forms.

module SVAOpenRangeNexttimeSequence(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // nexttime[m:$] over a sequence should lower to an unbounded delay.
  // CHECK: %[[D2:.*]] = ltl.delay %{{.*}}, 2 : !ltl.sequence
  // CHECK: verif.assert %[[D2]] : !ltl.sequence
  assert property (nexttime [2:$] s);

  // s_nexttime[m:$] over a sequence should require eventual progress.
  // CHECK: %[[D3:.*]] = ltl.delay %{{.*}}, 3 : !ltl.sequence
  // CHECK: %[[E3:.*]] = ltl.eventually %[[D3]] : !ltl.sequence
  // CHECK: %[[S3:.*]] = ltl.and %[[D3]], %[[E3]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert %[[S3]] : !ltl.property
  assert property (s_nexttime [3:$] s);
endmodule
