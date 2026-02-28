// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang
// XFAIL: *
// Slang currently rejects open-ended `$` bounds for unary eventually/always.

module SVAOpenRangeUnaryRepeat(input bit a);
  // CHECK-LABEL: moore.module @SVAOpenRangeUnaryRepeat

  // CHECK: %[[TRUE:.*]] = hw.constant true
  // CHECK: [[CONV_A:%.+]] = moore.to_builtin_bool %a : i1

  // CHECK: [[NCR_OPEN:%.+]] = ltl.non_consecutive_repeat [[CONV_A]], 2 : i1
  // CHECK: verif.assert [[NCR_OPEN]] : !ltl.sequence
  assert property (a [= 2:$]);

  // CHECK: [[GOTO_OPEN:%.+]] = ltl.goto_repeat [[CONV_A]], 2 : i1
  // CHECK: verif.assert [[GOTO_OPEN]] : !ltl.sequence
  assert property (a [-> 2:$]);

  // CHECK: [[SE_DELAY_OPEN:%.+]] = ltl.delay [[CONV_A]], 2 : i1
  // CHECK: [[SE_EVENTUALLY:%.+]] = ltl.eventually [[SE_DELAY_OPEN]] : !ltl.sequence
  // CHECK: [[SE_STRONG:%.+]] = ltl.and [[SE_DELAY_OPEN]], [[SE_EVENTUALLY]] : !ltl.sequence, !ltl.property
  // CHECK: verif.assert [[SE_STRONG]] : !ltl.property
  assert property (s_eventually [2:$] a);

  // CHECK: verif.assert [[SE_DELAY_OPEN]] : !ltl.sequence
  assert property (eventually [2:$] a);

  // CHECK: [[SA_DELAY_OPEN:%.+]] = ltl.delay %[[TRUE]], 2, 0 : i1
  // CHECK: [[SA_SHIFTED:%.+]] = ltl.implication [[SA_DELAY_OPEN]], [[CONV_A]] : !ltl.sequence, i1
  // CHECK: [[SA_STRONG_DELAY:%.+]] = ltl.and [[SA_DELAY_OPEN]], [[SA_SHIFTED]] : !ltl.sequence, !ltl.property
  // CHECK: [[SA_NEG:%.+]] = ltl.not [[SA_STRONG_DELAY]] : !ltl.property
  // CHECK: [[SA_EVENTUALLY:%.+]] = ltl.eventually [[SA_NEG]] : !ltl.property
  // CHECK: [[SA_STRONG:%.+]] = ltl.not [[SA_EVENTUALLY]] : !ltl.property
  // CHECK: verif.assert [[SA_STRONG]] : !ltl.property
  assert property (s_always [2:$] a);

  // CHECK: [[A_NEG:%.+]] = ltl.not [[SA_SHIFTED]] : !ltl.property
  // CHECK: [[A_EVENTUALLY:%.+]] = ltl.eventually [[A_NEG]] {ltl.weak} : !ltl.property
  // CHECK: [[A_WEAK:%.+]] = ltl.not [[A_EVENTUALLY]] : !ltl.property
  // CHECK: verif.assert [[A_WEAK]] : !ltl.property
  assert property (always [2:$] a);

  // CHECK: moore.output
endmodule
