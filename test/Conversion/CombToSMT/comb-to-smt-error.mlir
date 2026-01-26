// RUN: circt-opt %s --convert-comb-to-smt --split-input-file --verify-diagnostics

func.func @zero_width_parity(%a0: !smt.bv<32>) {
  %arg0 = builtin.unrealized_conversion_cast %a0 : !smt.bv<32> to i0
  // Fails because "unable to convert type for operand #0, type was 'i0'"
  // expected-error @below {{failed to legalize operation 'comb.parity' that was explicitly marked illegal}}
  %0 = comb.parity %arg0 : i0
  return
}
