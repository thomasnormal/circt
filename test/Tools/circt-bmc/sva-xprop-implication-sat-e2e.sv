// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 1 --module=sva_xprop_implication_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_implication_sat(input logic clk, input logic in);
  property p_imp;
    in |-> 1'b1;
  endproperty
  // Implication with X antecedent can be X, so the negated property can fail.
  assert property (@(posedge clk) not p_imp);
endmodule

// CHECK: BMC_RESULT=SAT
