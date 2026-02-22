// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --no-uvm-auto-include %s | FileCheck %s --check-prefix=CORE
// REQUIRES: slang

module ImmediateAssertAction(input logic a, output logic b);
  always @* assert (a) else b = 1'b1;
  always @* assert #0 (a) else $error("deferred fail");
endmodule

// MOORE-LABEL: moore.module @ImmediateAssertAction(
// MOORE: moore.procedure always_comb {
// MOORE: moore.assert immediate
// MOORE: cf.cond_br
// MOORE: moore.blocking_assign
// MOORE: moore.procedure always_comb {
// MOORE: moore.assert observed
// MOORE: cf.cond_br

// CORE-LABEL: hw.module @ImmediateAssertAction(
// CORE-COUNT-2: verif.assert
