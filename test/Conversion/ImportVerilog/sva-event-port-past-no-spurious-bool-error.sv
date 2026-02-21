// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

// Regression: event-typed assertion ports used for explicit $past clocking
// should not emit a spurious "cannot be cast to a boolean" diagnostic.

module SVAEventPortPastNoSpuriousBoolError(input logic clk, input logic a);
  sequence inner(event e, logic x);
    $past(x, 1, @(e));
  endsequence

  sequence outer(event e, logic x);
    inner(e, x);
  endsequence

  // CHECK-NOT: cannot be cast to a boolean
  // CHECK: moore.wait_event
  // CHECK: verif.assert
  assert property (outer(posedge clk, a));
endmodule

// IR-LABEL: moore.module @SVAEventPortPastNoSpuriousBoolError
// IR: moore.wait_event
// IR: verif.assert
