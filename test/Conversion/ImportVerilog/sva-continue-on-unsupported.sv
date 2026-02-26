// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module SvaContinueOnUnsupported(input logic clk, a);
  event ev;
  always_ff @(posedge clk) if (a) -> ev;

  // `event` + sampled-value controls in `$past` is currently unsupported.
  bad_assert: assert property (@(posedge clk)
      $past(ev, 1, 1'b1, @(posedge clk)) == ev);
endmodule

// ERR: error: unsupported $past value type with sampled-value controls

// WARN: warning: unsupported $past value type with sampled-value controls
// WARN: warning: skipping unsupported SVA assertion in continue mode: property lowering failed

// IR: verif.assert
// IR-SAME: {circt.unsupported_sva
// IR-SAME: circt.unsupported_sva_reason = "property lowering failed"
