// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module SvaContinueOnUnsupported(input logic clk, a);
  covergroup cg_t;
    coverpoint a;
  endgroup
  cg_t cg;
  initial cg = new();

  // Covergroup-handle sampled value for `$rose` is currently unsupported.
  bad_assert: assert property (@(posedge clk)
      $rose(cg));
endmodule

// ERR: error: unsupported sampled value type for $rose

// WARN: warning: unsupported sampled value type for $rose
// WARN: warning: skipping unsupported SVA assertion in continue mode: property lowering failed

// IR: verif.assert
// IR-SAME: {circt.unsupported_sva
// IR-SAME: circt.unsupported_sva_reason = "property lowering failed"
