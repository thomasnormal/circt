// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

module SvaContinueOnUnsupported(input logic clk, a);
  covergroup cg_t;
    coverpoint a;
  endgroup
  cg_t cg;
  initial cg = new();

  // Covergroup-handle sampled value for `$rose` should lower as handle truthiness.
  bad_assert: assert property (@(posedge clk)
      $rose(cg));
endmodule

// ERR-NOT: error: unsupported sampled value type for $rose

// WARN-NOT: warning: unsupported sampled value type for $rose
// WARN-NOT: warning: skipping unsupported SVA assertion in continue mode: property lowering failed

// IR: verif.clocked_assert
// IR-NOT: circt.unsupported_sva
