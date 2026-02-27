// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR

module immediate_past_covergroup_continue_on_unsupported(input logic clk);
  covergroup cg_t;
    coverpoint clk;
  endgroup
  cg_t cg;
  initial cg = new();
  initial begin
    assert ($rose(cg));
  end
endmodule

// ERR: error: unsupported sampled value type for $rose

// WARN: warning: $rose has unsupported sampled value type

// IR-LABEL: moore.module @immediate_past_covergroup_continue_on_unsupported
// IR: moore.assert
