// RUN: not circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: circt-verilog --no-uvm-auto-include --sva-continue-on-unsupported --ir-moore %s | FileCheck %s --check-prefix=IR

module immediate_past_event_continue_on_unsupported(input logic clk);
  event ev;
  initial begin
    assert ($past(ev, 1, 1'b1, @(posedge clk)));
  end
endmodule

// ERR: error: unsupported $past value type with sampled-value controls

// WARN: warning: unsupported $past value type with sampled-value controls

// IR-LABEL: moore.module @immediate_past_event_continue_on_unsupported
// IR: moore.assert
