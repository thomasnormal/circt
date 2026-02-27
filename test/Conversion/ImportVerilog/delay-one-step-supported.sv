// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

`timescale 1ns/1ps

module DelayOneStepSupported;
  // DIAG-NOT: unsupported delay control: OneStepDelay
  initial begin
    #1step;
  end
endmodule

// IR-LABEL: moore.module @DelayOneStepSupported
// IR: %[[STEP:.+]] = moore.constant_time 1000 fs
// IR: moore.procedure initial
// IR: moore.wait_delay %[[STEP]]
