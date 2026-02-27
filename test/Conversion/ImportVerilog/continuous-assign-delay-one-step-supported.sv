// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s --check-prefix=IR
// REQUIRES: slang

`timescale 1ns/1ps

module ContinuousAssignDelayOneStepSupported(input logic a, output logic y);
  assign #1step y = a;
endmodule

// DIAG-NOT: unsupported continuous assignment timing control: OneStepDelay

// IR-LABEL: moore.module @ContinuousAssignDelayOneStepSupported
// IR: %[[TIME:.+]] = moore.constant_time 1000 fs
// IR: %[[Y:.+]] = moore.variable : <l1>
// IR: moore.delayed_assign %[[Y]], %a, %[[TIME]] : l1
// IR: %[[Y_READ:.+]] = moore.read %[[Y]] : <l1>
// IR: moore.output %[[Y_READ]] : !moore.l1
