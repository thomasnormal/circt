// RUN: circt-verilog %s --ignore-unknown-modules --ir-hw | FileCheck %s

module tb;
  unknown_mod u();

  initial begin
    // When the instance target is unknown, hierarchical member references can
    // be semantically invalid. With --ignore-unknown-modules this should not
    // hard-fail conversion.
    wait (u.some_signal == 1'b1);
  end
endmodule

// CHECK: hw.module @tb
