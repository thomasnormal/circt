// RUN: circt-verilog --no-uvm-auto-include %s --ir-hw 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: task body accesses a module-scope interface instance.
// This must thread the interface ref as a capture into the lowered task
// function, otherwise MLIR verifier trips region isolation:
//   'moore.read' op using value defined outside the region
//
// CHECK-NOT: error:
// CHECK-NOT: 'moore.read' op using value defined outside the region
// CHECK: hw.module @top

interface ifc;
  logic sig;
endinterface

module top;
  ifc vif();

  task automatic do_it();
    if (vif.sig)
      vif.sig = 1'b0;
  endtask

  initial do_it();
endmodule
