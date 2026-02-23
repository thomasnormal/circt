// RUN: not circt-sim %s 2>&1 | FileCheck %s

// CHECK: error: circt-sim expects MLIR input; for SystemVerilog run circt-verilog first and pass the lowered MLIR

module top;
endmodule
