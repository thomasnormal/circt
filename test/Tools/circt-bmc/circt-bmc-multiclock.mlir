// RUN: circt-bmc --emit-mlir -b 1 --module testModule --allow-multi-clock %s | FileCheck %s
// RUN: not circt-bmc --emit-mlir -b 1 --module testModule %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

module {
  hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %in : i1) {
    verif.assert %in : i1
    hw.output
  }
}

// CHECK: func.func @testModule
// CHECK-ERR: modules with multiple clocks not yet supported
