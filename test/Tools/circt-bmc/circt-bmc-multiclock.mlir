// RUN: circt-bmc --emit-mlir -b 1 --module testModule --allow-multi-clock %s | FileCheck %s
// RUN: not circt-bmc --emit-mlir -b 1 --module testModule %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

module {
  hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %in : i1) {
    %c0 = seq.from_clock %clk0
    %c1 = seq.from_clock %clk1
    verif.clocked_assert %in, posedge %c0 : i1
    verif.clocked_assert %in, posedge %c1 : i1
    hw.output
  }
}

// CHECK: func.func @testModule
// CHECK-ERR: modules with multiple clocks not yet supported
