// RUN: circt-bmc --emit-mlir -b 1 --module top %s | FileCheck %s

module {
  hw.module @top(in %in: i1, in %reg_in: i1, out out: i1, out reg_out: i1) attributes {num_regs = 1 : i32, initial_values = [false]} {
    %c0 = hw.constant false
    verif.assert %in : i1
    hw.output %c0, %reg_in : i1, i1
  }
}

// CHECK: func.func @bmc_circuit
// CHECK: %c0_bv1 = smt.bv.constant
// CHECK: return %c0_bv1, %{{.*}}
