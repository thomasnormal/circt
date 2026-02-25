// RUN: circt-bmc --emit-mlir -b 1 --module top %s | FileCheck %s

hw.module @leaf(out x : i1) {
  %s = verif.symbolic_value : i1
  hw.output %s : i1
}

hw.module @top() attributes {num_regs = 0 : i32, initial_values = []} {
  %u.x = hw.instance "u" @leaf() -> (x : i1)
  verif.assert %u.x : i1
  hw.output
}

// CHECK: func.func @top()
// CHECK: smt.solver
// CHECK: smt.declare_fun "bmc_symbolic_value" : !smt.bv<1>
// CHECK-NOT: verif.symbolic_value
// CHECK: func.func @bmc_circuit
