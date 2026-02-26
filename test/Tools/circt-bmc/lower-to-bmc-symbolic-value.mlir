// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=1" %s | FileCheck %s

hw.module @testModule() attributes {num_regs = 0 : i32, initial_values = []} {
  %s = verif.symbolic_value : i1
  verif.assert %s : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK: bmc_input_names = ["bmc_symbolic_value"]
