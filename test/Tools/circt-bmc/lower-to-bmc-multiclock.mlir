// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=3 allow-multi-clock" %s | FileCheck %s

// CHECK: verif.bmc bound 12
// CHECK: loop
// CHECK: ^bb0(%[[CLK0:.*]]: !seq.clock, %[[CLK1:.*]]: !seq.clock, %[[PHASE:.*]]: i32):

hw.module @testModule(in %clk0 : !seq.clock, in %clk1 : !seq.clock, in %in : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  verif.assert %in : i1
  hw.output
}
