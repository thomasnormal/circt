// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @bmc_array_reg_int_init
// CHECK: smt.declare_fun : !smt.array<[!smt.bv<1> -> !smt.bv<2>]>
// CHECK: smt.array.store
// CHECK: smt.array.store

func.func @bmc_array_reg_int_init() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 1 initial_values [0 : i4] attributes {
    bmc_input_names = ["clk", "arr_reg"],
    bmc_reg_clocks = ["clk"]
  }
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arr: !hw.array<2xi2>):
    %idx = hw.constant 0 : i1
    %elem = hw.array_get %arr[%idx] : !hw.array<2xi2>, i1
    %c0 = hw.constant 0 : i2
    %ok = comb.icmp eq %elem, %c0 : i2
    verif.assert %ok : i1
    verif.yield %arr : !hw.array<2xi2>
  }
  func.return %bmc : i1
}
