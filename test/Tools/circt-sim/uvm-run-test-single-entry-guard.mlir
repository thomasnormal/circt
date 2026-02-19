// RUN: env CIRCT_SIM_ENFORCE_SINGLE_RUN_TEST=1 circt-sim %s --top test 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// RUN: circt-sim %s --top test 2>&1 | FileCheck %s --check-prefix=CHECK-PASS

// CHECK-FAIL: [circt-sim] error: UVM run_test entered more than once
// CHECK-PASS: [circt-sim] Simulation completed

module {
  func.func @"uvm_pkg::uvm_root::run_test"() {
    return
  }

  hw.module @test() {
    llhd.process {
      func.call @"uvm_pkg::uvm_root::run_test"() : () -> ()
      func.call @"uvm_pkg::uvm_root::run_test"() : () -> ()
      llhd.halt
    }
    hw.output
  }
}
