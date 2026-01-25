// RUN: circt-opt --convert-moore-to-core --split-input-file %s | FileCheck %s

// Track E: Modules with "hvl" in the name should use llhd.process instead of
// seq.initial for their initial blocks. This preserves func.call operations
// which are needed for UVM runtime function calls like run_test().

func.func private @run_test()

// CHECK-LABEL: hw.module @HvlTop()
moore.module @HvlTop() {
  // CHECK: llhd.process
  // CHECK-NOT: seq.initial
  moore.procedure initial {
    func.call @run_test() : () -> ()
    moore.return
  }
  moore.output
}

// -----

func.func private @run_test()

// CHECK-LABEL: hw.module @hvl_top()
moore.module @hvl_top() {
  // CHECK: llhd.process
  // CHECK-NOT: seq.initial
  moore.procedure initial {
    func.call @run_test() : () -> ()
    moore.return
  }
  moore.output
}

// -----

func.func private @run_test()

// CHECK-LABEL: hw.module @my_hvl_module()
moore.module @my_hvl_module() {
  // CHECK: llhd.process
  // CHECK-NOT: seq.initial
  moore.procedure initial {
    func.call @run_test() : () -> ()
    moore.return
  }
  moore.output
}

// -----

func.func private @run_test()

// Non-hvl modules should still use seq.initial for simple initial blocks
// CHECK-LABEL: hw.module @HdlTop()
moore.module @HdlTop() {
  // CHECK: seq.initial
  // CHECK-NOT: llhd.process
  moore.procedure initial {
    func.call @run_test() : () -> ()
    moore.return
  }
  moore.output
}

// -----

func.func private @run_test()

// Regular modules without "hvl" should use seq.initial
// CHECK-LABEL: hw.module @TestBench()
moore.module @TestBench() {
  // CHECK: seq.initial
  // CHECK-NOT: llhd.process
  moore.procedure initial {
    func.call @run_test() : () -> ()
    moore.return
  }
  moore.output
}
