// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that simple initial blocks (with only constant captures) use seq.initial
// for arcilator support.

// CHECK-LABEL: hw.module @SimpleInitial
moore.module @SimpleInitial() {
  // CHECK: seq.initial() {
  // CHECK:   func.call @dummyA()
  // CHECK: } : () -> ()
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.return
  }
  moore.output
}

func.func private @dummyA()
func.func private @dummyB()

// Test initial block with constants - should use seq.initial
// CHECK-LABEL: hw.module @InitialWithConstants
moore.module @InitialWithConstants() {
  // CHECK: seq.initial() {
  // CHECK:   %c42_i32 = hw.constant 42 : i32
  // CHECK:   func.call @printInt(%c42_i32)
  // CHECK: } : () -> ()
  moore.procedure initial {
    %c = moore.constant 42 : i32
    func.call @printInt(%c) : (!moore.i32) -> ()
    moore.return
  }
  moore.output
}

func.func private @printInt(%arg0: !moore.i32)

// Test initial block that captures module signals - should use llhd.process
// CHECK-LABEL: hw.module @InitialWithSignals
moore.module @InitialWithSignals() {
  %x = moore.variable : !moore.ref<i32>
  // Initial block that references a signal must use llhd.process
  // CHECK: llhd.process {
  // CHECK:   llhd.prb
  // CHECK:   llhd.halt
  // CHECK: }
  moore.procedure initial {
    %0 = moore.read %x : !moore.ref<i32>
    func.call @printInt(%0) : (!moore.i32) -> ()
    moore.return
  }
  moore.output
}

// Test initial block with wait_event - must use llhd.process
// CHECK-LABEL: hw.module @InitialWithWaitEvent
moore.module @InitialWithWaitEvent() {
  %a = moore.variable : !moore.ref<i1>
  // CHECK: llhd.process {
  // CHECK:   llhd.wait
  // CHECK: }
  moore.procedure initial {
    moore.wait_event {
      %0 = moore.read %a : !moore.ref<i1>
      moore.detect_event any %0 : i1
    }
    moore.return
  }
  moore.output
}

// Test multiple simple initial blocks
// CHECK-LABEL: hw.module @MultipleSimpleInitials
moore.module @MultipleSimpleInitials() {
  // CHECK: seq.initial() {
  // CHECK:   func.call @dummyA()
  // CHECK: } : () -> ()
  moore.procedure initial {
    func.call @dummyA() : () -> ()
    moore.return
  }
  // CHECK: seq.initial() {
  // CHECK:   func.call @dummyB()
  // CHECK: } : () -> ()
  moore.procedure initial {
    func.call @dummyB() : () -> ()
    moore.return
  }
  moore.output
}
