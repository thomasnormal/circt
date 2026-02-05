// RUN: circt-bmc --emit-mlir -b 1 --module m %s | FileCheck %s

module {
  hw.module @m(in %clock: !seq.clock, in %enable: i1) {
    %lit = sim.fmt.literal "hello"
    sim.print %lit on %clock if %enable
    sim.terminate success, quiet
    hw.output
  }
}

// CHECK: func.func @m
// CHECK-NOT: sim.
