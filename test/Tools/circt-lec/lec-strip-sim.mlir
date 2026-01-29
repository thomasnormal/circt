// RUN: circt-lec --emit-mlir -c1=m -c2=m %s %s | FileCheck %s

module {
  hw.module @m(in %clock: !seq.clock, in %enable: i1) {
    %lit = sim.fmt.literal "hello"
    sim.print %lit on %clock if %enable
    sim.terminate success, quiet
    hw.output
  }
}

// CHECK: smt.solver
// CHECK-NOT: sim.
