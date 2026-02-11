// RUN: circt-lec --emit-mlir -c1=lhs -c2=rhs %s 2>&1 | FileCheck %s

// Keep an unused Moore helper in the input module. `circt-lec` should parse
// this successfully while proving equivalence for the selected hw modules.
moore.module @unused_moore_helper() {
  %a = moore.variable : !moore.ref<i1>
  moore.procedure initial {
    moore.wait_event {
      %ra = moore.read %a : !moore.ref<i1>
      moore.detect_event any %ra : i1
    }
    moore.return
  }
  moore.output
}

hw.module @lhs(in %i: i1, out o: i1) {
  hw.output %i : i1
}

hw.module @rhs(in %i: i1, out o: i1) {
  hw.output %i : i1
}

// CHECK: @"c1 == c2\0A"
