// RUN: not circt-lec --emit-mlir --strict-llhd --lec-approx -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: error: --lec-approx is incompatible with --lec-strict or --strict-llhd

module {
  hw.module @top() {
    hw.output
  }
}
