// RUN: circt-bmc --emit-mlir -b 1 --module top %s 2>&1 | FileCheck %s --check-prefix=KEEP
// RUN: circt-bmc --emit-mlir -b 1 --module top --drop-unsupported-sva %s 2>&1 | FileCheck %s --check-prefix=DROP

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.assert %true : i1
  verif.assert %true {circt.unsupported_sva} : i1
  hw.output %true : i1
}

// KEEP-NOT: circt-bmc: dropped
// DROP: circt-bmc: dropped 1 unsupported SVA assert-like op(s)
