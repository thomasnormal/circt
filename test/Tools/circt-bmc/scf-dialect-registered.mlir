// RUN: circt-bmc -b 1 --emit-mlir --module top %s | FileCheck %s

// Ensure circt-bmc can parse inputs containing SCF operations. This guards
// against regressions where SCF dialect registration is missing.

func.func @dummy(%cond: i1) -> i1 {
  %c0 = arith.constant false
  %c1 = arith.constant true
  %v = scf.if %cond -> (i1) {
    scf.yield %c1 : i1
  } else {
    scf.yield %c0 : i1
  }
  return %v : i1
}

hw.module @top(out out: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %true = hw.constant true
  verif.assert %true : i1
  hw.output %true : i1
}

// CHECK-LABEL: func.func @top
// CHECK: llvm.call @circt_bmc_report_result
