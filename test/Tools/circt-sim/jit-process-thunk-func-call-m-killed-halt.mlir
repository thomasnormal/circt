// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: killed=0
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_sequence_base::m_killed"(%self: !llvm.ptr) -> i1 {
    %false = arith.constant false
    return %false : i1
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt_prefix = sim.fmt.literal "killed="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      %killed = func.call @"uvm_pkg::uvm_sequence_base::m_killed"(%null) : (!llvm.ptr) -> i1
      %val = sim.fmt.dec %killed : i1
      %out = sim.fmt.concat (%fmt_prefix, %val, %nl)
      sim.proc.print %out
      llhd.halt
    }

    hw.output
  }
}
