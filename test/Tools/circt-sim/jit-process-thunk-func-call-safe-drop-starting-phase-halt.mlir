// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: drop_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_sequence_base::m_safe_drop_starting_phase"(%self: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt = sim.fmt.literal "drop_ok\0A"

    llhd.process {
      func.call @"uvm_pkg::uvm_sequence_base::m_safe_drop_starting_phase"(%null) : (!llvm.ptr) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
