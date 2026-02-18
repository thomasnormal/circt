// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: apo=1
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @"uvm_pkg::uvm_sequence_base::get_automatic_phase_objection"(%self: !llvm.ptr) -> i1 {
    %true = arith.constant true
    return %true : i1
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %fmt_prefix = sim.fmt.literal "apo="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      %apo = func.call @"uvm_pkg::uvm_sequence_base::get_automatic_phase_objection"(%null) : (!llvm.ptr) -> i1
      %apo_fmt = sim.fmt.dec %apo : i1
      %out = sim.fmt.concat (%fmt_prefix, %apo_fmt, %nl)
      sim.proc.print %out
      llhd.halt
    }

    hw.output
  }
}
