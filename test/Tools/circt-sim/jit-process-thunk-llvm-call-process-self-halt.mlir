// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: self_handle_nonzero=1
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_exec_hits_total": 1
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_process_self() -> i64

  hw.module @top() {
    %zero = hw.constant 0 : i64
    %fmt_prefix = sim.fmt.literal "self_handle_nonzero="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %self = llvm.call @__moore_process_self() : () -> i64
      %is_nonzero = arith.cmpi ne, %self, %zero : i64
      %fmt_val = sim.fmt.dec %is_nonzero : i1
      %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      llhd.halt
    }

    hw.output
  }
}
