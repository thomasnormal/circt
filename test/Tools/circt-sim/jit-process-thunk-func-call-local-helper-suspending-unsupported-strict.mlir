// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG-DAG: [circt-sim] Strict JIT policy violation: deopts_total=1
// LOG-DAG: [circt-sim] Strict JIT deopt process: id={{[1-9][0-9]*}} name=llhd_process_{{[0-9]+}} reason=unsupported_operation detail=first_op:func.call:local_helper_with_delay
// LOG-DAG: local_helper_suspending_unsupported
// LOG: [circt-sim] Simulation finished with exit code 1
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 0
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 1
// JSON: "jit_deopt_reason_missing_thunk": 0
// JSON: "jit_strict_violations_total": 1

module {
  llvm.func @__moore_delay(i64)

  func.func private @local_helper_with_delay(%arg0: i64) {
    llvm.call @__moore_delay(%arg0) : (i64) -> ()
    return
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "local_helper_suspending_unsupported\0A"
    %delay = arith.constant 10 : i64

    llhd.process {
      func.call @local_helper_with_delay(%delay) : (i64) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
