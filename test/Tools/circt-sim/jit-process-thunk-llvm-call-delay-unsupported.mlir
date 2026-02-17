// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Strict JIT policy violation: deopts_total=1
// LOG: [circt-sim] Strict JIT deopt process: id={{[1-9][0-9]*}} name=llhd_process_{{[0-9]+}} reason=unsupported_operation detail=first_op:llvm.call:__moore_delay
// LOG: [circt-sim] Simulation finished with exit code 1
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_unsupported_operation": 1
// JSON: "jit_strict_violations_total": 1
// JSON: "jit_deopt_processes":
// JSON: "reason": "unsupported_operation"
// JSON: "detail": "first_op:llvm.call:__moore_delay"

module {
  llvm.func @__moore_delay(i64)

  hw.module @top() {
    %delay = hw.constant 1 : i64
    %fmt = sim.fmt.literal "jit-process-thunk-llvm-call-delay-unsupported\0A"

    llhd.process {
      llvm.call @__moore_delay(%delay) : (i64) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
