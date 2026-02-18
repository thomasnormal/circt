// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Strict JIT policy violation: deopts_total=1
// LOG: [circt-sim] Strict JIT deopt process: id={{[1-9][0-9]*}} name=llhd_process_{{[0-9]+}} reason=guard_failed
// LOG-NOT: detail=multiblock_no_terminal
// LOG: [circt-sim] Simulation finished with exit code 1
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_guard_failed": 1
// JSON: "jit_deopt_reason_unsupported_operation": 0

module {
  hw.module @top() {
    llhd.process {
      cf.br ^bb1
    ^bb1:
      %h = sim.fork join_type "join_none" {
        sim.fork.terminator
      }
      cf.br ^bb1
    }

    hw.output
  }
}
