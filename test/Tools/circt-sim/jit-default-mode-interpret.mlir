// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --jit-hot-threshold=1 --jit-compile-budget=-1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: jit-default-mode-interpret
//
// JSON: "mode": "interpret"
// JSON: "jit_config":
// JSON: "hot_threshold": 1
// JSON: "compile_budget": -1
// JSON: "jit":
// JSON: "jit_compiles_total": 0
// JSON: "jit_exec_hits_total": 0
// JSON: "jit_deopts_total": 0

hw.module @top() {
  %fmt = sim.fmt.literal "jit-default-mode-interpret\0A"

  llhd.process {
    sim.proc.print %fmt
    llhd.halt
  }

  hw.output
}
