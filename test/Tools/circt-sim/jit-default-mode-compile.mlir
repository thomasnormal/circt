// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: jit-default-mode-compile
//
// JSON: "mode": "compile"
// JSON: "jit_config":
// JSON: "hot_threshold": 1
// JSON: "compile_budget": 100000

hw.module @top() {
  %fmt = sim.fmt.literal "jit-default-mode-compile\0A"

  llhd.process {
    sim.proc.print %fmt
    llhd.halt
  }

  hw.output
}
