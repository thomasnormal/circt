// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=5000 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: jit-process-thunk-multiblock-llvm-call-delay-halt
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0

module {
  llvm.func @__moore_delay(i64)

  hw.module @top() {
    %delay = hw.constant 1 : i64
    %fmt = sim.fmt.literal "jit-process-thunk-multiblock-llvm-call-delay-halt\0A"

    llhd.process {
      cf.br ^bb1
    ^bb1:
      llvm.call @__moore_delay(%delay) : (i64) -> ()
      cf.br ^bb2
    ^bb2:
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
