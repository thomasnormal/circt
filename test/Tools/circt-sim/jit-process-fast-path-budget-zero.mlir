// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --max-time=1000 --jit-hot-threshold=1 --jit-compile-budget=0 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "compile_budget": 0
// JSON: "jit_compiles_total": 0
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %zero = llhd.constant_time <0ns, 0d, 1e>
  %tick = arith.constant 10 : i64
  %clk = llhd.sig %false : i1
  %mirror = llhd.sig %false : i1
  %observedClk = llhd.prb %clk : i1

  // Self-looping wait process (hot in AVIP top-level signal mirrors).
  llhd.process {
    cf.br ^loop
  ^loop:
    %cur = llhd.prb %clk : i1
    llhd.drv %mirror, %cur after %zero : i1
    llhd.wait (%observedClk : i1), ^loop
  }

  // Periodic toggle clock process.
  llhd.process {
    llhd.drv %clk, %false after %zero : i1
    cf.br ^wait
  ^wait:
    %delay = llhd.int_to_time %tick
    llhd.wait delay %delay, ^toggle
  ^toggle:
    %cur = llhd.prb %clk : i1
    %next = comb.xor %cur, %true : i1
    llhd.drv %clk, %next after %zero : i1
    cf.br ^wait
  }

  hw.output
}
