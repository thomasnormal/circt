// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --max-time=80 --jit-hot-threshold=1 --jit-compile-budget=0 --sim-stats --process-stats --process-stats-top=8 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: [circt-sim] Simulation completed
// LOG: proc {{[0-9]+}} 'llhd_process_0' steps=0
// LOG: proc {{[0-9]+}} 'llhd_process_1' steps=0
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
  %one = llvm.mlir.constant(1 : i64) : i64

  %rawMem = llvm.call @malloc(%one) : (i64) -> !llvm.ptr
  %mem = llhd.sig %rawMem : !llvm.ptr

  %clk = llhd.sig %false : i1
  %observedClk = llhd.prb %clk : i1

  // Hot mirror loop: probe/store/wait-self-loop.
  llhd.process {
    cf.br ^loop
  ^loop:
    %ptr = llhd.prb %mem : !llvm.ptr
    %cur = llhd.prb %clk : i1
    llvm.store %cur, %ptr : i1, !llvm.ptr
    llhd.wait (%observedClk : i1), ^loop
  }

  // Keep the mirror loop active with a periodic clock toggler.
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

llvm.func @malloc(i64) -> !llvm.ptr
