// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: env CIRCT_SIM_JIT_FORCE_DEOPT_REQUEST=1 not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=2 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: jit-process-thunk-wait-event-derived-observed-dest-operand-halt-yield
// LOG: [circt-sim] Strict JIT policy violation: deopts_total=1
// LOG: [circt-sim] Simulation finished with exit code 1
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_cache_hits_total": {{[1-9][0-9]*}}
// JSON: "jit_exec_hits_total": {{[1-9][0-9]*}}
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_guard_failed": 1
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0
// JSON: "jit_strict_violations_total": 1

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c1000_i64 = arith.constant 1000 : i64
  %clk = llhd.sig %false : i1
  %sig_out = llhd.sig %false : i1
  %fmt = sim.fmt.literal "jit-process-thunk-wait-event-derived-observed-dest-operand-halt-yield\0A"

  llhd.process {
    llhd.drv %clk, %false after %eps : i1
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %delay = llhd.int_to_time %c1000_i64
    llhd.wait delay %delay, ^bb2
  ^bb2:  // pred: ^bb1
    %cur = llhd.prb %clk : i1
    %next = comb.xor %cur, %true : i1
    llhd.drv %clk, %next after %eps : i1
    cf.br ^bb1
  }

  %proc_val = llhd.process -> i1 {
    %val = llhd.prb %clk : i1
    %inv = comb.xor %val, %true : i1
    llhd.wait yield (%val : i1), (%inv : i1), ^bb1(%val : i1)
  ^bb1(%v: i1):
    sim.proc.print %fmt
    llhd.halt %v : i1
  }

  llhd.drv %sig_out, %proc_val after %eps : i1
  hw.output
}
