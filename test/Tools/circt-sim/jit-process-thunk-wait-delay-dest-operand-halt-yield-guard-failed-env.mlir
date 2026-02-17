// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: env CIRCT_SIM_JIT_FORCE_DEOPT_REQUEST=1 not circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
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
  %delay = llhd.constant_time <1ns, 0d, 0e>
  %sig_out = llhd.sig %false : i1

  %proc_val = llhd.process -> i1 {
    llhd.wait yield (%true : i1), delay %delay, ^bb1(%true : i1)
  ^bb1(%v: i1):
    llhd.halt %v : i1
  }

  llhd.drv %sig_out, %proc_val after %eps : i1
  hw.output
}
