// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: comb_cf=0
// LOG: comb_cf=1
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "final_time_fs": 1000000
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_cache_hits_total": 2
// JSON: "jit_exec_hits_total": 2
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %delay = llhd.constant_time <1ns, 0d, 0e>

  %a = llhd.sig %false : i1
  %fmt_prefix = sim.fmt.literal "comb_cf="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.combinational {
    %a_val = llhd.prb %a : i1
    cf.cond_br %a_val, ^bb_true, ^bb_false
  ^bb_true:
    cf.br ^bb_emit(%true : i1)
  ^bb_false:
    cf.br ^bb_emit(%false : i1)
  ^bb_emit(%v: i1):
    %fmt_val = sim.fmt.dec %v : i1
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.yield
  }

  llhd.drv %a, %true after %delay : i1

  hw.output
}
