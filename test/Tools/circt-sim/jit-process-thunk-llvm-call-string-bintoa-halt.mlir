// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: string_bintoa_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_string_bintoa(i64) -> !llvm.struct<(ptr, i64)>

  hw.module @top() {
    %fmt = sim.fmt.literal "string_bintoa_ok\0A"
    %value = hw.constant 42 : i64

    llhd.process {
      %s = llvm.call @__moore_string_bintoa(%value) : (i64) -> !llvm.struct<(ptr, i64)>
      %len = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>
      %unused = comb.icmp uge %len, %len : i64
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
