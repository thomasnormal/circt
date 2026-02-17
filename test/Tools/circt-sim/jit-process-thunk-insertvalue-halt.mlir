// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: insert_halt=1
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  hw.module @top() {
    %true = hw.constant true
    %false = hw.constant false
    %base = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %fmt_prefix = sim.fmt.literal "insert_halt="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %v0 = llvm.insertvalue %true, %base[0] : !llvm.struct<(i1, i1)>
      %v1 = llvm.insertvalue %false, %v0[1] : !llvm.struct<(i1, i1)>
      %bit = llvm.extractvalue %v1[0] : !llvm.struct<(i1, i1)>
      %fmt_val = sim.fmt.dec %bit : i1
      %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      llhd.halt
    }

    hw.output
  }
}
