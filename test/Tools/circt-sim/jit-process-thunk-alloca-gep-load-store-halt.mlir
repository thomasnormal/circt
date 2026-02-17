// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: alloca_gep=42
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_exec_hits_total": 1
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  hw.module @top() {
    %one = hw.constant 1 : i64
    %zero = hw.constant 0 : i32
    %forty_two = hw.constant 42 : i32
    %fmt_prefix = sim.fmt.literal "alloca_gep="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %slot = llvm.alloca %one x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
      %field0 = llvm.getelementptr %slot[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      %field1 = llvm.getelementptr %slot[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
      llvm.store %zero, %field0 : i32, !llvm.ptr
      llvm.store %forty_two, %field1 : i32, !llvm.ptr
      %value = llvm.load %field1 : !llvm.ptr -> i32
      %fmt_val = sim.fmt.dec %value : i32
      %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %fmt_out
      llhd.halt
    }

    hw.output
  }
}
