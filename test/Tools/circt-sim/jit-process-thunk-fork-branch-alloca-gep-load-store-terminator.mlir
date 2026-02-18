// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=8 --jit-fail-on-deopt --max-time=10 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: fork_child_alloca_gep=42
// LOG: fork_parent_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  hw.module @top() {
    %one = hw.constant 1 : i64
    %zero = hw.constant 0 : i32
    %forty_two = hw.constant 42 : i32
    %fmt_child_prefix = sim.fmt.literal "fork_child_alloca_gep="
    %fmt_parent = sim.fmt.literal "fork_parent_done\0A"
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %handle = sim.fork join_type "join" {
        %slot = llvm.alloca %one x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
        %field0 = llvm.getelementptr %slot[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
        %field1 = llvm.getelementptr %slot[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
        llvm.store %zero, %field0 : i32, !llvm.ptr
        llvm.store %forty_two, %field1 : i32, !llvm.ptr
        %value = llvm.load %field1 : !llvm.ptr -> i32
        %fmt_val = sim.fmt.dec %value : i32
        %fmt_out = sim.fmt.concat (%fmt_child_prefix, %fmt_val, %fmt_nl)
        sim.proc.print %fmt_out
        sim.fork.terminator
      }

      sim.proc.print %fmt_parent
      llhd.halt
    }

    hw.output
  }
}
