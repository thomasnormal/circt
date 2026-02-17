// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=2 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: fork_child=1
// LOG: fork_parent_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 1
// JSON: "jit_deopt_reason_missing_thunk": 0
// JSON: "jit_deopt_processes":
// JSON: "reason": "unsupported_operation"
// JSON: "detail": "first_op:sim.fork"

module {
  hw.module @top() {
    %true = hw.constant true
    %false = hw.constant false
    %base = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %fmt_child_prefix = sim.fmt.literal "fork_child="
    %fmt_parent = sim.fmt.literal "fork_parent_done\0A"
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %handle = sim.fork join_type "join" {
        %v0 = llvm.insertvalue %true, %base[0] : !llvm.struct<(i1, i1)>
        %v1 = llvm.insertvalue %false, %v0[1] : !llvm.struct<(i1, i1)>
        %bit = llvm.extractvalue %v1[0] : !llvm.struct<(i1, i1)>
        %fmt_val = sim.fmt.dec %bit : i1
        %fmt_out = sim.fmt.concat (%fmt_child_prefix, %fmt_val, %fmt_nl)
        sim.proc.print %fmt_out
        sim.fork.terminator
      }

      sim.proc.print %fmt_parent
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
