// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=32 --jit-fail-on-deopt --max-time=5000000000 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG-DAG: target_done
// LOG-DAG: await_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_process_self() -> !llvm.ptr
  llvm.func @__moore_process_await(!llvm.ptr)

  llvm.mlir.global internal @"target_handle"(#llvm.zero) {
    addr_space = 0 : i32
  } : !llvm.ptr

  hw.module @top() {
    %delay = llhd.constant_time <1ns, 0d, 0e>
    %target_fmt = sim.fmt.literal "target_done\0A"
    %await_fmt = sim.fmt.literal "await_done\0A"

    llhd.process {
      %self = llvm.call @__moore_process_self() : () -> !llvm.ptr
      %global = llvm.mlir.addressof @"target_handle" : !llvm.ptr
      llvm.store %self, %global : !llvm.ptr, !llvm.ptr
      llhd.wait delay %delay, ^bb1
    ^bb1:
      sim.proc.print %target_fmt
      llhd.halt
    }

    llhd.process {
      cf.br ^bb1
    ^bb1:
      %global = llvm.mlir.addressof @"target_handle" : !llvm.ptr
      %target = llvm.load %global : !llvm.ptr -> !llvm.ptr
      llvm.call @__moore_process_await(%target) : (!llvm.ptr) -> ()
      cf.br ^bb2
    ^bb2:
      sim.proc.print %await_fmt
      llhd.halt
    }

    hw.output
  }
}
