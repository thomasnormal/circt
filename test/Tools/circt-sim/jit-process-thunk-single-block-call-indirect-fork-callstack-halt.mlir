// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=4 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @callee() {
    %h = sim.fork {
      sim.fork.terminator
    }
    return
  }

  llvm.mlir.global internal @"cls::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @callee]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %fmt = sim.fmt.literal "done\0A"

    llhd.process {
      %vt = llvm.mlir.addressof @"cls::__vtable__" : !llvm.ptr
      %slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fnptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to () -> ()
      func.call_indirect %fn() : () -> ()
      sim.proc.print %fmt
      llhd.halt
    }
    hw.output
  }
}
