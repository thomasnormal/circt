// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=32 --jit-fail-on-deopt --max-time=5000000000 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: call_indirect_delay_done
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  llvm.func @__moore_delay(i64)

  func.func @callee_wait_delay(%delay: i64) {
    llvm.call @__moore_delay(%delay) : (i64) -> ()
    return
  }

  llvm.mlir.global internal @"cls::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @callee_wait_delay]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %delay = hw.constant 1 : i64
    %fmt = sim.fmt.literal "call_indirect_delay_done\0A"

    llhd.process {
      cf.br ^bb1
    ^bb1:
      %vt = llvm.mlir.addressof @"cls::__vtable__" : !llvm.ptr
      %slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fnptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to (i64) -> ()
      func.call_indirect %fn(%delay) : (i64) -> ()
      cf.br ^bb2
    ^bb2:
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
