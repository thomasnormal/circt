// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: to_class_wrapper_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func private @to_class_9876(%arg0: i32, %arg1: !llhd.ref<!llvm.ptr>) {
    func.return
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "to_class_wrapper_ok\0A"
    %one = hw.constant 1 : i64
    %value = hw.constant 7 : i32

    llhd.process {
      %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
      %slot_ref = builtin.unrealized_conversion_cast %slot : !llvm.ptr to !llhd.ref<!llvm.ptr>
      func.call @to_class_9876(%value, %slot_ref) : (i32, !llhd.ref<!llvm.ptr>) -> ()
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
