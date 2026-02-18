// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --max-time=2500 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=1 --jit-fail-on-deopt --parallel=4 --work-stealing --auto-partition --max-time=2500 --jit-report=%t/jit-parallel.json > %t/log-parallel.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
// RUN: FileCheck %s --check-prefix=LOG < %t/log-parallel.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit-parallel.json
//
// LOG: local_helper_call_indirect_static_nonsuspending_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_deopts_total": 0
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 0
// JSON: "jit_deopt_reason_missing_thunk": 0

module {
  func.func @callee_indirect_nonsuspending(%arg0: i32) -> i32 {
    %one = arith.constant 1 : i32
    %sum = arith.addi %arg0, %one : i32
    return %sum : i32
  }

  llvm.mlir.global internal @"cls::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @callee_indirect_nonsuspending]]
  } : !llvm.array<1 x ptr>

  func.func private @local_helper_call_indirect_nonsuspending(%arg0: i32) -> i32 {
    %vt = llvm.mlir.addressof @"cls::__vtable__" : !llvm.ptr
    %slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fnptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to (i32) -> i32
    %res = func.call_indirect %fn(%arg0) : (i32) -> i32
    return %res : i32
  }

  hw.module @top() {
    %fmt = sim.fmt.literal "local_helper_call_indirect_static_nonsuspending_ok\0A"
    %seed = arith.constant 7 : i32

    llhd.process {
      %result = func.call @local_helper_call_indirect_nonsuspending(%seed) : (i32) -> i32
      %unused = comb.icmp uge %result, %seed : i32
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
