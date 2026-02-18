// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=1 --jit-compile-budget=32 --max-time=100 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: profile_guided_indirect_nonsuspending_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit":
// JSON: "jit_compiles_total": 1
// JSON: "jit_deopts_total": 1
// JSON: "jit_deopt_reason_guard_failed": 0
// JSON: "jit_deopt_reason_unsupported_operation": 1
// JSON: "jit_deopt_reason_missing_thunk": 0
// JSON: "jit_call_indirect_sites_total": 1
// JSON: "jit_call_indirect_unresolved_total": 0
// JSON-DAG: "target_set_version": 1
// JSON-DAG: "targets_total": 1
// JSON: "target_name": "callee_a"

module {
  func.func @callee_a(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %sum = arith.addi %x, %one : i32
    return %sum : i32
  }

  func.func @callee_b(%x: i32) -> i32 {
    %two = arith.constant 2 : i32
    %sum = arith.addi %x, %two : i32
    return %sum : i32
  }

  llvm.mlir.global internal @profile_slot(0 : i32) : i32

  llvm.mlir.global internal @profile_cls(0 : i32) { addr_space = 0 : i32,
    circt.vtable_entries = [[0, @callee_a], [1, @callee_b]]
  } : !llvm.array<2 x ptr>

  func.func private @local_helper_profiled_indirect(%obj: !llvm.ptr,
                                                     %x: i32) -> i32 {
    %slot_addr = llvm.mlir.addressof @profile_slot : !llvm.ptr
    %slot_idx = llvm.load %slot_addr : !llvm.ptr -> i32
    %vt_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
    %vt = llvm.load %vt_field : !llvm.ptr -> !llvm.ptr
    %slot_ptr = llvm.getelementptr %vt[0, %slot_idx] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x ptr>
    %fnptr = llvm.load %slot_ptr : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fnptr : !llvm.ptr to (i32) -> i32
    %res = func.call_indirect %fn(%x) : (i32) -> i32
    return %res : i32
  }

  llvm.mlir.global internal @profile_iter(0 : i32) : i32

  hw.module @top() {
    %fmt = sim.fmt.literal "profile_guided_indirect_nonsuspending_ok\0A"
    %delay = llhd.constant_time <0ns, 0d, 1e>
    %one_i64 = arith.constant 1 : i64
    %class_id = arith.constant 123 : i32
    %vt = llvm.mlir.addressof @profile_cls : !llvm.ptr

    llhd.process {
      cf.br ^bb0
    ^bb0:
      %iter_addr = llvm.mlir.addressof @profile_iter : !llvm.ptr
      %iter = llvm.load %iter_addr : !llvm.ptr -> i32
      %limit = arith.constant 3 : i32
      %run = arith.cmpi ult, %iter, %limit : i32
      cf.cond_br %run, ^bb1, ^bb2
    ^bb1:
      %obj = llvm.alloca %one_i64 x !llvm.struct<(i32, ptr)> : (i64) -> !llvm.ptr
      %class_field = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %class_id, %class_field : i32, !llvm.ptr
      %vt_field = llvm.getelementptr %obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, ptr)>
      llvm.store %vt, %vt_field : !llvm.ptr, !llvm.ptr
      %res = func.call @local_helper_profiled_indirect(%obj, %iter) : (!llvm.ptr, i32) -> i32
      %_0 = comb.icmp uge %res, %iter : i32
      %one = arith.constant 1 : i32
      %next = arith.addi %iter, %one : i32
      llvm.store %next, %iter_addr : i32, !llvm.ptr
      llhd.wait delay %delay, ^bb0
    ^bb2:
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
