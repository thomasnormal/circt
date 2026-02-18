// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: circt-sim %s --skip-passes --mode=compile --jit-hot-threshold=99 --jit-compile-budget=1 --jit-report=%t/jit.json > %t/log.txt 2>&1
// RUN: FileCheck %s --check-prefix=LOG < %t/log.txt
// RUN: FileCheck %s --check-prefix=JSON < %t/jit.json
//
// LOG: jit_report_call_indirect_target_profile_ok
// LOG: [circt-sim] Simulation completed
//
// JSON: "mode": "compile"
// JSON: "jit_call_indirect_sites_total": 2
// JSON: "jit_call_indirect_calls_total": 2
// JSON: "jit_call_indirect_unresolved_total": 1
// JSON: "jit_call_indirect_sites": [
// JSON-DAG: "unresolved_calls": 1
// JSON-DAG: "targets_total": 0
// JSON-DAG: "targets_total": 1
// JSON-DAG: "target_set_version": 1
// JSON-DAG: "target_name": "callee_a"
// JSON-DAG: "calls": 1

module {
  func.func @callee_a(%x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %sum = arith.addi %x, %one : i32
    return %sum : i32
  }

  llvm.mlir.global internal @"profile_cls::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @callee_a]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %fmt = sim.fmt.literal "jit_report_call_indirect_target_profile_ok\0A"
    %seed0 = arith.constant 7 : i32
    %seed1 = arith.constant 11 : i32
    %vt = llvm.mlir.addressof @"profile_cls::__vtable__" : !llvm.ptr

    llhd.process {
      %slot = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fp = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fp : !llvm.ptr to (i32) -> i32
      %res0 = func.call_indirect %fn(%seed0) : (i32) -> i32 loc("site.resolved")

      %zero = llvm.mlir.constant(0 : i64) : i64
      %null = llvm.inttoptr %zero : i64 to !llvm.ptr
      %fn_null = builtin.unrealized_conversion_cast %null : !llvm.ptr to (i32) -> i32
      %res1 = func.call_indirect %fn_null(%seed1) : (i32) -> i32 loc("site.unresolved")

      %_0 = comb.icmp uge %res0, %seed0 : i32
      %_1 = comb.icmp uge %res1, %res1 : i32
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}
