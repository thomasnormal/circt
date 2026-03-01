// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN
// XFAIL: *

// Regression tracker: sim.fork-created child processes currently do not have a
// safe runtime exec-model classifier, so MAY_YIELD opt-in still demotes child
// call_indirect dispatch as optin-non-coro. Naive model propagation in runtime
// fork creation regresses APB (`uvm_root::m_do_verbosity_settings` at 0 ns) by
// over-enabling MAY_YIELD native dispatch in dynamic fork trees.
//
// COMPILE: [circt-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// OPTIN: Entry-table native calls:         1
// OPTIN: Entry-table skipped (yield):      0
// OPTIN: [circt-sim] entry_skipped_yield_optin_non_coro:     0
// OPTIN: [circt-sim] direct_skipped_yield_optin_non_coro:     0
// OPTIN: Hot entry MAY_YIELD optin-non-coro skip processes (top 50):
// OPTIN: [circt-sim]   (none)

func.func private @"uvm_pkg::inner_add_one"(%x: i32) -> i32 {
  %one = hw.constant 1 : i32
  %r = arith.addi %x, %one : i32
  return %r : i32
}

func.func private @"uvm_pkg::wrapper_may_yield"(%x: i32) -> i32 {
  // Conservative MAY_YIELD trigger.
  %never = hw.constant false
  cf.cond_br %never, ^do_indirect, ^ret
^do_indirect:
  %zero = llvm.mlir.constant(0 : i64) : i64
  %null = llvm.inttoptr %zero : i64 to !llvm.ptr
  %fn = builtin.unrealized_conversion_cast %null : !llvm.ptr to (i32) -> i32
  %dead = func.call_indirect %fn(%x) : (i32) -> i32
  cf.br ^ret
^ret:
  %r = func.call @"uvm_pkg::inner_add_one"(%x) : (i32) -> i32
  return %r : i32
}

func.func private @spawn_fork_call_indirect(%x: i32) {
  %h = sim.fork join_type "join" {
    %vt = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
    %r = func.call_indirect %fn(%x) : (i32) -> i32
    sim.fork.terminator
  }
  return
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::wrapper_may_yield"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %x = hw.constant 41 : i32
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    func.call @spawn_fork_call_indirect(%x) : (i32) -> ()
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
