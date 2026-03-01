// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE=0 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=FIDALLOW

// Regression: call_indirect should keep MAY_YIELD FuncIds interpreted by
// default. In opt-in mode, non-coroutine contexts can still native-dispatch
// only when static body analysis proves the callee is non-suspending.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// DEFAULT: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// DEFAULT: Entry-table native calls:         0
// DEFAULT: Entry-table trampoline calls:     0
// DEFAULT: Entry-table skipped (yield):      1
// DEFAULT: [circt-sim] entry_skipped_yield_native_default:     1
// DEFAULT: Hot entry-table MAY_YIELD skips (top 50):
// DEFAULT: [circt-sim]{{[[:space:]]+}}1x fid=0 uvm_pkg::wrapper_may_yield
// DEFAULT: Hot entry MAY_YIELD optin-non-coro skip processes (top 50):
// DEFAULT: [circt-sim]   (none)
// DEFAULT: out=42{{$}}
//
// OPTIN: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// OPTIN: Entry-table native calls:         1
// OPTIN: Entry-table trampoline calls:     0
// OPTIN: Entry-table skipped (yield):      0
// OPTIN: [circt-sim] entry_skipped_yield_optin_non_coro:     0
// OPTIN: Hot entry-table MAY_YIELD skips (top 50):
// OPTIN: [circt-sim]   (none)
// OPTIN: Hot entry MAY_YIELD optin-non-coro skip processes (top 50):
// OPTIN: [circt-sim]   (none)
// OPTIN: out=42{{$}}
//
// FIDALLOW: [circt-sim] AOT unsafe MAY_YIELD allow list: 1 fids
// FIDALLOW: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// FIDALLOW: Entry-table native calls:         1
// FIDALLOW: Entry-table trampoline calls:     0
// FIDALLOW: Entry-table skipped (yield):      0
// FIDALLOW: [circt-sim] entry_skipped_yield_native_default:     0
// FIDALLOW: [circt-sim] entry_skipped_yield_optin_non_coro:     0
// FIDALLOW: out=42{{$}}

func.func private @"uvm_pkg::inner_add_one"(%x: i32) -> i32 {
  %one = hw.constant 1 : i32
  %r = arith.addi %x, %one : i32
  return %r : i32
}

func.func private @"uvm_pkg::wrapper_may_yield"(%x: i32) -> i32 {
  // Conservative MAY_YIELD classification trigger: call_indirect in body.
  // This branch is never taken at runtime, but the symbol still carries the
  // MAY_YIELD flag in all_func_flags.
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

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::wrapper_may_yield"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %x = hw.constant 41 : i32
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
    %r = func.call_indirect %fn(%x) : (i32) -> i32
    %vf = sim.fmt.dec %r signed : i32
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
