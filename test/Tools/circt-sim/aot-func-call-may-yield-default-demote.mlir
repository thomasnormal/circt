// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE=0 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=FIDALLOW
// RUN: env CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 CIRCT_AOT_DENY_FID=',bad,4294967296,1' CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE=',foo,4294967296,0' CIRCT_AOT_TRAP_FID=invalid circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=BADENV

// Regression: direct func.call should honor MAY_YIELD policy the same way as
// call_indirect. By default, MAY_YIELD FuncIds stay interpreted. In opt-in
// mode, non-coroutine contexts can still native-dispatch when static body
// analysis proves the callee is non-suspending.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// DEFAULT: [circt-sim] func.call skipped (yield):        1
// DEFAULT: [circt-sim] direct_calls_native:              1
// DEFAULT: [circt-sim] direct_calls_interpreted:         1
// DEFAULT: [circt-sim] direct_skipped_yield_default:            1
// DEFAULT: Hot func.call MAY_YIELD skips (top 50):
// DEFAULT: [circt-sim]{{[[:space:]]+}}1x fid=0 uvm_pkg::wrapper_may_yield
// DEFAULT: Hot func.call MAY_YIELD optin-non-coro skip processes (top 50):
// DEFAULT: [circt-sim]   (none)
// DEFAULT: out=42{{$}}
//
// OPTIN: [circt-sim] func.call skipped (yield):        0
// OPTIN: [circt-sim] direct_calls_native:              1
// OPTIN: [circt-sim] direct_calls_interpreted:         0
// OPTIN: [circt-sim] direct_skipped_yield_optin_non_coro:     0
// OPTIN: Hot func.call MAY_YIELD skips (top 50):
// OPTIN: [circt-sim]   (none)
// OPTIN: Hot func.call MAY_YIELD optin-non-coro skip processes (top 50):
// OPTIN: [circt-sim]   (none)
// OPTIN: out=42{{$}}
//
// FIDALLOW: [circt-sim] AOT unsafe MAY_YIELD allow list: 1 fids
// FIDALLOW: [circt-sim] func.call skipped (yield):        0
// FIDALLOW: [circt-sim] direct_calls_native:              1
// FIDALLOW: [circt-sim] direct_calls_interpreted:         0
// FIDALLOW: [circt-sim] direct_skipped_yield_default:            0
// FIDALLOW: [circt-sim] direct_skipped_yield_optin_non_coro:     0
// FIDALLOW: out=42{{$}}
//
// BADENV: [circt-sim] WARNING: ignoring invalid FuncId token 'bad' in CIRCT_AOT_DENY_FID
// BADENV: [circt-sim] WARNING: ignoring out-of-range FuncId token '4294967296' in CIRCT_AOT_DENY_FID
// BADENV: [circt-sim] WARNING: ignoring invalid FuncId token 'foo' in CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE
// BADENV: [circt-sim] WARNING: ignoring out-of-range FuncId token '4294967296' in CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD_FIDS_UNSAFE
// BADENV: [circt-sim] WARNING: ignoring invalid integer value 'invalid' in CIRCT_AOT_TRAP_FID
// BADENV: out=42{{$}}

func.func private @"uvm_pkg::inner_add_one"(%x: i32) -> i32 {
  %one = hw.constant 1 : i32
  %r = arith.addi %x, %one : i32
  return %r : i32
}

func.func private @"uvm_pkg::wrapper_may_yield"(%x: i32) -> i32 {
  // Conservative MAY_YIELD classification trigger: call_indirect in body.
  // This branch is never taken at runtime, but the function still gets the
  // MAY_YIELD bit in all_func_flags.
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
    %r = func.call @"uvm_pkg::wrapper_may_yield"(%x) : (i32) -> i32
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
