// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// Regression: direct func.call should honor MAY_YIELD policy the same way as
// call_indirect. By default, MAY_YIELD FuncIds stay interpreted; explicit
// opt-in can re-enable native dispatch within active process context.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// DEFAULT: [circt-sim] func.call skipped (yield):        1
// DEFAULT: [circt-sim] direct_calls_native:              1
// DEFAULT: [circt-sim] direct_calls_interpreted:         1
// DEFAULT: out=42{{$}}
//
// OPTIN: [circt-sim] func.call skipped (yield):        0
// OPTIN: [circt-sim] direct_calls_native:              1
// OPTIN: [circt-sim] direct_calls_interpreted:         0
// OPTIN: out=42{{$}}

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
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
