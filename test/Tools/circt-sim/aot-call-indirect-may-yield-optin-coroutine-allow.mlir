// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_NATIVE_MAY_YIELD=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=OPTIN

// Regression: MAY_YIELD call_indirect entries stay interpreted by default, but
// opt-in should allow native dispatch when the active process is coroutine-
// classified.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Collected 1 vtable FuncIds
//
// DEFAULT: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// DEFAULT: Entry-table native calls:         0
// DEFAULT: Entry-table trampoline calls:     0
// DEFAULT: Entry-table skipped (yield):      1
// DEFAULT: out=42{{$}}
//
// OPTIN: Entry table: 1 entries for tagged-FuncId dispatch (1 native, 0 non-native)
// OPTIN: Entry-table native calls:         1
// OPTIN: Entry-table trampoline calls:     0
// OPTIN: Entry-table skipped (yield):      0
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
  %t1 = hw.constant 1 : i64

  llhd.process {
    %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i32) -> i32
    %r = func.call_indirect %fn(%x) : (i32) -> i32
    %vf = sim.fmt.dec %r signed : i32
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
    sim.proc.print %msg

    // Force coroutine classification (multiple waits in one process).
    %d1 = llhd.int_to_time %t1
    llhd.wait delay %d1, ^mid
  ^mid:
    %d2 = llhd.int_to_time %t1
    llhd.wait delay %d2, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
