// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: call_indirect entry-table dispatch must only normalize pointer
// arguments for native entries. Trampoline entries dispatch back into the
// interpreter and must preserve raw pointer payloads (including low-bit tags).
//
// COMPILE: [circt-sim-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE: [circt-sim-compile] Demoted 2 intercepted functions to trampolines
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// RUNTIME: Entry table: 1 entries for tagged-FuncId dispatch (0 native, 1 non-native)
// RUNTIME: Entry-table native calls:         0
// RUNTIME: Entry-table trampoline calls:     1
// RUNTIME: out=1{{$}}

llvm.func @malloc(i64) -> !llvm.ptr

func.func @dummy_native() -> i64 {
  %c = llvm.mlir.constant(0 : i64) : i64
  return %c : i64
}

func.func @"uvm_pkg::dummy::tagbit"(%arg0: !llvm.ptr) -> i64 {
  %run = hw.constant true
  cf.cond_br %run, ^live, ^dead
^dead:
  // Force trampoline demotion while preserving deterministic semantics.
  sim.pause quiet
  cf.br ^live
^live:
  %i = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.and %i, %one : i64
  return %b : i64
}

func.func @caller_indirect(%fptr: !llvm.ptr, %obj: !llvm.ptr) -> i64 {
  %run = hw.constant true
  cf.cond_br %run, ^live, ^dead
^dead:
  // Force trampoline demotion so call_indirect executes in the interpreter.
  sim.pause quiet
  cf.br ^live
^live:
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i64
  %r = func.call_indirect %fn(%obj) : (!llvm.ptr) -> i64
  return %r : i64
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::dummy::tagbit"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %c8 = hw.constant 8 : i64
  %c1 = hw.constant 1 : i64
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %obj = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr
    %objI = llvm.ptrtoint %obj : !llvm.ptr to i64
    %taggedI = llvm.or %objI, %c1 : i64
    %tagged = llvm.inttoptr %taggedI : i64 to !llvm.ptr

    %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr

    %v = func.call @caller_indirect(%fptr, %tagged) : (!llvm.ptr, !llvm.ptr) -> i64
    %vf = sim.fmt.dec %v signed : i64
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
