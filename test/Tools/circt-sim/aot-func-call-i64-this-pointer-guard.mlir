// RUN: circt-compile -v %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: direct native func.call with pointer-like i64 `this` arguments
// must demote to interpreted dispatch when the pointer payload is invalid,
// instead of calling native code and crashing.
//
// RUNTIME: Top interpreted callees (candidates for compilation):
// RUNTIME: get_797
// RUNTIME: Top interpreted func.call fallback reasons (top 50):
// RUNTIME: get_797 [pointer-safety=1]

func.func private @get_797(%this: i64) -> i64 {
  %p = llvm.inttoptr %this : i64 to !llvm.ptr
  %v = llvm.load %p : !llvm.ptr -> i64
  return %v : i64
}

hw.module @top() {
  %one = hw.constant 1 : i64
  %prefix = sim.fmt.literal "ret="
  %nl = sim.fmt.literal "\0A"

  llhd.process {
    %r = func.call @get_797(%one) : (i64) -> i64
    %vr = sim.fmt.dec %r signed : i64
    %msg = sim.fmt.concat (%prefix, %vr, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %one
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
