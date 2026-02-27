// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUN

// Regression: lowering sim.proc.print introduces external vararg @printf.
// AOT trampoline generation must not treat host extern varargs as
// interpreter-trampoline candidates.
//
// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE-NOT: cannot generate interpreter trampoline for referenced external vararg function
// COMPILE: [circt-sim-compile] Wrote {{.*}}.so
//
// RUN: V=7

func.func @emit_msg() {
  %v = arith.constant 7 : i32
  %prefix = sim.fmt.literal "V="
  %vf = sim.fmt.dec %v : i32
  %nl = sim.fmt.literal "\0A"
  %msg = sim.fmt.concat (%prefix, %vf, %nl)
  sim.proc.print %msg
  return
}

hw.module @top() {
  llhd.process {
    func.call @emit_msg() : () -> ()
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
