// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: HELLO_FROM_FUNC=42
//
// COMPILED: HELLO_FROM_FUNC=42

func.func @emit_msg() {
  %v = arith.constant 42 : i32
  %prefix = sim.fmt.literal "HELLO_FROM_FUNC="
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
