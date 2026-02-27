// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] 2 functions + 0 processes ready for codegen
//
// COMPILED: out=1

func.func @parity_wrapper(%x: i8) -> i1 {
  %p = comb.parity %x : i8
  return %p : i1
}

func.func @keep_alive(%x: i1) -> i1 {
  return %x : i1
}

hw.module @top() {
  %c20_i64 = hw.constant 20000000 : i64
  %c173 = hw.constant 173 : i8

  llhd.process {
    %rv0 = func.call @parity_wrapper(%c173) : (i8) -> i1
    %rv = func.call @keep_alive(%rv0) : (i1) -> i1
    %rv32 = arith.extui %rv : i1 to i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %rv32 signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
