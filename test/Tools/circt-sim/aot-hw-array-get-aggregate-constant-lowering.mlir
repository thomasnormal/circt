// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=-1
//
// COMPILED: out=-1

func.func @next_index(%idx: i2) -> i2 {
  // Array literal order is MSB..LSB; hw.array_get logical index 0 selects
  // the last literal element.
  %arr = hw.aggregate_constant [0 : i2, -1 : i2, -2 : i2, 1 : i2] : !hw.array<4xi2>
  %next = hw.array_get %arr[%idx] : !hw.array<4xi2>, i2
  return %next : i2
}

hw.module @top() {
  %c0_i2 = hw.constant 0 : i2
  %c20_i64 = hw.constant 20000000 : i64

  llhd.process {
    %r0 = func.call @next_index(%c0_i2) : (i2) -> i2
    %r1 = func.call @next_index(%r0) : (i2) -> i2
    %r2 = func.call @next_index(%r1) : (i2) -> i2
    %r2_i32 = arith.extsi %r2 : i2 to i32

    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %r2_i32 signed : i32
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
