// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | FileCheck %s --check-prefix=OUT
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | grep '^R=' > %t.r1
// RUN: env CIRCT_SIM_RANDOM_SEED=12345 circt-sim %s --compiled=%t.so --top test 2>&1 | grep '^R=' > %t.r2
// RUN: diff %t.r1 %t.r2

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
// OUT: R={{-?[0-9]+}}

llvm.func @__moore_random() -> i32

func.func @get_rand() -> i32 {
  %r = llvm.call @__moore_random() : () -> i32
  return %r : i32
}

hw.module @test() {
  llhd.process {
    %r = func.call @get_rand() : () -> i32
    %lit = sim.fmt.literal "R="
    %d = sim.fmt.dec %r signed : i32
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%lit, %d, %nl)
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
