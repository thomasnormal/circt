// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=7
//
// COMPILED: out=7

func.func @divmod_mix(%a: i32, %b: i32) -> i32 {
  %d = comb.divs %a, %b : i32
  %ru = comb.modu %a, %b : i32
  %rs = comb.mods %a, %b : i32
  %sum0 = comb.add %d, %ru : i32
  %sum1 = comb.add %sum0, %rs : i32
  return %sum1 : i32
}

hw.module @top() {
  llhd.process {
    %a = hw.constant 17 : i32
    %b = hw.constant 5 : i32
    %x = func.call @divmod_mix(%a, %b) : (i32, i32) -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %x signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
