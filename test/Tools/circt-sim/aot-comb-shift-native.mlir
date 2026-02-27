// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: comb shift ops should lower in AOT function pipeline.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen
//
// SIM: out=46
//
// COMPILED: Loaded 2 compiled functions: 2 native-dispatched, 0 not-native-dispatched, 0 intercepted
// COMPILED: out=46

func.func @shift_mix(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %shl = comb.shl %x, %c1 : i32
  %shru = comb.shru %shl, %c1 : i32
  %shrs = comb.shrs %shru, %c1 : i32
  %back = comb.shl %shrs, %c1 : i32
  return %back : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @test() {
  %c47_i32 = hw.constant 47 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c15_i64 = hw.constant 15000000 : i64
  %fmt_prefix = sim.fmt.literal "out="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %r = func.call @shift_mix(%c47_i32) : (i32) -> i32
    %fmt_val = sim.fmt.dec %r : i32
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c15_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
