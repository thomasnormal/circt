// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=7000000
//
// COMPILED: out=7000000

func.func @sample_time_math() -> i64 {
  %nowT = llhd.current_time
  %now = llhd.time_to_int %nowT
  %offT = llhd.constant_time <2ns, 0d, 0e>
  %off = llhd.time_to_int %offT
  %offAsTime = llhd.int_to_time %off
  %offRoundTrip = llhd.time_to_int %offAsTime
  %sum = comb.add %now, %offRoundTrip : i64
  return %sum : i64
}

hw.module @top() {
  %fmtPrefix = sim.fmt.literal "out="
  %fmtNl = sim.fmt.literal "\0A"
  %c5_i64 = hw.constant 5000000 : i64
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^print
  ^print:
    %v = func.call @sample_time_math() : () -> i64
    %fmtV = sim.fmt.dec %v signed : i64
    %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtV, %fmtNl)
    sim.proc.print %fmtOut
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c10_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
