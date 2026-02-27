// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED
//
// Regression: func.call native fast path must reject float signatures. Calling
// a compiled f64->f64 function through the raw uint64 ABI returned garbage.
//
// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILED: Loaded 1 compiled functions: 1 native-dispatched
// SIM: bits=4612811918334230528
// COMPILED: bits=4612811918334230528

func.func @id_f64(%x: f64) -> f64 {
  return %x : f64
}

hw.module @top() {
  llhd.process {
    %bits_in = hw.constant 4612811918334230528 : i64
    %in = llvm.bitcast %bits_in : i64 to f64
    %out = func.call @id_f64(%in) : (f64) -> f64
    %bits_out = llvm.bitcast %out : f64 to i64

    %pfx = sim.fmt.literal "bits="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %bits_out : i64
    %msg = sim.fmt.concat (%pfx, %fmt, %nl)
    sim.proc.print %msg

    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
