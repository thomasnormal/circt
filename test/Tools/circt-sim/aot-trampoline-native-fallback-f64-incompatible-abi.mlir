// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Regression: trampoline native fallback must not dispatch external llvm.func
// float signatures via raw uint64 ABI casts. This previously produced
// deterministic garbage output in compiled mode.
//
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
//
// RUNTIME: WARNING: unsupported trampoline native fallback ABI
// RUNTIME: WARNING: external llvm.func trampoline sqrt has no compatible native fallback; returning zeros
// RUNTIME: bits=0

llvm.func @sqrt(f64) -> f64

func.func @call_sqrt_bits(%x: i64) -> i64 {
  %xf = llvm.uitofp %x : i64 to f64
  %r = llvm.call @sqrt(%xf) : (f64) -> f64
  %bits = llvm.bitcast %r : f64 to i64
  return %bits : i64
}

hw.module @test() {
  %c4 = hw.constant 4 : i64
  %fmt = sim.fmt.literal "bits="
  %fmt_nl = sim.fmt.literal "\0A"
  %s = llhd.sig %c4 : i64

  llhd.process {
    %x = llhd.prb %s : i64
    %bits = func.call @call_sqrt_bits(%x) : (i64) -> i64
    %bits_fmt = sim.fmt.dec %bits : i64
    %out = sim.fmt.concat (%fmt, %bits_fmt, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
