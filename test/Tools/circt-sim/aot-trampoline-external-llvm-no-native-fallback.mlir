// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME
//
// Regression: compiled trampoline dispatch for external llvm.func declarations
// must not crash when no native fallback symbol exists.
//
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
//
// RUNTIME: WARNING: trampoline 0 'missing_fn' is LLVM func but no native fallback
// RUNTIME: WARNING: external llvm.func trampoline missing_fn has no compatible native fallback; returning zeros
// RUNTIME-NOT: PLEASE submit a bug report
// RUNTIME-NOT: Stack dump
// RUNTIME: r=0

llvm.func @missing_fn(i64, i64) -> !llvm.struct<(i64, i64)>

func.func @call_missing_rem(%a: i64, %b: i64) -> i64 {
  %qr = llvm.call @missing_fn(%a, %b) : (i64, i64) -> !llvm.struct<(i64, i64)>
  %r = llvm.extractvalue %qr[1] : !llvm.struct<(i64, i64)>
  return %r : i64
}

hw.module @test() {
  %c10 = hw.constant 10 : i64
  %c3 = hw.constant 3 : i64
  %fmt = sim.fmt.literal "r="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %r = func.call @call_missing_rem(%c10, %c3) : (i64, i64) -> i64
    %rf = sim.fmt.dec %r : i64
    %out = sim.fmt.concat (%fmt, %rf, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
