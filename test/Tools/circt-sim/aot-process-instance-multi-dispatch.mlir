// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s --compiled=%t.so --max-time=12000000 --aot-stats 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS
//
// Regression: when the same child module is instantiated multiple times, each
// runtime process instance must be wired to the compiled process entry.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
//
// Two instances of @child each toggle once at 5ns and once at 10ns after the
// initial interpreted activation, so compiled callback invocations should be
// at least 4.
// STATS: Compiled callback invocations: {{[4-9][0-9]*}}

hw.module private @child(out out : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c5_i64 = hw.constant 5000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %clk = llhd.sig %false : i1

  llhd.process {
    cf.br ^bb_wait
  ^bb_wait:
    %d = llhd.int_to_time %c5_i64
    llhd.wait delay %d, ^bb_tick
  ^bb_tick:
    %v = llhd.prb %clk : i1
    %n = comb.xor %v, %true : i1
    llhd.drv %clk, %n after %eps : i1
    cf.br ^bb_wait
  }

  %out_v = llhd.prb %clk : i1
  hw.output %out_v : i1
}

hw.module @test() {
  %inst0.out = hw.instance "u0" @child() -> (out: i1)
  %inst1.out = hw.instance "u1" @child() -> (out: i1)
  hw.output
}
