// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_TRACE_COMPILED_PROCESSES=1 \
// RUN:   circt-sim %s --compiled=%t.so --max-time=12000000 --aot-stats 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TRACE
//
// Regression: child-module llhd.process registration must classify callback
// models too. If instance processes skip classification, process traces show
// model=<missing> and time-only callbacks can fail to rearm after first fire.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
//
// TRACE: compiled-proc name=child.process_0
// TRACE-SAME: kind=CALLBACK model=CallbackTimeOnly
// TRACE: Compiled callback invocations: {{[2-9][0-9]*}}

hw.module private @child(out out : i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c5_i64 = hw.constant 5000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %clk = llhd.sig %false : i1

  // Time-only callback candidate: loops with fixed delay and a simple drive.
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
  %inst.out = hw.instance "u" @child() -> (out: i1)
  hw.output
}
