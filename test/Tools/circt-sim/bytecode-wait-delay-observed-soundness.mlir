// RUN: env CIRCT_SIM_COMPILE_REPORT=1 circt-sim %s --max-time=3000000 --sim-stats 2>&1 | FileCheck %s

// This process shape mixes a delay wait and an observed wait in one process.
// Bytecode must reject it and run it in the generic interpreter to preserve
// wait semantics and reset propagation.

// CHECK-DAG: [Bytecode Stats] 0/3 processes compiled to bytecode
// CHECK-DAG: llhd.wait(delay):
// CHECK-DAG: RST_RISE

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1000000_i64 = hw.constant 1000000 : i64
  %c3000000_i64 = hw.constant 3000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %fmt_rise = sim.fmt.literal "RST_RISE\0A"

  %rst = llhd.sig %false : i1

  llhd.process {
    llhd.drv %rst, %false after %eps : i1
    %delay = llhd.int_to_time %c1000000_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %rst, %true after %eps : i1
    %old = llhd.prb %rst : i1
    llhd.wait (%old : i1), ^bb2
  ^bb2:
    llhd.halt
  }

  llhd.process {
    cf.br ^wait
  ^wait:
    %old = llhd.prb %rst : i1
    llhd.wait (%old : i1), ^check(%old : i1)
  ^check(%prev: i1):
    %new = llhd.prb %rst : i1
    %prev_not = comb.xor %prev, %true : i1
    %rising = comb.and %prev_not, %new : i1
    cf.cond_br %rising, ^print, ^wait
  ^print:
    sim.proc.print %fmt_rise
    llhd.halt
  }

  llhd.process {
    %t = llhd.int_to_time %c3000000_i64
    llhd.wait delay %t, ^term
  ^term:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
