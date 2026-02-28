// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so --aot-stats 2>&1 | FileCheck %s --check-prefix=STATS
//
// Ensure callback process extraction is not blocked by sim.proc.print.
//
// COMPILE: [circt-compile] Compiled 1 process bodies
// COMPILE: [circt-compile] Processes: 2 total, 1 callback-eligible, 1 rejected
//
// SIM: tick
//
// STATS: Compiled callback invocations: {{[1-9][0-9]*}}
// STATS: tick

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %fmt_tick = sim.fmt.literal "tick\0A"
  %sig = llhd.sig %false : i1

  // Callback process: 1 wait + sim.proc.print.
  llhd.process {
    %d = llhd.int_to_time %c1_i64
    llhd.wait delay %d, ^emit
  ^emit:
    %v = llhd.prb %sig : i1
    %n = comb.xor %v, %true : i1
    llhd.drv %sig, %n after %eps : i1
    sim.proc.print %fmt_tick
    llhd.halt
  }

  // Terminator process (kept unsupported so only one process is compiled).
  llhd.process {
    %d = llhd.int_to_time %c2_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
