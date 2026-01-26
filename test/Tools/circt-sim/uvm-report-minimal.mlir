// RUN: circt-sim %s --max-time 1000000 2>&1 | FileCheck %s
// CHECK: Found 1 LLHD processes
// CHECK: value=1

module {
  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    %false = hw.constant false
    %true = hw.constant true
    %eps = llhd.constant_time <0ns, 0d, 1e>

    // Need a signal to prevent the process from being canonicalized away
    %sig = llhd.sig %false : i1

    llhd.process {
      cf.br ^start
    ^start:
      // Drive the signal to keep the process alive
      llhd.drv %sig, %true after %eps : i1
      %fmt_lit = sim.fmt.literal "value="
      %fmt_val = sim.fmt.dec %true : i1
      %fmt_nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%fmt_lit, %fmt_val, %fmt_nl)
      sim.proc.print %fmt
      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }

    hw.output
  }
}
