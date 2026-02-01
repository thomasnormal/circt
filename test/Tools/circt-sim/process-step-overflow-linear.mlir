// RUN: circt-sim --max-process-steps=5 %s | FileCheck %s

// CHECK: linear-ok

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %fmt = sim.fmt.literal "linear-ok\0A"

  llhd.process {
    %v0 = comb.xor %true, %false : i1
    %v1 = comb.xor %v0, %true : i1
    %v2 = comb.xor %v1, %false : i1
    %v3 = comb.xor %v2, %true : i1
    %v4 = comb.xor %v3, %false : i1
    %v5 = comb.xor %v4, %true : i1
    %v6 = comb.xor %v5, %false : i1
    %v7 = comb.xor %v6, %true : i1
    %v8 = comb.xor %v7, %false : i1
    %v9 = comb.xor %v8, %true : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
