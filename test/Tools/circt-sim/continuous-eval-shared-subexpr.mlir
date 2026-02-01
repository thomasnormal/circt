// RUN: circt-sim %s | FileCheck %s

// CHECK: b=0

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %a = llhd.sig %true : i1
  %b = llhd.sig %false : i1

  %a_val = llhd.prb %a : i1
  %v1 = comb.xor %a_val, %a_val : i1
  %v2 = comb.xor %v1, %v1 : i1
  %v3 = comb.xor %v2, %v2 : i1
  %v4 = comb.xor %v3, %v3 : i1
  %v5 = comb.xor %v4, %v4 : i1
  %v6 = comb.xor %v5, %v5 : i1
  %v7 = comb.xor %v6, %v6 : i1
  %v8 = comb.xor %v7, %v7 : i1
  %v9 = comb.xor %v8, %v8 : i1
  %v10 = comb.xor %v9, %v9 : i1
  %v11 = comb.xor %v10, %v10 : i1
  %v12 = comb.xor %v11, %v11 : i1
  %v13 = comb.xor %v12, %v12 : i1
  %v14 = comb.xor %v13, %v13 : i1
  %v15 = comb.xor %v14, %v14 : i1
  %v16 = comb.xor %v15, %v15 : i1
  %v17 = comb.xor %v16, %v16 : i1
  %v18 = comb.xor %v17, %v17 : i1
  %v19 = comb.xor %v18, %v18 : i1
  %v20 = comb.xor %v19, %v19 : i1

  llhd.drv %b, %v20 after %eps : i1

  llhd.process {
    llhd.wait delay %eps, ^bb1
  ^bb1:
    %b_val = llhd.prb %b : i1
    %fmt_b = sim.fmt.literal "b="
    %fmt_v = sim.fmt.dec %b_val : i1
    %fmt_nl = sim.fmt.literal "\0A"
    %out = sim.fmt.concat (%fmt_b, %fmt_v, %fmt_nl)
    sim.proc.print %out
    llhd.halt
  }
}
