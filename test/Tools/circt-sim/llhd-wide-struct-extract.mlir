// RUN: circt-sim %s | FileCheck %s

// CHECK: wide_hi_ok=1

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c1_i64 = hw.constant 1000000 : i64
  %false = hw.constant false

  %hi = hw.constant 0x10000000000000000 : i65
  %lo = hw.constant 0 : i65
  %s = hw.struct_create (%hi, %lo) : !hw.struct<hi: i65, lo: i65>
  %hi_ex = hw.struct_extract %s["hi"] : !hw.struct<hi: i65, lo: i65>

  %sig = llhd.sig %lo : i65
  llhd.drv %sig, %hi_ex after %eps : i65

  %fmt_pre = sim.fmt.literal "wide_hi_ok="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %sig_val = llhd.prb %sig : i65
    %eq = comb.icmp eq %sig_val, %hi : i65
    %fmt_val = sim.fmt.dec %eq : i1
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.halt
  }

  hw.output
}
