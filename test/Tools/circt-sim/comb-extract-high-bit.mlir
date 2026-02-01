// RUN: circt-sim %s | FileCheck %s

// CHECK: bit0=0
// CHECK: bit64=1

hw.module @test() {
  %val = hw.constant 18446744073709551616 : i65
  %fmt_bit0 = sim.fmt.literal "bit0="
  %fmt_bit64 = sim.fmt.literal "bit64="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %b0 = comb.extract %val from 0 : (i65) -> i1
    %b64 = comb.extract %val from 64 : (i65) -> i1
    %b0_fmt = sim.fmt.dec %b0 : i1
    %b64_fmt = sim.fmt.dec %b64 : i1
    %out0 = sim.fmt.concat (%fmt_bit0, %b0_fmt, %fmt_nl)
    %out64 = sim.fmt.concat (%fmt_bit64, %b64_fmt, %fmt_nl)
    sim.proc.print %out0
    sim.proc.print %out64
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
