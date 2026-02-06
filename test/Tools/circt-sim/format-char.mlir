// RUN: circt-sim %s | FileCheck %s

// CHECK: char_A=A
// CHECK: char_z=z
// CHECK: char_0=0

hw.module @test() {
  // ASCII 65 = 'A', 122 = 'z', 48 = '0'
  %val_A = hw.constant 65 : i8
  %val_z = hw.constant 122 : i8
  %val_0 = hw.constant 48 : i8

  %lbl_A = sim.fmt.literal "char_A="
  %lbl_z = sim.fmt.literal "char_z="
  %lbl_0 = sim.fmt.literal "char_0="
  %nl = sim.fmt.literal "\0A"

  llhd.process {
    %fmt_A = sim.fmt.char %val_A : i8
    %fmt_z = sim.fmt.char %val_z : i8
    %fmt_0 = sim.fmt.char %val_0 : i8

    %out_A = sim.fmt.concat (%lbl_A, %fmt_A, %nl)
    %out_z = sim.fmt.concat (%lbl_z, %fmt_z, %nl)
    %out_0 = sim.fmt.concat (%lbl_0, %fmt_0, %nl)

    sim.proc.print %out_A
    sim.proc.print %out_z
    sim.proc.print %out_0

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
