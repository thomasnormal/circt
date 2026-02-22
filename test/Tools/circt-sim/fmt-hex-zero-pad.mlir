// RUN: circt-sim %s | FileCheck %s

// Regression: sim.fmt.hex with specifierWidth must zero-pad the result.
// For example, specifierWidth 8 on an 8-bit value 0x0A should produce
// "0000000a", not "a".
//
// The bug: the hex formatter ignored specifierWidth and returned the
// minimal representation (no leading zeros).
//
// Fix: after formatting, if result.size() < specifierWidth, prepend zeros.

// CHECK: no_pad=a
// CHECK: pad8=0000000a
// CHECK: pad4=000a
// CHECK: pad2=0a
// CHECK: upper_pad4=000A
// CHECK: full=ff
// CHECK: wide_pad6=0000ff

module {
  hw.module @test() {
    %val_0a = hw.constant 10 : i8
    %val_ff = hw.constant 255 : i8

    %lbl_nopad = sim.fmt.literal "no_pad="
    %lbl_pad8 = sim.fmt.literal "pad8="
    %lbl_pad4 = sim.fmt.literal "pad4="
    %lbl_pad2 = sim.fmt.literal "pad2="
    %lbl_upper = sim.fmt.literal "upper_pad4="
    %lbl_full = sim.fmt.literal "full="
    %lbl_wide = sim.fmt.literal "wide_pad6="
    %nl = sim.fmt.literal "\0A"

    llhd.process {
      // No padding: should just be "a"
      %hex_nopad = sim.fmt.hex %val_0a, isUpper false : i8
      %out0 = sim.fmt.concat (%lbl_nopad, %hex_nopad, %nl)
      sim.proc.print %out0

      // Pad to width 8: "0000000a"
      %hex_pad8 = sim.fmt.hex %val_0a, isUpper false specifierWidth 8 : i8
      %out1 = sim.fmt.concat (%lbl_pad8, %hex_pad8, %nl)
      sim.proc.print %out1

      // Pad to width 4: "000a"
      %hex_pad4 = sim.fmt.hex %val_0a, isUpper false specifierWidth 4 : i8
      %out2 = sim.fmt.concat (%lbl_pad4, %hex_pad4, %nl)
      sim.proc.print %out2

      // Pad to width 2: "0a"
      %hex_pad2 = sim.fmt.hex %val_0a, isUpper false specifierWidth 2 : i8
      %out3 = sim.fmt.concat (%lbl_pad2, %hex_pad2, %nl)
      sim.proc.print %out3

      // Upper case with padding: "000A"
      %hex_upper = sim.fmt.hex %val_0a, isUpper true specifierWidth 4 : i8
      %out4 = sim.fmt.concat (%lbl_upper, %hex_upper, %nl)
      sim.proc.print %out4

      // Value fills width exactly (ff, width 2): "ff"
      %hex_full = sim.fmt.hex %val_ff, isUpper false specifierWidth 2 : i8
      %out5 = sim.fmt.concat (%lbl_full, %hex_full, %nl)
      sim.proc.print %out5

      // Wider padding on ff (width 6): "0000ff"
      %hex_wide = sim.fmt.hex %val_ff, isUpper false specifierWidth 6 : i8
      %out6 = sim.fmt.concat (%lbl_wide, %hex_wide, %nl)
      sim.proc.print %out6

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
