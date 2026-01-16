// RUN: arcilator %s --run | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// CHECK: value = {{[ ]*}}42
// CHECK: hex = {{0*}}2a
// CHECK: Hello, World!

func.func @entry() {
  %val = arith.constant 42 : i32

  // Test decimal format
  %lit1 = sim.fmt.literal "value = "
  %dec = sim.fmt.dec %val : i32
  %nl1 = sim.fmt.literal "\0A"
  %fmt1 = sim.fmt.concat (%lit1, %dec, %nl1)
  sim.proc.print %fmt1

  // Test hex format
  %lit2 = sim.fmt.literal "hex = "
  %hex = sim.fmt.hex %val, isUpper false : i32
  %nl2 = sim.fmt.literal "\0A"
  %fmt2 = sim.fmt.concat (%lit2, %hex, %nl2)
  sim.proc.print %fmt2

  // Test literal only
  %lit3 = sim.fmt.literal "Hello, World!\0A"
  sim.proc.print %lit3

  return
}
