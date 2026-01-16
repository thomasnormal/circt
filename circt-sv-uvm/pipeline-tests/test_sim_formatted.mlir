// Test sim.fmt operations with arcilator JIT
func.func @entry() {
  %c42 = arith.constant 42 : i32
  %c255 = arith.constant 255 : i32

  // Test literal
  %lit1 = sim.fmt.literal "Testing formatting:\n"
  sim.proc.print %lit1

  // Test decimal formatting
  %dec = sim.fmt.dec %c42 signed : i32
  %lit2 = sim.fmt.literal "Decimal value: "
  %newline = sim.fmt.literal "\n"
  %msg1 = sim.fmt.concat (%lit2, %dec, %newline)
  sim.proc.print %msg1

  // Test hex formatting (note the comma and isUpper attribute)
  %hex = sim.fmt.hex %c255, isUpper false : i32
  %lit3 = sim.fmt.literal "Hex value: "
  %msg2 = sim.fmt.concat (%lit3, %hex, %newline)
  sim.proc.print %msg2

  // Test binary formatting
  %bin = sim.fmt.bin %c42 : i32
  %lit4 = sim.fmt.literal "Binary value: "
  %msg3 = sim.fmt.concat (%lit4, %bin, %newline)
  sim.proc.print %msg3

  return
}
