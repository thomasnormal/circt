// RUN: arcilator %s --run | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// CHECK: value = {{[ ]*}}42
// CHECK: hex = {{0*}}2a
// CHECK: Hello, World!
// CHECK: Counter: 1
// CHECK: Counter: 2
// CHECK: Counter: 3
// CHECK: Char: A

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

  // Test dynamic values with arc model
  %zero = arith.constant 0 : i1
  %one = arith.constant 1 : i1
  %lb = arith.constant 0 : index
  %ub = arith.constant 3 : index
  %step = arith.constant 1 : index

  arc.sim.instantiate @Counter as %model {
    scf.for %i = %lb to %ub step %step {
      arc.sim.set_input %model, "clk" = %one : i1, !arc.sim.instance<@Counter>
      arc.sim.step %model : !arc.sim.instance<@Counter>
      arc.sim.set_input %model, "clk" = %zero : i1, !arc.sim.instance<@Counter>
      arc.sim.step %model : !arc.sim.instance<@Counter>

      %counter_val = arc.sim.get_port %model, "counter" : i8, !arc.sim.instance<@Counter>

      // Use sim.proc.print with dynamic value
      %lit = sim.fmt.literal "Counter: "
      %dec_dyn = sim.fmt.dec %counter_val : i8
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %dec_dyn, %nl)
      sim.proc.print %fmt
    }
  }

  // Test character format
  %char_val = arith.constant 65 : i8  // 'A'
  %lit_char = sim.fmt.literal "Char: "
  %c = sim.fmt.char %char_val : i8
  %nl_char = sim.fmt.literal "\0A"
  %fmt_char = sim.fmt.concat (%lit_char, %c, %nl_char)
  sim.proc.print %fmt_char

  return
}

// Counter module for dynamic value testing
hw.module @Counter(in %clk: i1, out counter: i8) {
  %seq_clk = seq.to_clock %clk
  %c1 = hw.constant 1 : i8
  %reg = seq.compreg %reg_next, %seq_clk : i8
  %reg_next = comb.add %reg, %c1 : i8
  hw.output %reg : i8
}
