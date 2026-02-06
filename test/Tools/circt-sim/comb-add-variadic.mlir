// RUN: circt-sim %s | FileCheck %s

// Test variadic comb.add with 3+ operands.
// Canonicalization merges nested binary adds into a single N-ary add:
//   comb.add(comb.add(%a, %b), %c) -> comb.add(%a, %b, %c)
// The interpreter must handle all operands, not just the first two.

// CHECK: sum=30

func.func private @"add10"(%arg0: i32) -> i32 {
  %c10 = hw.constant 10 : i32
  %result = comb.add %arg0, %c10 : i32
  return %result : i32
}

hw.module @test() {
  llhd.process {
    %c0 = hw.constant 0 : i32
    %r1 = func.call @"add10"(%c0) : (i32) -> i32
    %r2 = func.call @"add10"(%c0) : (i32) -> i32
    %r3 = func.call @"add10"(%c0) : (i32) -> i32

    // After canonicalization, this becomes: comb.add %r1, %r2, %r3 : i32
    %sum12 = comb.add %r1, %r2 : i32
    %sum = comb.add %sum12, %r3 : i32

    %fmt_prefix = sim.fmt.literal "sum="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_val = sim.fmt.dec %sum : i32
    %out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
