// XFAIL: *
// RUN: circt-sim %s | FileCheck %s
// GreedyPatternRewriteDriver OOMs during pass pipeline on this IR.

// Test for the DAG false-cycle-detection fix in evaluateContinuousValueImpl.
//
// The old code used an `inProgress` DenseSet to detect combinational cycles.
// When a shared node was referenced by two different consumers (a diamond DAG),
// the second reference was falsely treated as a cycle and the value was
// replaced with X.  The fix uses a `pushCount` DenseMap that allows each value
// to be pushed up to 2 times on the evaluation stack, which correctly handles
// shared nodes while still detecting true cycles.
//
// This test creates a diamond-shaped DAG:
//
//          a_val (probed signal, value = 5)
//          /   \
//      add(a,3) and(a,0xF)
//          \   /
//         or(left, right)   -->  result driven to %out
//
// Expected: left  = 5 + 3 = 8  (0b1000)
//           right = 5 & 15 = 5  (0b0101)
//           out   = 8 | 5 = 13

// CHECK: result=13

hw.module @test() {
  %c5_i8 = hw.constant 5 : i8
  %c3_i8 = hw.constant 3 : i8
  %c15_i8 = hw.constant 15 : i8
  %c0_i8 = hw.constant 0 : i8
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Source signal initialised to 5.
  %a = llhd.sig %c5_i8 : i8

  // Output signal.
  %out = llhd.sig %c0_i8 : i8

  // Probe the source -- this is the shared node in the DAG.
  %a_val = llhd.prb %a : i8

  // Two different consumers of the same probed value.
  %left = comb.add %a_val, %c3_i8 : i8    // 5 + 3 = 8
  %right = comb.and %a_val, %c15_i8 : i8  // 5 & 15 = 5

  // Combine the two branches.
  %result = comb.or %left, %right : i8     // 8 | 5 = 13

  // Drive the output continuously.
  llhd.drv %out, %result after %eps : i8

  // Print the result after two epsilon steps (the continuous drive needs one
  // epsilon to propagate, and we need to wait past that).
  llhd.process {
    llhd.wait delay %eps, ^bb1
  ^bb1:
    llhd.wait delay %eps, ^bb2
  ^bb2:
    %out_val = llhd.prb %out : i8
    %fmt_pre = sim.fmt.literal "result="
    %fmt_val = sim.fmt.dec %out_val : i8
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt
    llhd.halt
  }
}
