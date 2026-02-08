// RUN: circt-sim %s | FileCheck %s

// Test that struct field drives via llhd.sig.struct_extract correctly update
// pendingEpsilonDrives so that a subsequent probe in the same process sees
// the accumulated writes. This is the pattern used for:
//   s.a = 10; s.b = 20; $display(s.a, s.b);
// Without the fix, the display would show 0,0 because the epsilon drives
// were not accumulated in pendingEpsilonDrives.

// CHECK: [circt-sim] Starting simulation
// CHECK: a=10 b=20
// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  %c0_i32 = arith.constant 0 : i32
  %c10_i32 = arith.constant 10 : i32
  %c20_i32 = arith.constant 20 : i32
  %struct_init = hw.struct_create (%c0_i32, %c0_i32) : !hw.struct<a: i32, b: i32>
  %sig = llhd.sig %struct_init : !hw.struct<a: i32, b: i32>

  %fmt_a = sim.fmt.literal "a="
  %fmt_b = sim.fmt.literal " b="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %time = llhd.constant_time <0ns, 0d, 1e>
    %c1000000_i64 = hw.constant 1000000 : i64
    %delay = llhd.int_to_time %c1000000_i64

    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Drive struct field "a" with epsilon delay (blocking assignment semantics)
    %ref_a = llhd.sig.struct_extract %sig["a"] : <!hw.struct<a: i32, b: i32>>
    llhd.drv %ref_a, %c10_i32 after %time : i32

    // Drive struct field "b" with epsilon delay
    %ref_b = llhd.sig.struct_extract %sig["b"] : <!hw.struct<a: i32, b: i32>>
    llhd.drv %ref_b, %c20_i32 after %time : i32

    // Probe the full struct - should see accumulated writes {10, 20}
    %val = llhd.prb %sig : !hw.struct<a: i32, b: i32>

    %va = hw.struct_extract %val["a"] : !hw.struct<a: i32, b: i32>
    %vb = hw.struct_extract %val["b"] : !hw.struct<a: i32, b: i32>

    %fa = sim.fmt.dec %va : i32
    %fb = sim.fmt.dec %vb : i32
    %out = sim.fmt.concat (%fmt_a, %fa, %fmt_b, %fb, %fmt_nl)
    sim.proc.print %out

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
