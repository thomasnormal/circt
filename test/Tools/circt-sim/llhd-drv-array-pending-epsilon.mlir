// RUN: circt-sim %s | FileCheck %s

// Test that array element drives via llhd.sig.array_get correctly update
// pendingEpsilonDrives so that a subsequent probe in the same process sees
// the accumulated writes. This is the pattern used for:
//   b[0] = 0; b[1] = 1; b[2] = 2; $display(b[0], b[1], b[2]);
// Without the fix, the display would show 0,0,0 because the epsilon drives
// were not accumulated in pendingEpsilonDrives.

// CHECK: [circt-sim] Starting simulation
// CHECK: b0=0 b1=1 b2=2
// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  %c0_i8 = arith.constant 0 : i8
  %c1_i8 = arith.constant 1 : i8
  %c2_i8 = arith.constant 2 : i8
  %arr_init = hw.array_create %c0_i8, %c0_i8, %c0_i8 : i8
  %arr = llhd.sig %arr_init : !hw.array<3xi8>

  %fmt_b0 = sim.fmt.literal "b0="
  %fmt_b1 = sim.fmt.literal " b1="
  %fmt_b2 = sim.fmt.literal " b2="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %time = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = arith.constant 0 : i2
    %c1_i2 = arith.constant 1 : i2
    %c2_i2 = arith.constant 2 : i2
    %c1000000_i64 = hw.constant 1000000 : i64
    %delay = llhd.int_to_time %c1000000_i64

    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Drive array elements with epsilon delay (blocking assignment semantics)
    %ref0 = llhd.sig.array_get %arr[%c0_i2] : <!hw.array<3xi8>>
    llhd.drv %ref0, %c0_i8 after %time : i8

    %ref1 = llhd.sig.array_get %arr[%c1_i2] : <!hw.array<3xi8>>
    llhd.drv %ref1, %c1_i8 after %time : i8

    %ref2 = llhd.sig.array_get %arr[%c2_i2] : <!hw.array<3xi8>>
    llhd.drv %ref2, %c2_i8 after %time : i8

    // Probe the full array - should see accumulated writes {0,1,2}
    %val = llhd.prb %arr : !hw.array<3xi8>

    %v0 = hw.array_get %val[%c0_i2] : !hw.array<3xi8>, i2
    %v1 = hw.array_get %val[%c1_i2] : !hw.array<3xi8>, i2
    %v2 = hw.array_get %val[%c2_i2] : !hw.array<3xi8>, i2

    %f0 = sim.fmt.dec %v0 : i8
    %f1 = sim.fmt.dec %v1 : i8
    %f2 = sim.fmt.dec %v2 : i8
    %out = sim.fmt.concat (%fmt_b0, %f0, %fmt_b1, %f1, %fmt_b2, %f2, %fmt_nl)
    sim.proc.print %out

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
