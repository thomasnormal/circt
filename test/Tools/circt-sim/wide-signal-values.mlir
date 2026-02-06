// RUN: circt-sim %s | FileCheck %s

// Test that wide signal values (> 64 bits) are handled correctly by truncating
// to 64 bits for the SignalValue representation.

// This tests the fix for wide APInt values that would previously cause an
// assertion failure: "Too many bits for uint64_t"

// CHECK: [circt-sim] Starting simulation
// CHECK: Wide signal test: initialized
// CHECK: i128 signal LSB: 1
// CHECK: i256 signal LSB: 1
// CHECK: Wide signal test: completed
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c10000000_i64 = hw.constant 10000000 : i64  // 10 time units
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Wide signal values - these should be truncated to 64 bits in SignalValue
  // but still work without crashing
  // 0x123456789ABCDEF0_123456789ABCDEF1 as decimal = 24197857203266734881049227610739797745
  %c_wide_128 = hw.constant 24197857203266734881049227610739797745 : i128
  // A smaller 256-bit value
  %c_wide_256 = hw.constant 1311768467294899695 : i256

  // Create signals with wide initial values
  %sig_128 = llhd.sig %c_wide_128 : i128
  %sig_256 = llhd.sig %c_wide_256 : i256

  %fmt_init = sim.fmt.literal "Wide signal test: initialized\0A"
  %fmt_done = sim.fmt.literal "Wide signal test: completed\0A"
  %fmt_128_prefix = sim.fmt.literal "i128 signal LSB: "
  %fmt_256_prefix = sim.fmt.literal "i256 signal LSB: "
  %fmt_nl = sim.fmt.literal "\0A"

  // Process that probes and verifies the wide signals
  llhd.process {
    cf.br ^bb1
  ^bb1:
    sim.proc.print %fmt_init

    // Probe wide signals - these should work even though they're truncated
    %val_128 = llhd.prb %sig_128 : i128
    %val_256 = llhd.prb %sig_256 : i256

    // Extract LSB to verify the value is accessible
    %lsb_128 = comb.extract %val_128 from 0 : (i128) -> i1
    %lsb_256 = comb.extract %val_256 from 0 : (i256) -> i1

    // Print the LSB values
    %fmt_lsb_128 = sim.fmt.bin %lsb_128 : i1
    %fmt_str_128 = sim.fmt.concat (%fmt_128_prefix, %fmt_lsb_128, %fmt_nl)
    sim.proc.print %fmt_str_128

    %fmt_lsb_256 = sim.fmt.bin %lsb_256 : i1
    %fmt_str_256 = sim.fmt.concat (%fmt_256_prefix, %fmt_lsb_256, %fmt_nl)
    sim.proc.print %fmt_str_256

    sim.proc.print %fmt_done

    // Terminate
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
