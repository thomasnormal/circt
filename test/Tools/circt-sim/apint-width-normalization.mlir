// RUN: circt-sim %s | FileCheck %s

// Test that arithmetic operations correctly normalize APInt widths when operands
// have mismatched bit widths. This can occur when a value read from a signal
// (which may have a different width due to SignalValue limitations) is combined
// with a constant.

// This tests the fix for the assertion failure:
// "Assertion `BitWidth == RHS.BitWidth && "Bit widths must be the same"' failed."

// CHECK: [circt-sim] Starting simulation
// CHECK: Width normalization test
// CHECK: Counter initial value: 0
// CHECK: Counter after add: 1
// CHECK: Counter after add: 2
// CHECK: Compare result (2 == 2): 4294967295
// CHECK: AND result: 2
// CHECK: OR result: 3
// CHECK: XOR result: 1
// CHECK: Sub result: 1
// CHECK: Mul result: 4
// CHECK: Width normalization test: PASSED
// CHECK: [circt-sim] Simulation finished successfully

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Create a counter signal with i32 type
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  %c1000000_i64 = hw.constant 1000000 : i64

  // Signal to hold the counter value
  %counter = llhd.sig %c0_i32 : i32

  // Format strings
  %fmt_test = sim.fmt.literal "Width normalization test\0A"
  %fmt_init = sim.fmt.literal "Counter initial value: "
  %fmt_after_add = sim.fmt.literal "Counter after add: "
  %fmt_cmp = sim.fmt.literal "Compare result (2 == 2): "
  %fmt_and = sim.fmt.literal "AND result: "
  %fmt_or = sim.fmt.literal "OR result: "
  %fmt_xor = sim.fmt.literal "XOR result: "
  %fmt_sub = sim.fmt.literal "Sub result: "
  %fmt_mul = sim.fmt.literal "Mul result: "
  %fmt_pass = sim.fmt.literal "Width normalization test: PASSED\0A"
  %fmt_nl = sim.fmt.literal "\0A"

  // Process that tests arithmetic operations with potentially mismatched widths
  llhd.process {
    cf.br ^bb1
  ^bb1:
    sim.proc.print %fmt_test

    // Test 1: Probe and print initial value
    %val0 = llhd.prb %counter : i32
    %fmt_val0 = sim.fmt.dec %val0 : i32
    %fmt_str0 = sim.fmt.concat (%fmt_init, %fmt_val0, %fmt_nl)
    sim.proc.print %fmt_str0

    // Test 2: Add 1 to the probed value (this is where the width mismatch can occur)
    // The probed value might have a different internal width than the constant
    %val1 = comb.add %val0, %c1_i32 : i32

    // Drive the new value
    llhd.drv %counter, %val1 after %eps : i32

    // Wait for the drive to take effect
    %delay = llhd.int_to_time %c1000000_i64
    llhd.wait delay %delay, ^bb2

  ^bb2:
    // Print the updated value
    %val2 = llhd.prb %counter : i32
    %fmt_val2 = sim.fmt.dec %val2 : i32
    %fmt_str2 = sim.fmt.concat (%fmt_after_add, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_str2

    // Test 3: Add again
    %val3 = comb.add %val2, %c1_i32 : i32
    llhd.drv %counter, %val3 after %eps : i32
    %delay2 = llhd.int_to_time %c1000000_i64
    llhd.wait delay %delay2, ^bb3

  ^bb3:
    %val4 = llhd.prb %counter : i32
    %fmt_val4 = sim.fmt.dec %val4 : i32
    %fmt_str4 = sim.fmt.concat (%fmt_after_add, %fmt_val4, %fmt_nl)
    sim.proc.print %fmt_str4

    // Test 4: Compare - probed value with constant
    %cmp_result = comb.icmp eq %val4, %c2_i32 : i32
    %cmp_ext = comb.replicate %cmp_result : (i1) -> i32
    %fmt_cmp_val = sim.fmt.dec %cmp_ext : i32
    %fmt_cmp_str = sim.fmt.concat (%fmt_cmp, %fmt_cmp_val, %fmt_nl)
    sim.proc.print %fmt_cmp_str

    // Test 5: AND operation
    %and_result = comb.and %val4, %c3_i32 : i32
    %fmt_and_val = sim.fmt.dec %and_result : i32
    %fmt_and_str = sim.fmt.concat (%fmt_and, %fmt_and_val, %fmt_nl)
    sim.proc.print %fmt_and_str

    // Test 6: OR operation
    %or_result = comb.or %val4, %c1_i32 : i32
    %fmt_or_val = sim.fmt.dec %or_result : i32
    %fmt_or_str = sim.fmt.concat (%fmt_or, %fmt_or_val, %fmt_nl)
    sim.proc.print %fmt_or_str

    // Test 7: XOR operation
    %xor_result = comb.xor %val4, %c3_i32 : i32
    %fmt_xor_val = sim.fmt.dec %xor_result : i32
    %fmt_xor_str = sim.fmt.concat (%fmt_xor, %fmt_xor_val, %fmt_nl)
    sim.proc.print %fmt_xor_str

    // Test 8: Sub operation
    %sub_result = comb.sub %val4, %c1_i32 : i32
    %fmt_sub_val = sim.fmt.dec %sub_result : i32
    %fmt_sub_str = sim.fmt.concat (%fmt_sub, %fmt_sub_val, %fmt_nl)
    sim.proc.print %fmt_sub_str

    // Test 9: Mul operation
    %mul_result = comb.mul %val4, %c2_i32 : i32
    %fmt_mul_val = sim.fmt.dec %mul_result : i32
    %fmt_mul_str = sim.fmt.concat (%fmt_mul, %fmt_mul_val, %fmt_nl)
    sim.proc.print %fmt_mul_str

    // All tests passed
    sim.proc.print %fmt_pass

    // Terminate
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
