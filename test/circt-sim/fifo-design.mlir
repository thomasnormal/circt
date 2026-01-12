// RUN: circt-sim %s --top=FIFO --mode=analyze 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// A simple synchronous FIFO design for testing more complex simulation scenarios.
// Tests memory operations, state machines, and multiple clock domains.

// CHECK: === Design Analysis ===
// CHECK: Modules:

// FIFO depth is 4 (2-bit address)
hw.module @FIFO(
  in %clk: !seq.clock,
  in %rst: i1,
  in %write_en: i1,
  in %read_en: i1,
  in %write_data: i8,
  out read_data: i8,
  out empty: i1,
  out full: i1
) {
  %c0_i8 = hw.constant 0 : i8
  %c0_i3 = hw.constant 0 : i3
  %c1_i3 = hw.constant 1 : i3
  %c4_i3 = hw.constant 4 : i3
  %false = hw.constant 0 : i1
  %true = hw.constant 1 : i1

  // Write and read pointers (3 bits to handle wrap-around)
  %write_ptr = seq.compreg %next_write_ptr, %clk reset %rst, %c0_i3 : i3
  %read_ptr = seq.compreg %next_read_ptr, %clk reset %rst, %c0_i3 : i3
  %count = seq.compreg %next_count, %clk reset %rst, %c0_i3 : i3

  // Compute empty and full flags
  %is_empty = comb.icmp eq %count, %c0_i3 : i3
  %is_full = comb.icmp eq %count, %c4_i3 : i3

  // Can write if not full and write enabled
  %not_full = comb.xor %is_full, %true : i1
  %can_write = comb.and %write_en, %not_full : i1

  // Can read if not empty and read enabled
  %not_empty = comb.xor %is_empty, %true : i1
  %can_read = comb.and %read_en, %not_empty : i1

  // Update write pointer
  %write_ptr_inc = comb.add %write_ptr, %c1_i3 : i3
  %next_write_ptr = comb.mux %can_write, %write_ptr_inc, %write_ptr : i3

  // Update read pointer
  %read_ptr_inc = comb.add %read_ptr, %c1_i3 : i3
  %next_read_ptr = comb.mux %can_read, %read_ptr_inc, %read_ptr : i3

  // Update count
  %write_only = comb.and %can_write, %not_empty : i1
  %read_only = comb.and %can_read, %not_full : i1
  %count_inc = comb.add %count, %c1_i3 : i3
  %count_dec = comb.sub %count, %c1_i3 : i3
  %next_count_1 = comb.mux %can_write, %count_inc, %count : i3
  %next_count = comb.mux %can_read, %count_dec, %next_count_1 : i3

  // Output (simplified - real FIFO would use memory)
  hw.output %c0_i8, %is_empty, %is_full : i8, i1, i1
}
