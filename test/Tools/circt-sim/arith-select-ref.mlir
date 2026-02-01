// RUN: circt-sim %s | FileCheck %s

// Test that arith.select on !llhd.ref types is handled correctly in probe and drive.

// CHECK: probed_a=100
// CHECK: probed_b=200
// CHECK: after_drive_a=42
// CHECK: struct_b_unchanged=200

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %false = hw.constant false

  // Initial struct values
  %s_a = hw.aggregate_constant [100 : i32] : !hw.struct<x: i32>
  %s_b = hw.aggregate_constant [200 : i32] : !hw.struct<x: i32>

  // Create two struct signals
  %struct_a = llhd.sig %s_a : !hw.struct<x: i32>
  %struct_b = llhd.sig %s_b : !hw.struct<x: i32>

  // Format strings
  %fmt_probed_a = sim.fmt.literal "probed_a="
  %fmt_probed_b = sim.fmt.literal "probed_b="
  %fmt_after_drive = sim.fmt.literal "after_drive_a="
  %fmt_unchanged = sim.fmt.literal "struct_b_unchanged="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    cf.br ^init
  ^init:
    // Test 1: arith.select with condition=true should select struct_a
    %selected_ref_a = arith.select %true, %struct_a, %struct_b : !llhd.ref<!hw.struct<x: i32>>
    %probed_a = llhd.prb %selected_ref_a : !hw.struct<x: i32>
    %x_a = hw.struct_extract %probed_a["x"] : !hw.struct<x: i32>
    %fmt_x_a = sim.fmt.dec %x_a : i32
    %fmt_out_a = sim.fmt.concat (%fmt_probed_a, %fmt_x_a, %fmt_nl)
    sim.proc.print %fmt_out_a

    // Test 2: arith.select with condition=false should select struct_b
    %selected_ref_b = arith.select %false, %struct_a, %struct_b : !llhd.ref<!hw.struct<x: i32>>
    %probed_b = llhd.prb %selected_ref_b : !hw.struct<x: i32>
    %x_b = hw.struct_extract %probed_b["x"] : !hw.struct<x: i32>
    %fmt_x_b = sim.fmt.dec %x_b : i32
    %fmt_out_b = sim.fmt.concat (%fmt_probed_b, %fmt_x_b, %fmt_nl)
    sim.proc.print %fmt_out_b

    // Test 3: Drive through arith.select (should drive to struct_a since true)
    %new_x = hw.constant 42 : i32
    %new_struct = hw.struct_create (%new_x) : !hw.struct<x: i32>
    llhd.drv %selected_ref_a, %new_struct after %eps : !hw.struct<x: i32>

    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^check
  ^check:
    // Verify drive worked on struct_a
    %probed_a2 = llhd.prb %struct_a : !hw.struct<x: i32>
    %x_a2 = hw.struct_extract %probed_a2["x"] : !hw.struct<x: i32>
    %fmt_x_a2 = sim.fmt.dec %x_a2 : i32
    %fmt_out_a2 = sim.fmt.concat (%fmt_after_drive, %fmt_x_a2, %fmt_nl)
    sim.proc.print %fmt_out_a2

    // Verify struct_b was NOT modified
    %probed_b2 = llhd.prb %struct_b : !hw.struct<x: i32>
    %x_b2 = hw.struct_extract %probed_b2["x"] : !hw.struct<x: i32>
    %fmt_x_b2 = sim.fmt.dec %x_b2 : i32
    %fmt_out_b2 = sim.fmt.concat (%fmt_unchanged, %fmt_x_b2, %fmt_nl)
    sim.proc.print %fmt_out_b2

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
