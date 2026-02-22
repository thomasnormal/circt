// RUN: circt-sim %s | FileCheck %s

// Test that llhd.drv to sig.extract correctly performs read-modify-write on
// the parent struct, preserving other fields while updating the target bits.
//
// This covers a fix where driving to a sig.extract ref on an alloca-backed
// struct would incorrectly skip the write when the drive value was X (unknown).
// The fix treats X as 0 (consistent with SigStructExtractOp) rather than
// silently skipping.
//
// Pattern tested:
//   %alloca = llvm.alloca -> unrealized_cast to !llhd.ref<struct>
//   %field = llhd.sig.extract %ref from %offset : <iN> -> <i1>
//   llhd.drv %field, %val
//
// This test verifies the read-modify-write behavior at various bit offsets
// within a single integer value.

// CHECK: [circt-sim] Starting simulation
// CHECK: Initial: 0
// CHECK: After bit 0: 1
// CHECK: After bit 3: 9
// CHECK: After bit 7: 137
// CHECK: After clear bit 0: 136
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %c1000000_i64 = hw.constant 1000000 : i64
  %true = hw.constant true
  %false = hw.constant false

  %fmt_init = sim.fmt.literal "Initial: "
  %fmt_b0 = sim.fmt.literal "After bit 0: "
  %fmt_b3 = sim.fmt.literal "After bit 3: "
  %fmt_b7 = sim.fmt.literal "After bit 7: "
  %fmt_clr = sim.fmt.literal "After clear bit 0: "
  %fmt_nl = sim.fmt.literal "\0A"

  %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>

  llhd.process {
    %delay = llhd.int_to_time %c1000000_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Allocate a single i32 field
    %obj = llvm.alloca %c1_i64 x !llvm.struct<(i32)> : (i64) -> !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(i32)>
    %s0 = llvm.insertvalue %c0_i32, %undef[0] : !llvm.struct<(i32)>
    llvm.store %s0, %obj : !llvm.struct<(i32)>, !llvm.ptr

    // Get pointer to the i32 field via GEP
    %field_ptr = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>

    // Cast to LLHD ref for sig.extract
    %ref = builtin.unrealized_conversion_cast %field_ptr : !llvm.ptr to !llhd.ref<i32>

    // Print initial value: should be 0
    %v0 = llvm.load %field_ptr : !llvm.ptr -> i32
    %fmt_v0 = sim.fmt.dec %v0 : i32
    %fmt_out0 = sim.fmt.concat (%fmt_init, %fmt_v0, %fmt_nl)
    sim.proc.print %fmt_out0

    // Drive bit 0 = 1 via sig.extract
    %c0_offset = hw.constant 0 : i5
    %bit0_ref = llhd.sig.extract %ref from %c0_offset : <i32> -> <i1>
    llhd.drv %bit0_ref, %true after %time0 : i1

    // Should be 1 (0b00000001)
    %v1 = llvm.load %field_ptr : !llvm.ptr -> i32
    %fmt_v1 = sim.fmt.dec %v1 : i32
    %fmt_out1 = sim.fmt.concat (%fmt_b0, %fmt_v1, %fmt_nl)
    sim.proc.print %fmt_out1

    // Drive bit 3 = 1 via sig.extract (value should become 9 = 0b00001001)
    %c3_offset = hw.constant 3 : i5
    %bit3_ref = llhd.sig.extract %ref from %c3_offset : <i32> -> <i1>
    llhd.drv %bit3_ref, %true after %time0 : i1

    %v2 = llvm.load %field_ptr : !llvm.ptr -> i32
    %fmt_v2 = sim.fmt.dec %v2 : i32
    %fmt_out2 = sim.fmt.concat (%fmt_b3, %fmt_v2, %fmt_nl)
    sim.proc.print %fmt_out2

    // Drive bit 7 = 1 (value should become 137 = 0b10001001)
    %c7_offset = hw.constant 7 : i5
    %bit7_ref = llhd.sig.extract %ref from %c7_offset : <i32> -> <i1>
    llhd.drv %bit7_ref, %true after %time0 : i1

    %v3 = llvm.load %field_ptr : !llvm.ptr -> i32
    %fmt_v3 = sim.fmt.dec %v3 : i32
    %fmt_out3 = sim.fmt.concat (%fmt_b7, %fmt_v3, %fmt_nl)
    sim.proc.print %fmt_out3

    // Clear bit 0 back to 0 (value should become 136 = 0b10001000)
    llhd.drv %bit0_ref, %false after %time0 : i1

    %v4 = llvm.load %field_ptr : !llvm.ptr -> i32
    %fmt_v4 = sim.fmt.dec %v4 : i32
    %fmt_out4 = sim.fmt.concat (%fmt_clr, %fmt_v4, %fmt_nl)
    sim.proc.print %fmt_out4

    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
