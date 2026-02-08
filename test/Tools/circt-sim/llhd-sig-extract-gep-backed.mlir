// RUN: circt-sim %s | FileCheck %s

// Test that llhd.sig.extract correctly works on GEP-backed refs (class
// properties), not just alloca-backed refs. This is the pattern used by
// JTAG AVIP's JtagControllerDeviceCoverage::write function where a struct
// field is accessed via GEP+cast, then individual bits are extracted and
// driven using sig.extract.

// CHECK: [circt-sim] Starting simulation
// CHECK: Initial val: 0
// CHECK: After bit 3 set: 9
// CHECK: After bit 0 set: 9
// CHECK: [circt-sim] Simulation completed

module {
  // Function that takes a pointer to an i32 field (simulating a GEP into
  // a class property), casts to !llhd.ref<i32>, extracts and drives bits.
  llvm.func @drive_bits(%ptr: !llvm.ptr) {
    %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i32>
    %time = llhd.constant_time #llhd.time<0ns, 0d, 1e>
    %true = hw.constant true

    // Set bit 3
    %c3 = hw.constant 3 : i5
    %bit3_ref = llhd.sig.extract %ref from %c3 : <i32> -> <i1>
    llhd.drv %bit3_ref, %true after %time : i1

    // Set bit 0
    %c0 = hw.constant 0 : i5
    %bit0_ref = llhd.sig.extract %ref from %c0 : <i32> -> <i1>
    llhd.drv %bit0_ref, %true after %time : i1

    llvm.return
  }

  hw.module @test() {
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c1000000_i64 = hw.constant 1000000 : i64

    %fmt_init = sim.fmt.literal "Initial val: "
    %fmt_after3 = sim.fmt.literal "After bit 3 set: "
    %fmt_after0 = sim.fmt.literal "After bit 0 set: "
    %fmt_nl = sim.fmt.literal "\0A"

    %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>

    llhd.process {
      %delay = llhd.int_to_time %c1000000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      // Allocate a struct with one i32 field (simulating a class object)
      %obj = llvm.alloca %c1_i64 x !llvm.struct<(i32)> : (i64) -> !llvm.ptr
      %undef = llvm.mlir.undef : !llvm.struct<(i32)>
      %s0 = llvm.insertvalue %c0_i32, %undef[0] : !llvm.struct<(i32)>
      llvm.store %s0, %obj : !llvm.struct<(i32)>, !llvm.ptr

      // Get pointer to the field via GEP (like class property access)
      %field_ptr = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>

      // Print initial value
      %init_val = llvm.load %field_ptr : !llvm.ptr -> i32
      %fmt_iv = sim.fmt.dec %init_val : i32
      %fmt_out0 = sim.fmt.concat (%fmt_init, %fmt_iv, %fmt_nl)
      sim.proc.print %fmt_out0

      // Call function passing the GEP pointer (not alloca)
      llvm.call @drive_bits(%field_ptr) : (!llvm.ptr) -> ()

      // Print value after driving bit 3
      %val1 = llvm.load %field_ptr : !llvm.ptr -> i32
      %fmt_v1 = sim.fmt.dec %val1 : i32
      %fmt_out1 = sim.fmt.concat (%fmt_after3, %fmt_v1, %fmt_nl)
      sim.proc.print %fmt_out1

      // Also probe via sig.extract to verify probe works too
      %ref = builtin.unrealized_conversion_cast %field_ptr : !llvm.ptr to !llhd.ref<i32>
      %c0 = hw.constant 0 : i5
      %bit0_ref = llhd.sig.extract %ref from %c0 : <i32> -> <i1>
      %bit0_val = llhd.prb %bit0_ref : i1

      // Print final value
      %val2 = llvm.load %field_ptr : !llvm.ptr -> i32
      %fmt_v2 = sim.fmt.dec %val2 : i32
      %fmt_out2 = sim.fmt.concat (%fmt_after0, %fmt_v2, %fmt_nl)
      sim.proc.print %fmt_out2

      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
