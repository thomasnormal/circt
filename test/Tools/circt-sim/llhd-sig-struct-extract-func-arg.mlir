// RUN: circt-sim %s | FileCheck %s

// Test that llhd.sig.struct_extract correctly works when the parent ref is
// passed as a function argument. This is the pattern used by JTAG AVIP's
// fromClass_2714 function where a struct reference from a GEP+cast is passed
// to a callee that does sig.struct_extract + drv/prb on it.
//
// The interpreter's SigStructExtractOp handler must propagate the memory
// address from the parent ref so that the drv/prb fallback paths can find
// the memory block via findMemoryBlockByAddress.

// CHECK: [circt-sim] Starting simulation
// CHECK: Before call: a=0, b=0
// CHECK: Inside func: read a=0
// CHECK: After call: a=77, b=0
// CHECK: [circt-sim] Simulation completed

module {
  // Function that takes a struct ref as argument, extracts field "a", and
  // drives it to a new value. Also probes field "a" before driving.
  llvm.func @drive_field_a(%ref_ptr: !llvm.ptr, %new_val: i32) {
    %ref = builtin.unrealized_conversion_cast %ref_ptr : !llvm.ptr to !llhd.ref<!hw.struct<a: i32, b: i32>>

    // Probe field "a" through sig.struct_extract
    %a_ref = llhd.sig.struct_extract %ref["a"] : <!hw.struct<a: i32, b: i32>>
    %a_val = llhd.prb %a_ref : i32

    // Print probed value
    %fmt_pre = sim.fmt.literal "Inside func: read a="
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_a = sim.fmt.dec %a_val : i32
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_a, %fmt_nl)
    sim.proc.print %fmt_out

    // Drive field "a" to the new value
    %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>
    llhd.drv %a_ref, %new_val after %time0 : i32

    llvm.return
  }

  hw.module @test() {
    %c1_i64 = llvm.mlir.constant(1 : i64) : i64
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %c77_i32 = llvm.mlir.constant(77 : i32) : i32
    %c1000000_i64 = hw.constant 1000000 : i64

    %fmt_before = sim.fmt.literal "Before call: a="
    %fmt_after = sim.fmt.literal "After call: a="
    %fmt_comma_b = sim.fmt.literal ", b="
    %fmt_nl = sim.fmt.literal "\0A"

    %time0 = llhd.constant_time #llhd.time<0ns, 0d, 0e>

    llhd.process {
      %delay = llhd.int_to_time %c1000000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      // Allocate a struct with two i32 fields
      %local_struct = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr

      // Initialize to zeros
      %undef = llvm.mlir.undef : !llvm.struct<(i32, i32)>
      %s0 = llvm.insertvalue %c0_i32, %undef[0] : !llvm.struct<(i32, i32)>
      %s1 = llvm.insertvalue %c0_i32, %s0[1] : !llvm.struct<(i32, i32)>
      llvm.store %s1, %local_struct : !llvm.struct<(i32, i32)>, !llvm.ptr

      // Cast to !llhd.ref
      %ref = builtin.unrealized_conversion_cast %local_struct : !llvm.ptr to !llhd.ref<!hw.struct<a: i32, b: i32>>

      // Print initial values
      %val0 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
      %a0 = hw.struct_extract %val0["a"] : !hw.struct<a: i32, b: i32>
      %b0 = hw.struct_extract %val0["b"] : !hw.struct<a: i32, b: i32>
      %fmt_a0 = sim.fmt.dec %a0 : i32
      %fmt_b0 = sim.fmt.dec %b0 : i32
      %fmt_out0 = sim.fmt.concat (%fmt_before, %fmt_a0, %fmt_comma_b, %fmt_b0, %fmt_nl)
      sim.proc.print %fmt_out0

      // Call the function, passing the struct ref as a pointer argument
      %ref_ptr = builtin.unrealized_conversion_cast %ref : !llhd.ref<!hw.struct<a: i32, b: i32>> to !llvm.ptr
      llvm.call @drive_field_a(%ref_ptr, %c77_i32) : (!llvm.ptr, i32) -> ()

      // Print values after the call - field "a" should be 77
      %val1 = llhd.prb %ref : !hw.struct<a: i32, b: i32>
      %a1 = hw.struct_extract %val1["a"] : !hw.struct<a: i32, b: i32>
      %b1 = hw.struct_extract %val1["b"] : !hw.struct<a: i32, b: i32>
      %fmt_a1 = sim.fmt.dec %a1 : i32
      %fmt_b1 = sim.fmt.dec %b1 : i32
      %fmt_out1 = sim.fmt.concat (%fmt_after, %fmt_a1, %fmt_comma_b, %fmt_b1, %fmt_nl)
      sim.proc.print %fmt_out1

      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
